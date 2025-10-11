# app/routers/virtual.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Literal
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone

from app.indicators import add_indicators
from app.market_loader import load_ohlcv_15m
from app.virtual_engine import Variant, EntryRule, ExitRule, run_virtual_variants
from db import engine  # price_cache を読むだけなら不要でもOK

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

def _norm_symbol(s: str) -> str:
    s = s.strip().lower().replace("/", "_")
    return s

class VariantIn(BaseModel):
    id: str
    atr_min: Optional[float] = None
    adx_min: Optional[float] = None
    rsi_long_min: Optional[float] = None
    macd_cross: Optional[bool] = None
    ema_cross_req: Optional[bool] = None
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    max_holding_min: Optional[int] = None
    use_dynamic_levels: Optional[bool] = False
    trade_size_jpy: Optional[float] = 10_000
    cross_mode: Optional[Literal["any","all"]] = "any"

class SimReq(BaseModel):
    symbols: List[str] = Field(..., example=["xrp_jpy","eth_jpy"])
    timeframe: Literal["15m","5m","1h"] = "15m"   # ← ここを修正
    lookback_min: int = Field(720, ge=30, le=7*24*60)
    variants: List[VariantIn]

@router.post("/simulate")
def simulate(body: SimReq):
    # 1) シンボルの正規化
    symbols = [_norm_symbol(s) for s in body.symbols]

    # 2) OHLCV 読み込み + 指標付与
    ohlcv_map: Dict[str, "pd.DataFrame"] = {}
    for sym in symbols:
        df = None
        if body.timeframe == "15m":
            df = load_ohlcv_15m(sym, lookback_min=body.lookback_min)
        # 必要なら 5m/1h を追加
        if df is not None and not df.empty:
            df = add_indicators(df)
        ohlcv_map[sym] = df

    # 3) 現在価格（price_cache）
    price_now, price_ts = {}, {}
    try:
        with engine.begin() as conn:
            rows = conn.execute("""
                SELECT lower(symbol) AS symbol, last, ts
                  FROM price_cache
                 WHERE lower(symbol) = ANY(:syms)
            """, {"syms": symbols}).mappings().all()
        for r in rows:
            price_now[r["symbol"]] = float(r["last"])
            price_ts[r["symbol"]] = r["ts"].astimezone(timezone.utc).isoformat() if r["ts"] else None
    except Exception:
        # price_cache が無くても動くように
        pass

    # 4) バリアントを内部表現に変換
    variants = []
    for v in body.variants:
        variants.append(Variant(
            id=v.id,
            entry=EntryRule(
                atr_min=v.atr_min,
                adx_min=v.adx_min,
                rsi_long_min=v.rsi_long_min,
                macd_cross=v.macd_cross,
                ema_cross_req=v.ema_cross_req,
                cross_mode=v.cross_mode or "any",
            ),
            exit=ExitRule(
                take_profit_pct=v.take_profit_pct,
                stop_loss_pct=v.stop_loss_pct,
                max_holding_min=v.max_holding_min,
                use_dynamic_levels=bool(v.use_dynamic_levels),
            ),
            trade_size_jpy=v.trade_size_jpy or 10_000
        ))

    # 5) 実行
    results = run_virtual_variants(ohlcv_map, price_now, variants)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": body.timeframe,
        "results": results,
        "price_now": price_now,
        "price_ts": price_ts,
    }
