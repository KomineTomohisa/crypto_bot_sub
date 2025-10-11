# app/virtual_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import pandas as pd

@dataclass
class EntryRule:
    atr_min: Optional[float] = None
    adx_min: Optional[float] = None
    rsi_long_min: Optional[float] = None
    macd_cross: Optional[bool] = None
    ema_cross_req: Optional[bool] = None
    cross_mode: str = "any"   # "any"=MACDまたはEMA, "all"=両方
    atr_pct_min: Optional[float] = None

@dataclass
class ExitRule:
    take_profit_pct: Optional[float] = None    # 例: 0.016 (=+1.6%)
    stop_loss_pct: Optional[float] = None      # 例:-0.009 (=-0.9%)
    max_holding_min: Optional[int] = None
    use_dynamic_levels: bool = False           # 将来: ATR/ADXに応じた可変TP/SL

@dataclass
class Variant:
    id: str
    entry: EntryRule
    exit: ExitRule
    trade_size_jpy: float = 10_000

def _row_entry_ok(row: pd.Series, r: EntryRule) -> bool:
    conds = []
    if r.atr_min is not None:      conds.append(row.get("ATR") and row["ATR"] >= r.atr_min)
    if r.adx_min is not None:      conds.append(row.get("ADX") and row["ADX"] >= r.adx_min)
    if r.rsi_long_min is not None: conds.append(row.get("RSI") and row["RSI"] >= r.rsi_long_min)
    if r.atr_pct_min is not None:  conds.append(row.get("ATR_PCT") and row["ATR_PCT"] >= r.atr_pct_min)

    # MACD / EMA の複合ロジック
    macd_ok = True if r.macd_cross is None else (
        bool(row.get("MACD_cross_up")) if r.macd_cross else bool(row.get("MACD_cross_down"))
    )
    ema_ok = True if r.ema_cross_req is None else bool(row.get("EMA_cross_up"))

    if r.macd_cross is not None or r.ema_cross_req is not None:
        if r.cross_mode == "all":
            conds.append(macd_ok and ema_ok)
        else:  # "any"
            conds.append(macd_ok or ema_ok)

    return all(conds) if conds else False

def _should_exit(entry_price: float, now_price: float, opened_at: datetime, ex: ExitRule) -> bool:
    if ex.take_profit_pct is not None:
        if (now_price / entry_price - 1.0) >= ex.take_profit_pct:
            return True
    if ex.stop_loss_pct is not None:
        if (now_price / entry_price - 1.0) <= ex.stop_loss_pct:
            return True
    if ex.max_holding_min is not None:
        if datetime.now(timezone.utc) - opened_at >= timedelta(minutes=ex.max_holding_min):
            return True
    return False

def run_virtual_variants(
    ohlcv_map: Dict[str, pd.DataFrame],
    price_now: Dict[str, float],
    variants: List[Variant],
) -> List[Dict[str, Any]]:
    """
    各symbolを「ロングのみ」で走査し、variantごとの結果を返す。
    ohlcv_map[symbol]: 指標列まで入ったDF（timestamp昇順）
    """
    out = []
    for v in variants:
        trades: List[Dict[str, Any]] = []
        open_pos: Dict[str, Dict[str, Any]] = {}

        for sym, df in ohlcv_map.items():
            if df is None or df.empty or "close" not in df.columns:
                continue

            df = df.dropna(subset=["close"])
            for _, row in df.iterrows():
                ts = row["timestamp"]
                px = float(row["close"])

                if sym not in open_pos:
                    # エントリー判定（ロングのみ）
                    if _row_entry_ok(row, v.entry):
                        open_pos[sym] = {
                            "symbol": sym,
                            "side": "long",
                            "entry_price": px,
                            "time": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else pd.to_datetime(ts).to_pydatetime(),
                        }
                else:
                    p = open_pos[sym]
                    if _should_exit(p["entry_price"], px, p["time"], v.exit):
                        size_jpy = v.trade_size_jpy
                        pnl_jpy = (px - p["entry_price"]) * (size_jpy / p["entry_price"])
                        trades.append({
                            "symbol": sym,
                            "side": "long",
                            "entry_time": p["time"].isoformat(),
                            "entry_price": p["entry_price"],
                            "exit_time": ts.isoformat(),
                            "exit_price": px,
                            "pnl_jpy": float(pnl_jpy),
                        })
                        del open_pos[sym]

        # 未決済（現在値で含み損益）
        open_positions = []
        for sym, p in open_pos.items():
            nowp = price_now.get(sym)
            u_pnl = None
            if nowp:
                u_pnl = (nowp - p["entry_price"]) * (v.trade_size_jpy / p["entry_price"])
            open_positions.append({
                "symbol": sym,
                "side": "long",
                "entry_time": p["time"].isoformat(),
                "entry_price": p["entry_price"],
                "unrealized_pnl_jpy": float(u_pnl) if u_pnl is not None else None
            })

        pnl_total = sum(t["pnl_jpy"] for t in trades) if trades else 0.0
        win = sum(1 for t in trades if t["pnl_jpy"] > 0)
        summary = {
            "trades": len(trades),
            "win_rate": round(win / len(trades) * 100.0, 1) if trades else None,
            "pnl_jpy": float(pnl_total),
            "pnl_pct": None,
            "avg_holding_min": None,
            "max_dd_jpy": None,
        }

        out.append({
            "variant": v.id,
            "summary": summary,
            "open_positions": open_positions,
            "trades": trades[-200:],  # 表示用に直近200件に制限
        })
    return out
