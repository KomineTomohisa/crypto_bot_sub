# app/routers/strategies.py
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Literal   # ← 追加
from datetime import datetime
import json                                  # ← 追加
import sqlalchemy as sa
from db import engine

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

# ---------- Pydantic ----------
class StrategyIn(BaseModel):
    user_id: int = Field(..., ge=1)
    name: str
    symbols: List[str] = Field(default_factory=list)
    timeframe: str = "15m"   # ← 既定を15mに（必要に応じて5mのままでもOK）

    # Entry thresholds
    atr_min: Optional[float] = None
    atr_pct_min: Optional[float] = None       # ← 追加
    adx_min: Optional[float] = None
    rsi_long_min: Optional[float] = None
    rsi_short_max: Optional[float] = None
    macd_cross: Optional[bool] = None
    ema_cross_req: Optional[bool] = None
    cross_mode: Literal["any","all"] = "any"  # ← 追加

    # Exit thresholds
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    max_holding_min: Optional[int] = None
    use_dynamic_levels: bool = False

    trade_size_jpy: float = 10_000
    is_enabled: bool = True

class StrategyOut(StrategyIn):
    strategy_id: str
    created_at: datetime
    updated_at: datetime

# ---------- SQL ----------
INSERT_SQL = sa.text("""
INSERT INTO user_strategies (
  strategy_id, user_id, name, symbols, timeframe,
  atr_min, atr_pct_min, adx_min, rsi_long_min, rsi_short_max,
  macd_cross, ema_cross_req, cross_mode,
  take_profit_pct, stop_loss_pct, max_holding_min, use_dynamic_levels,
  trade_size_jpy, is_enabled
) VALUES (
  gen_random_uuid(), :user_id, :name, :symbols, :timeframe,
  :atr_min, :atr_pct_min, :adx_min, :rsi_long_min, :rsi_short_max,
  :macd_cross, :ema_cross_req, :cross_mode,
  :take_profit_pct, :stop_loss_pct, :max_holding_min, :use_dynamic_levels,
  :trade_size_jpy, :is_enabled
)
RETURNING *;
""")

SELECT_LIST_SQL = sa.text("""
SELECT *
FROM user_strategies
WHERE (:user_id IS NULL OR user_id = :user_id)
ORDER BY updated_at DESC, created_at DESC;
""")

SELECT_ONE_SQL = sa.text("""
SELECT * FROM user_strategies WHERE strategy_id = :sid;
""")

UPDATE_SQL = sa.text("""
UPDATE user_strategies SET
  user_id = COALESCE(:user_id, user_id),
  name = COALESCE(:name, name),
  symbols = COALESCE(:symbols, symbols),
  timeframe = COALESCE(:timeframe, timeframe),
  atr_min = :atr_min,
  atr_pct_min = :atr_pct_min,
  adx_min = :adx_min,
  rsi_long_min = :rsi_long_min,
  rsi_short_max = :rsi_short_max,
  macd_cross = :macd_cross,
  ema_cross_req = :ema_cross_req,
  cross_mode = :cross_mode,
  take_profit_pct = :take_profit_pct,
  stop_loss_pct = :stop_loss_pct,
  max_holding_min = :max_holding_min,
  use_dynamic_levels = :use_dynamic_levels,
  trade_size_jpy = :trade_size_jpy,
  is_enabled = :is_enabled,
  updated_at = NOW()
WHERE strategy_id = :sid
RETURNING *;
""")

DELETE_SQL = sa.text("""
DELETE FROM user_strategies WHERE strategy_id = :sid;
""")

# ---------- Helpers ----------
def _to_symbols(v) -> List[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, (bytes, str)):
        s = v.decode() if isinstance(v, bytes) else v
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return [str(x) for x in j]
        except Exception:
            pass
        # fallback: {eth_jpy,xrp_jpy} 形式や "a,b" に緩く対応
        s = s.strip()
        if s.startswith("{") and s.endswith("}"):
            inner = s[1:-1]
            return [x.strip().strip('"') for x in inner.split(",") if x.strip()]
        if "," in s:
            return [x.strip() for x in s.split(",")]
        return [s]
    return []

def _row_to_out(row) -> StrategyOut:
    return StrategyOut(
        strategy_id=str(row["strategy_id"]),
        user_id=row["user_id"],
        name=row["name"],
        symbols=_to_symbols(row["symbols"]),    # ← 安全に復元
        timeframe=row["timeframe"],
        atr_min=row["atr_min"],
        atr_pct_min=row.get("atr_pct_min"),     # ← 追加
        adx_min=row["adx_min"],
        rsi_long_min=row["rsi_long_min"],
        rsi_short_max=row["rsi_short_max"],
        macd_cross=row["macd_cross"],
        ema_cross_req=row["ema_cross_req"],
        cross_mode=row.get("cross_mode", "any"),# ← 追加
        take_profit_pct=row["take_profit_pct"],
        stop_loss_pct=row["stop_loss_pct"],
        max_holding_min=row["max_holding_min"],
        use_dynamic_levels=bool(row["use_dynamic_levels"]),
        trade_size_jpy=float(row["trade_size_jpy"]),
        is_enabled=bool(row["is_enabled"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )

# ---------- Endpoints ----------
@router.get("", response_model=List[StrategyOut])
def list_strategies(user_id: int | None = Query(None)):
    with engine.begin() as conn:
        rows = conn.execute(SELECT_LIST_SQL, {"user_id": user_id}).mappings().all()
    return [_row_to_out(r) for r in rows]

@router.post("", response_model=StrategyOut)
def create_strategy(body: StrategyIn):
    with engine.begin() as conn:
        row = conn.execute(INSERT_SQL, {
            "user_id": body.user_id,
            "name": body.name,
            "symbols": body.symbols,
            "timeframe": body.timeframe,
            "atr_min": body.atr_min,
            "atr_pct_min": body.atr_pct_min,      # ← 追加
            "adx_min": body.adx_min,
            "rsi_long_min": body.rsi_long_min,
            "rsi_short_max": body.rsi_short_max,
            "macd_cross": body.macd_cross,
            "ema_cross_req": body.ema_cross_req,
            "cross_mode": body.cross_mode,        # ← 追加
            "take_profit_pct": body.take_profit_pct,
            "stop_loss_pct": body.stop_loss_pct,
            "max_holding_min": body.max_holding_min,
            "use_dynamic_levels": body.use_dynamic_levels,
            "trade_size_jpy": body.trade_size_jpy,
            "is_enabled": body.is_enabled,
        }).mappings().first()
    return _row_to_out(row)

@router.get("/{strategy_id}", response_model=StrategyOut)
def get_strategy(strategy_id: str):
    with engine.begin() as conn:
        row = conn.execute(SELECT_ONE_SQL, {"sid": strategy_id}).mappings().first()
    if not row:
        raise HTTPException(404, "strategy not found")
    return _row_to_out(row)

@router.put("/{strategy_id}", response_model=StrategyOut)
def update_strategy(strategy_id: str, body: StrategyIn):
    with engine.begin() as conn:
        cur = conn.execute(SELECT_ONE_SQL, {"sid": strategy_id}).mappings().first()
        if not cur:
            raise HTTPException(404, "strategy not found")

        row = conn.execute(UPDATE_SQL, {
            "sid": strategy_id,
            "user_id": body.user_id,
            "name": body.name,
            "symbols": body.symbols,
            "timeframe": body.timeframe,
            "atr_min": body.atr_min,
            "atr_pct_min": body.atr_pct_min,      # ← 追加
            "adx_min": body.adx_min,
            "rsi_long_min": body.rsi_long_min,
            "rsi_short_max": body.rsi_short_max,
            "macd_cross": body.macd_cross,
            "ema_cross_req": body.ema_cross_req,
            "cross_mode": body.cross_mode,        # ← 追加
            "take_profit_pct": body.take_profit_pct,
            "stop_loss_pct": body.stop_loss_pct,
            "max_holding_min": body.max_holding_min,
            "use_dynamic_levels": body.use_dynamic_levels,
            "trade_size_jpy": body.trade_size_jpy,
            "is_enabled": body.is_enabled,
        }).mappings().first()
    return _row_to_out(row)

@router.delete("/{strategy_id}")
def delete_strategy(strategy_id: str):
    with engine.begin() as conn:
        res = conn.execute(DELETE_SQL, {"sid": strategy_id})
        if res.rowcount == 0:
            raise HTTPException(404, "strategy not found")
    return {"ok": True}
