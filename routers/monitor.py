from datetime import datetime
from typing import List, Optional, Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import text

from db import engine

router = APIRouter(prefix="/api", tags=["monitor"])

# ---------- Pydantic models ----------
class SignalOut(BaseModel):
    signal_id: str
    user_id: int
    symbol: str
    timeframe: str
    status: str
    side: Optional[str] = None
    price: Optional[float] = None
    generated_at: datetime
    strategy_id: Optional[str] = None
    source: str

class PositionOut(BaseModel):
    position_id: str
    user_id: Optional[int] = None
    strategy_id: Optional[str] = None
    symbol: str
    side: str
    size: float
    avg_entry_price: float
    opened_at: datetime
    updated_at: datetime
    source: str

class TradeOut(BaseModel):
    trade_id: str
    user_id: Optional[int] = None
    strategy_id: Optional[str] = None
    symbol: str
    side: str
    entry_position_id: Optional[str] = None
    entry_price: float
    exit_price: float
    size: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_hours: Optional[float] = None
    closed_at: datetime
    source: str

# ---------- Helpers ----------
def _clamp_limit(x: int, lo: int = 1, hi: int = 1000) -> int:
    return max(lo, min(hi, x))

# ---------- Endpoints ----------
@router.get("/signals", response_model=List[SignalOut])
def list_signals(
    user_id: int,
    source: Annotated[str, Query(pattern=r"^(virtual|real|\*)$")] = "virtual",
    symbol: Optional[str] = None,
    strategy_id: Optional[str] = None,
    timeframe: Optional[str] = None,
    status: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 200,
    offset: int = 0,
):
    limit = _clamp_limit(limit, hi=1000)
    src = None if source == "*" else source
    sql = """
        SELECT signal_id::text          AS signal_id,
               user_id,
               symbol,
               timeframe,
               status,
               side,
               price::float8            AS price,
               generated_at,
               strategy_id::text        AS strategy_id,
               source
          FROM signals
         WHERE user_id = :uid
           AND (:src IS NULL OR source = :src)
           AND (:sym IS NULL OR symbol = :sym)
           AND (:sid IS NULL OR strategy_id::text = :sid)
           AND (:tf  IS NULL OR timeframe = :tf)
           AND (:st  IS NULL OR status = :st)
           AND (:since IS NULL OR generated_at >= :since)
           AND (:until IS NULL OR generated_at <  :until)
         ORDER BY generated_at DESC
         LIMIT :limit OFFSET :offset
    """
    params = dict(
        uid=user_id, src=src, sym=symbol, sid=strategy_id,
        tf=timeframe, st=status, since=since, until=until,
        limit=limit, offset=offset,
    )
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]

@router.get("/positions", response_model=List[PositionOut])
def list_positions(
    user_id: Optional[int] = None,
    source: Annotated[str, Query(pattern=r"^(virtual|real|\*)$")] = "virtual",
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    open_only: bool = True,
    limit: int = 200,
    offset: int = 0,
):
    limit = _clamp_limit(limit, hi=1000)
    src = None if source == "*" else source
    sql = """
        SELECT position_id::text        AS position_id,
               user_id,
               strategy_id::text        AS strategy_id,
               symbol,
               side,
               size::float8             AS size,
               avg_entry_price::float8  AS avg_entry_price,
               opened_at,
               updated_at,
               source
          FROM positions
         WHERE (:uid IS NULL OR user_id = :uid)
           AND (:src IS NULL OR source = :src)
           AND (:sid IS NULL OR strategy_id::text = :sid)
           AND (:sym IS NULL OR symbol = :sym)
           AND (:open_only = false OR size > 0)
         ORDER BY opened_at DESC
         LIMIT :limit OFFSET :offset
    """
    params = dict(
        uid=user_id, src=src, sid=strategy_id, sym=symbol,
        open_only=open_only, limit=limit, offset=offset,
    )
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]

@router.get("/trades", response_model=List[TradeOut])
def list_trades(
    user_id: Optional[int] = None,
    source: Annotated[str, Query(pattern=r"^(virtual|real|\*)$")] = "virtual",
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 200,
    offset: int = 0,
):
    limit = _clamp_limit(limit, hi=1000)
    src = None if source == "*" else source
    sql = """
        SELECT trade_id::text           AS trade_id,
               user_id,
               strategy_id::text        AS strategy_id,
               symbol,
               side,
               entry_position_id::text  AS entry_position_id,
               entry_price::float8      AS entry_price,
               exit_price::float8       AS exit_price,
               size::float8             AS size,
               pnl::float8              AS pnl,
               pnl_pct::float8          AS pnl_pct,
               holding_hours::float8    AS holding_hours,
               closed_at,
               source
          FROM trades
         WHERE (:uid IS NULL OR user_id = :uid)
           AND (:src IS NULL OR source = :src)
           AND (:sid IS NULL OR strategy_id::text = :sid)
           AND (:sym IS NULL OR symbol = :sym)
           AND (:since IS NULL OR closed_at >= :since)
           AND (:until IS NULL OR closed_at <  :until)
         ORDER BY closed_at DESC
         LIMIT :limit OFFSET :offset
    """
    params = dict(
        uid=user_id, src=src, sid=strategy_id, sym=symbol,
        since=since, until=until, limit=limit, offset=offset,
    )
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]