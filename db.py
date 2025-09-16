from __future__ import annotations
import os
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import uuid4

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
import sqlalchemy as sa
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import IntegrityError

# --- 接続とユーティリティ --------------------------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

engine: Engine = sa.create_engine(
    DATABASE_URL,
    pool_pre_ping=True,                # 死んだ接続を自動再接続
    isolation_level="AUTOCOMMIT",      # シンプルにイベント単位でコミット
)

def _exec(stmt: sa.TextClause, params: dict, conn: Connection | None) -> None:
    if conn is not None:
        conn.execute(stmt, params)
    else:
        with engine.begin() as c:
            c.execute(stmt, params)

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

@contextmanager
def begin() -> Connection:
    """明示トランザクション。複数の upsert を1イベントで確定したい時に使う"""
    with engine.begin() as conn:
        yield conn

# --- スキーマのメタ（Alembicで作ったテーブル名に一致させる） --------------
orders = sa.table(
    "orders",
    sa.column("order_id", sa.String),
    sa.column("symbol", sa.String),
    sa.column("side", sa.String),
    sa.column("type", sa.String),
    sa.column("size", sa.Numeric),
    sa.column("status", sa.String),
    sa.column("requested_at", sa.DateTime(timezone=True)),
    sa.column("placed_at", sa.DateTime(timezone=True)),
    sa.column("raw", sa.JSON),
)
fills = sa.table(
    "fills",
    sa.column("fill_id", sa.String),
    sa.column("order_id", sa.String),
    sa.column("price", sa.Numeric),
    sa.column("size", sa.Numeric),
    sa.column("fee", sa.Numeric),
    sa.column("executed_at", sa.DateTime(timezone=True)),
    sa.column("raw", sa.JSON),
)
positions = sa.table(
    "positions",
    sa.column("position_id", sa.String),
    sa.column("symbol", sa.String),
    sa.column("side", sa.String),
    sa.column("size", sa.Numeric),
    sa.column("avg_entry_price", sa.Numeric),
    sa.column("opened_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
    sa.column("raw", sa.JSON),
)
trades = sa.table(
    "trades",
    sa.column("trade_id", sa.String),
    sa.column("symbol", sa.String),
    sa.column("side", sa.String),
    sa.column("entry_position_id", sa.String),
    sa.column("exit_order_id", sa.String),
    sa.column("entry_price", sa.Numeric),
    sa.column("exit_price", sa.Numeric),
    sa.column("size", sa.Numeric),
    sa.column("pnl", sa.Numeric),
    sa.column("pnl_pct", sa.Numeric),
    sa.column("holding_hours", sa.Numeric),
    sa.column("closed_at", sa.DateTime(timezone=True)),
    sa.column("raw", sa.JSON),
)
balance_snapshots = sa.table(
    "balance_snapshots",
    sa.column("ts", sa.DateTime(timezone=True)),
    sa.column("total_balance", sa.Numeric),
    sa.column("available_margin", sa.Numeric),
    sa.column("profit_loss", sa.Numeric),
    sa.column("raw", sa.JSON),
)
errors = sa.table(
    "errors",
    sa.column("id", sa.BigInteger),
    sa.column("ts", sa.DateTime(timezone=True)),
    sa.column("where", sa.String),
    sa.column("message", sa.Text),
    sa.column("stack", sa.Text),
    sa.column("raw", sa.JSON),
)

def _jsonable(raw: Any) -> Any:
    try:
        json.dumps(raw)
        return raw
    except Exception:
        return {"repr": str(raw)}

# --- Upsert/Insert API（冪等） ----------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def insert_order(
    order_id: str, symbol: str, side: str, type_: str, size: float,
    status: str, requested_at: Optional[datetime], placed_at: Optional[datetime],
    raw: Optional[Dict[str, Any]] = None, conn: Optional[Connection] = None
) -> None:
    stmt = sa.text("""
        INSERT INTO orders (order_id, symbol, side, type, size, status, requested_at, placed_at, raw)
        VALUES (:order_id, :symbol, :side, :type, :size, :status, :requested_at, :placed_at, :raw)
        ON CONFLICT (order_id) DO UPDATE
        SET status = EXCLUDED.status,
            placed_at = COALESCE(EXCLUDED.placed_at, orders.placed_at),
            raw = COALESCE(EXCLUDED.raw, orders.raw)
    """)
    params = dict(order_id=order_id, symbol=symbol, side=side, type=type_,
                  size=size, status=status, requested_at=requested_at,
                  placed_at=placed_at, raw=_jsonable(raw))
    _exec(stmt, params, conn)

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def upsert_fill(
    fill_id: Optional[str], order_id: str, price: float, size: float,
    fee: Optional[float], executed_at: datetime, raw: Optional[Dict[str, Any]] = None,
    conn: Optional[Connection] = None
) -> str:
    fid = fill_id or str(uuid4())
    stmt = sa.text("""
        INSERT INTO fills (fill_id, order_id, price, size, fee, executed_at, raw)
        VALUES (:fill_id, :order_id, :price, :size, :fee, :executed_at, :raw)
        ON CONFLICT (fill_id) DO UPDATE
        SET price = EXCLUDED.price,
            size  = EXCLUDED.size,
            fee   = COALESCE(EXCLUDED.fee, fills.fee),
            raw   = COALESCE(EXCLUDED.raw, fills.raw)
    """)
    params = dict(fill_id=fid, order_id=order_id, price=price, size=size,
                  fee=fee, executed_at=executed_at, raw=_jsonable(raw))
    _exec(stmt, params, conn)
    return fid

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def upsert_position(
    position_id: str, symbol: str, side: str, size: float, avg_entry_price: float,
    opened_at: Optional[datetime], updated_at: Optional[datetime], raw: Optional[Dict[str, Any]] = None,
    conn: Optional[Connection] = None
) -> None:
    stmt = sa.text("""
        INSERT INTO positions (position_id, symbol, side, size, avg_entry_price, opened_at, updated_at, raw)
        VALUES (:position_id, :symbol, :side, :size, :avg_entry_price, :opened_at, :updated_at, :raw)
        ON CONFLICT (position_id) DO UPDATE
        SET symbol = EXCLUDED.symbol,
            side   = EXCLUDED.side,
            size   = EXCLUDED.size,
            avg_entry_price = EXCLUDED.avg_entry_price,
            opened_at  = COALESCE(EXCLUDED.opened_at, positions.opened_at),
            updated_at = COALESCE(EXCLUDED.updated_at, EXCLUDED.opened_at, positions.updated_at),
            raw = COALESCE(EXCLUDED.raw, positions.raw)
    """)
    params = dict(position_id=position_id, symbol=symbol, side=side, size=size,
                  avg_entry_price=avg_entry_price, opened_at=opened_at,
                  updated_at=updated_at or utcnow(), raw=_jsonable(raw))
    _exec(stmt, params, conn)

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def insert_trade(
    trade_id: Optional[str], symbol: str, side: str,
    entry_position_id: Optional[str], exit_order_id: Optional[str],
    entry_price: float, exit_price: float, size: float,
    pnl: float, pnl_pct: float, holding_hours: Optional[float],
    closed_at: datetime, raw: Optional[Dict[str, Any]] = None,
    conn: Optional[Connection] = None
) -> str:
    tid = trade_id or str(uuid4())
    stmt = sa.text("""
        INSERT INTO trades (
            trade_id, symbol, side, entry_position_id, exit_order_id,
            entry_price, exit_price, size, pnl, pnl_pct, holding_hours, closed_at, raw
        )
        VALUES (
            :trade_id, :symbol, :side, :entry_position_id, :exit_order_id,
            :entry_price, :exit_price, :size, :pnl, :pnl_pct, :holding_hours, :closed_at, :raw
        )
        ON CONFLICT (trade_id) DO NOTHING
    """)
    params = dict(trade_id=tid, symbol=symbol, side=side,
                  entry_position_id=entry_position_id, exit_order_id=exit_order_id,
                  entry_price=entry_price, exit_price=exit_price, size=size,
                  pnl=pnl, pnl_pct=pnl_pct, holding_hours=holding_hours,
                  closed_at=closed_at, raw=_jsonable(raw))
    _exec(stmt, params, conn)
    return tid

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def insert_balance_snapshot(
    ts: Optional[datetime], total_balance: float,
    available_margin: Optional[float], profit_loss: Optional[float],
    raw: Optional[Dict[str, Any]] = None, conn: Optional[Connection] = None
) -> None:
    stmt = sa.text("""
        INSERT INTO balance_snapshots (ts, total_balance, available_margin, profit_loss, raw)
        VALUES (:ts, :total_balance, :available_margin, :profit_loss, :raw)
        ON CONFLICT (ts) DO UPDATE
        SET total_balance    = EXCLUDED.total_balance,
            available_margin = COALESCE(EXCLUDED.available_margin, balance_snapshots.available_margin),
            profit_loss      = COALESCE(EXCLUDED.profit_loss, balance_snapshots.profit_loss),
            raw = COALESCE(EXCLUDED.raw, balance_snapshots.raw)
    """)
    params = dict(
        ts=ts or utcnow(),
        total_balance=total_balance,
        available_margin=available_margin,
        profit_loss=profit_loss,
        raw=_jsonable(raw),
    )
    _exec(stmt, params, conn)

def insert_error(where: str, message: str, stack: Optional[str] = None, raw: Any = None) -> None:
    try:
        stmt = sa.text("""
            INSERT INTO errors (ts, "where", message, stack, raw)
            VALUES (:ts, :where, :message, :stack, :raw)
        """)
        params = dict(ts=utcnow(), where=where, message=message, stack=stack, raw=_jsonable(raw))
        with engine.begin() as conn:
            conn.execute(stmt, params)
    except Exception:
        pass

# --- 便利: 注文→約定 の一括トランザクション例 -------------------------------

def mark_order_executed_with_fill(
    order_id: str, executed_size: float, price: float, fee: Optional[float],
    executed_at: datetime, fill_raw: Optional[Dict[str, Any]] = None,
    order_raw: Optional[Dict[str, Any]] = None,
) -> None:
    """orders.status を EXECUTED にし、fills を upsert（同一トランザクション）"""
    with begin() as conn:
        # status 更新
        conn.execute(sa.text("""
            UPDATE orders SET status = 'EXECUTED', raw = COALESCE(:raw, raw)
            WHERE order_id = :oid
        """), {"oid": order_id, "raw": _jsonable(order_raw)})

        # fill upsert
        upsert_fill(
            fill_id=None, order_id=order_id, price=price, size=executed_size,
            fee=fee, executed_at=executed_at, raw=fill_raw, conn=conn
        )
