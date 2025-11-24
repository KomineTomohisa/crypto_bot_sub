from __future__ import annotations

import os
import json
import logging
from contextlib import contextmanager
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Sequence
from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
import sqlalchemy as sa
from sqlalchemy.engine import Engine, Connection
from sqlalchemy import text

try:
    from psycopg2.extras import Json as _PsycoJson
except Exception:
    _PsycoJson = None

# -----------------------------------------------------------------------------
# 環境
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

JST = timezone(timedelta(hours=9))

def _redact_url(u: str) -> str:
    try:
        from urllib.parse import urlparse
        p = urlparse(u)
        user = p.username or ""
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        db   = p.path.lstrip("/") if p.path else ""
        return f"{p.scheme}://{user}:***@{host}{port}/{db}"
    except Exception:
        return "unknown"

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).info("DB target: %s", _redact_url(DATABASE_URL))

# -----------------------------------------------------------------------------
# 接続（AUTOCOMMITを撤廃。begin() ヘルパで明示トランザクション運用）
# -----------------------------------------------------------------------------
engine: Engine = sa.create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # isolation_level="AUTOCOMMIT",  # ← 撤廃：原子性担保のため
)

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

@contextmanager
def begin() -> Connection:
    """
    明示トランザクション。複数の upsert/insert/update を
    ひとまとまりに原子的に実行したい時に使う。
    """
    with engine.begin() as conn:
        yield conn

def _exec(stmt: sa.TextClause, params: dict, conn: Connection | None) -> None:
    if conn is not None:
        conn.execute(stmt, params)
    else:
        with engine.begin() as c:
            c.execute(stmt, params)

# -----------------------------------------------------------------------------
# JSON / 変換
# -----------------------------------------------------------------------------
try:
    import numpy as np
except Exception:
    np = None

def _to_plain(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if np is not None:
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_plain(v) for v in obj]
    return str(obj)

def _jsonable(v):
    # psycopg2.Json を使える場合は優先。なければプレーン（dict/list は事前plain化）
    if isinstance(v, (dict, list, tuple, set)):
        plain = _to_plain(v)
        return _PsycoJson(plain) if _PsycoJson else plain
    return _to_plain(v)

# -----------------------------------------------------------------------------
# テーブル（メタ）
# -----------------------------------------------------------------------------
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
    sa.column("user_id", sa.BigInteger),
    sa.column("symbol", sa.String),
    sa.column("side", sa.String),
    sa.column("size", sa.Numeric),
    sa.column("avg_entry_price", sa.Numeric),
    sa.column("opened_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
    sa.column("raw", sa.JSON),
    sa.column("strategy_id", sa.String),
    sa.column("source", sa.String),
)

trades = sa.table(
    "trades",
    sa.column("trade_id", sa.String),
    sa.column("user_id", sa.BigInteger),
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
    sa.column("strategy_id", sa.String),
    sa.column("source", sa.String),
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

user_integrations = sa.table(
    "user_integrations",
    sa.column("user_id", sa.BigInteger),
    sa.column("provider", sa.String),
    sa.column("access_token_enc", sa.LargeBinary),
    sa.column("token_last4", sa.String),
    sa.column("status", sa.String),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)

user_line_endpoints = sa.table(
    "user_line_endpoints",
    sa.column("user_id", sa.BigInteger),
    sa.column("line_user_id", sa.String),
    sa.column("status", sa.String),
    sa.column("display_name", sa.String),
    sa.column("last_seen_at", sa.DateTime(timezone=True)),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)

line_channels = sa.table(
    "line_channels",
    sa.column("id", sa.BigInteger),
    sa.column("provider_key", sa.String),
    sa.column("access_token_enc", sa.LargeBinary),
    sa.column("status", sa.String),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)

signals = sa.table(
    "signals",
    sa.column("signal_id", sa.String),
    sa.column("user_id", sa.BigInteger),
    sa.column("symbol", sa.String),
    sa.column("timeframe", sa.String),
    sa.column("side", sa.String),
    sa.column("strength_score", sa.Numeric),
    sa.column("rsi", sa.Numeric),
    sa.column("adx", sa.Numeric),
    sa.column("atr", sa.Numeric),
    sa.column("di_plus", sa.Numeric),
    sa.column("di_minus", sa.Numeric),
    sa.column("ema_fast", sa.Numeric),
    sa.column("ema_slow", sa.Numeric),
    sa.column("price", sa.Numeric),
    sa.column("generated_at", sa.DateTime(timezone=True)),
    sa.column("strategy_id", sa.String),
    sa.column("version", sa.String),
    sa.column("status", sa.String),
    sa.column("raw", sa.JSON),
    sa.column("source", sa.String),   # ← 追加：実スキーマ準拠
)

# -----------------------------------------------------------------------------
# 参照系
# -----------------------------------------------------------------------------
def get_trades_between(start_dt, end_dt, symbols=None, min_abs_pnl=None):
    start_utc = _as_utc(start_dt)
    end_utc   = _as_utc(end_dt)
    where = ["closed_at >= :start_dt", "closed_at < :end_dt"]
    params = {"start_dt": start_utc, "end_dt": end_utc}
    if symbols:
        where.append("symbol = ANY(:syms)")
        params["syms"] = list(symbols)
    if min_abs_pnl is not None:
        where.append("ABS(pnl) >= :min_abs_pnl")
        params["min_abs_pnl"] = float(min_abs_pnl)
    sql = f"""
        SELECT trade_id, symbol, side, entry_position_id, exit_order_id,
               entry_price, exit_price, size, pnl, pnl_pct,
               holding_hours, closed_at, raw
          FROM trades
         WHERE {' AND '.join(where)}
         ORDER BY closed_at ASC
    """
    with engine.begin() as conn:
        rows = conn.execute(sa.text(sql), params).mappings().all()
    return [dict(r) for r in rows]

def get_trades_for_day_jst(day_dt):
    day = day_dt
    if day.tzinfo is None: day = day.replace(tzinfo=JST)
    else:                  day = day.astimezone(JST)
    start_jst = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end_jst   = start_jst + timedelta(days=1)
    return get_trades_between(start_jst, end_jst)

def get_signals_between(start_dt, end_dt, *, symbol: str | None = None, timeframe: str | None = None):
    where = ["generated_at >= :start_dt", "generated_at < :end_dt"]
    params = {"start_dt": start_dt, "end_dt": end_dt}
    if symbol:
        where.append("symbol = :symbol"); params["symbol"] = symbol
    if timeframe:
        where.append("timeframe = :timeframe"); params["timeframe"] = timeframe
    sql = f"""
        SELECT *
          FROM signals
         WHERE {' AND '.join(where)}
         ORDER BY generated_at DESC
    """
    with engine.begin() as conn:
        rows = conn.execute(sa.text(sql), params).mappings().all()
    return [dict(r) for r in rows]

def get_public_metrics_daily(start, end) -> List[Dict[str, Any]]:
    sql = text("""
        SELECT metric_date, symbols
          FROM public_metrics_daily
         WHERE metric_date >= :start AND metric_date <= :end
         ORDER BY metric_date ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"start": start, "end": end}).mappings().all()
    return [dict(r) for r in rows]

def fetch_signal_rules(
    symbol: str,
    timeframe: str = "15m",
    *,
    version: str | None = None,
    user_id: str | None = None,
    strategy_id: str | None = None,
    only_open_ended: bool = False,  # ← 追加
):
    cond_valid_to = "valid_to IS NULL" if only_open_ended else "(valid_to IS NULL OR valid_to >= NOW())"
    sql = sa.text(f"""
        SELECT symbol, timeframe, score_col, op, v1, v2,
               target_side, action, priority
          FROM signal_rule_thresholds
         WHERE active = TRUE
           AND symbol = :symbol
           AND timeframe = :timeframe
           AND (valid_from IS NULL OR valid_from <= NOW())
           AND {cond_valid_to}
           AND (:version IS NULL OR version = :version)
           AND (:user_id IS NULL OR user_id = :user_id)
           AND (:strategy_id IS NULL OR strategy_id = :strategy_id)
         ORDER BY priority ASC, id ASC
    """)
    with begin() as conn:
        rows = conn.execute(sql, {
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
            "user_id": user_id,
            "strategy_id": strategy_id,
        }).mappings().all()
    return [dict(r) for r in rows]

def get_user_email_and_pw(user_id: int) -> Dict[str, Optional[str]]:
    """
    SIM通知用：user.id から宛先メールと（必要なら）暗号化済みメールパスワードを返す。
    password_hash はここでは使わず、別用途（ログイン等）。
    """
    with begin() as conn:
        row = conn.execute(text("""
            SELECT
              COALESCE(email, '') AS email,
              COALESCE(email_password_encrypted, '') AS email_password_encrypted,
              COALESCE(email_enabled, TRUE) AS email_enabled
            FROM "user"
            WHERE id = :uid
            LIMIT 1
        """), {"uid": user_id}).fetchone()
        if not row:
            return {"email": None, "email_password_encrypted": None, "email_enabled": None}
        return {
            "email": row[0] or None,
            "email_password_encrypted": row[1] or None,
            "email_enabled": bool(row[2]),
        }

# （必要なら）
def get_user_password_hash(user_id: int) -> Optional[str]:
    with begin() as conn:
        row = conn.execute(text('SELECT password_hash FROM "user" WHERE id = :uid'), {"uid": user_id}).fetchone()
        return row[0] if row and row[0] else None

# -----------------------------------------------------------------------------
# 書き込み系（すべて conn オプションを受け取り、_exec(..., conn=conn) で統一）
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def insert_order(
    order_id: str, symbol: str, side: str, type_: str, size: float,
    status: str, requested_at: Optional[datetime], placed_at: Optional[datetime],
    raw: Optional[Dict[str, Any]] = None, *, conn: Optional[Connection] = None
) -> None:
    stmt = sa.text("""
        INSERT INTO orders (order_id, symbol, side, type, size, status, requested_at, placed_at, raw)
        VALUES (:order_id, :symbol, :side, :type, :size, :status, :requested_at, :placed_at, :raw)
        ON CONFLICT (order_id) DO UPDATE
        SET status = EXCLUDED.status,
            placed_at = COALESCE(EXCLUDED.placed_at, orders.placed_at),
            raw = COALESCE(EXCLUDED.raw, orders.raw)
    """)
    params = dict(
        order_id=order_id, symbol=symbol, side=side, type=type_,
        size=size, status=status,
        requested_at=_as_utc(requested_at),
        placed_at=_as_utc(placed_at),
        raw=_jsonable(raw),
    )
    _exec(stmt, params, conn)

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def upsert_fill(
    fill_id: Optional[str], order_id: str, price: float, size: float,
    fee: Optional[float], executed_at: datetime, raw: Optional[Dict[str, Any]] = None,
    *, conn: Optional[Connection] = None
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
    params = dict(
        fill_id=fid, order_id=order_id, price=price, size=size,
        fee=fee, executed_at=_as_utc(executed_at), raw=_jsonable(raw)
    )
    _exec(stmt, params, conn)
    return fid

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def upsert_position(
    position_id: str,
    symbol: str,
    side: str,
    size: float,
    avg_entry_price: float,
    opened_at: Optional[datetime],
    updated_at: Optional[datetime],
    raw: Optional[Dict[str, Any]] = None,
    strategy_id: Optional[str] = None,
    *,
    user_id: Optional[int] = None,
    source: Optional[str] = None,
    conn: Optional[Connection] = None,
) -> None:
    stmt = sa.text("""
        INSERT INTO positions (
            position_id, user_id, symbol, side, size, avg_entry_price, opened_at, updated_at, raw, strategy_id, source
        )
        VALUES (
            :position_id, :user_id, :symbol, :side, :size, :avg_entry_price, :opened_at, :updated_at, :raw, :strategy_id, :source
        )
        ON CONFLICT (position_id) DO UPDATE
        SET symbol = EXCLUDED.symbol,
            side   = EXCLUDED.side,
            size   = EXCLUDED.size,
            avg_entry_price = EXCLUDED.avg_entry_price,
            opened_at  = COALESCE(EXCLUDED.opened_at, positions.opened_at),
            updated_at = COALESCE(EXCLUDED.updated_at, EXCLUDED.opened_at, positions.updated_at),
            raw = COALESCE(EXCLUDED.raw, positions.raw),
            strategy_id = COALESCE(positions.strategy_id, EXCLUDED.strategy_id),
            user_id     = COALESCE(positions.user_id, EXCLUDED.user_id),
            source      = COALESCE(positions.source, EXCLUDED.source)
    """)
    params = dict(
        position_id=position_id,
        user_id=user_id,
        symbol=symbol,
        side=side,
        size=size,
        avg_entry_price=avg_entry_price,
        opened_at=_as_utc(opened_at),
        updated_at=_as_utc(updated_at) or utcnow(),
        raw=_jsonable(raw),
        strategy_id=strategy_id,
        source=source,
     )
    _exec(stmt, params, conn)

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def insert_trade(
    trade_id: Optional[str],
    symbol: str,
    side: str,
    entry_position_id: Optional[str],
    exit_order_id: Optional[str],
    entry_price: float,
    exit_price: float,
    size: float,
    pnl: float,
    pnl_pct: float,
    holding_hours: Optional[float],
    closed_at: datetime,
    raw: Optional[Dict[str, Any]] = None,
    strategy_id: Optional[str] = None,
    *,
    user_id: Optional[int] = None,
    source: Optional[str] = None,
    conn: Optional[Connection] = None,
) -> str:
    tid = trade_id or str(uuid4())
    stmt = sa.text("""
        INSERT INTO trades (
            trade_id, user_id, symbol, side, entry_position_id, exit_order_id,
            entry_price, exit_price, size, pnl, pnl_pct, holding_hours, closed_at, raw, strategy_id, source
        )
        VALUES (
            :trade_id, :user_id, :symbol, :side, :entry_position_id, :exit_order_id,
            :entry_price, :exit_price, :size, :pnl, :pnl_pct, :holding_hours, :closed_at, :raw, :strategy_id, :source
        )
        ON CONFLICT (trade_id) DO NOTHING
    """)
    params = dict(
        trade_id=tid, user_id=user_id, symbol=symbol, side=side,
        entry_position_id=entry_position_id, exit_order_id=exit_order_id,
        entry_price=entry_price, exit_price=exit_price, size=size,
        pnl=pnl, pnl_pct=pnl_pct, holding_hours=holding_hours,
        closed_at=_as_utc(closed_at), raw=_jsonable(raw),
        strategy_id=strategy_id, source=source,
    )
    _exec(stmt, params, conn)
    return tid

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def insert_balance_snapshot(
    ts: Optional[datetime], total_balance: float,
    available_margin: Optional[float], profit_loss: Optional[float],
    raw: Optional[Dict[str, Any]] = None, *, conn: Optional[Connection] = None
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
        ts=_as_utc(ts) or utcnow(),
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

# signals は conn を受け取れるように修正（呼び出し側Txに参加）
def insert_signal(
    *,
    signal_id: Optional[str] = None,
    user_id: Optional[int],
    symbol: str,
    timeframe: str,
    side: str,
    strength_score: Optional[float] = None,
    rsi: Optional[float] = None,
    adx: Optional[float] = None,
    atr: Optional[float] = None,
    di_plus: Optional[float] = None,
    di_minus: Optional[float] = None,
    ema_fast: Optional[float] = None,
    ema_slow: Optional[float] = None,
    price: Optional[float] = None,
    generated_at: Optional[datetime] = None,
    strategy_id: Optional[str] = None,
    version: Optional[str] = None,
    status: Optional[str] = "new",
    raw: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    conn: Optional[Connection] = None,
    **extra_cols,
) -> str:
    sid = signal_id or str(uuid4())
    # 追加カラムのホワイトリスト（DDLに合わせて必要に応じ拡張）
    allowed_extras = {
        # 便利カラム（最小）
        "buy_signal", "sell_signal", "reason_tags", "rule_hits",
        # 指標系（既にDDLへ追加済み想定）
        "cci","mfi","sma_fast","sma_slow","bb_upper","bb_middle","bb_lower",
        "macd","macd_signal","macd_hist","stoch_k","stoch_d",
        "volatility","trend_strength",
        # スコア系
        "rsi_score_long","rsi_score_short","adx_score_long","adx_score_short",
        "atr_score_long","atr_score_short","cci_score_long","cci_score_short",
        "ma_score_long","ma_score_short","bb_score_long","bb_score_short",
        "score_breakdown",
    }
    # 許可されたキーだけ拾う（None はそのまま渡せる。型はDB側でNUMERIC/JSONB等）
    extras = {k: v for k, v in (extra_cols or {}).items() if k in allowed_extras}

    # ★ 追加: score_breakdown を JSON 化
    if "score_breakdown" in extras and isinstance(extras["score_breakdown"], dict):
        extras["score_breakdown"] = _jsonable(extras["score_breakdown"])

    base_cols = [
        "signal_id","user_id","symbol","timeframe","side","strength_score",
        "rsi","adx","atr","di_plus","di_minus","ema_fast","ema_slow","price",
        "generated_at","strategy_id","version","status","raw","source",
    ]
    all_cols = base_cols + list(extras.keys())
    placeholders = [f":{c}" for c in all_cols]

    sql = f"""
        INSERT INTO signals ({", ".join(all_cols)})
        VALUES ({", ".join(placeholders)})
        ON CONFLICT (signal_id) DO NOTHING
    """
    stmt = sa.text(sql)

    params = dict(
        signal_id=sid,
        user_id=user_id,
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        strength_score=strength_score,
        rsi=rsi, adx=adx, atr=atr,
        di_plus=di_plus, di_minus=di_minus,
        ema_fast=ema_fast, ema_slow=ema_slow,
        price=price,
        generated_at=_as_utc(generated_at) or utcnow(),
        strategy_id=strategy_id, version=version,
        status=status,
        raw=_jsonable(raw),
        source=source,
    )
    params.update(extras)
    _exec(stmt, params, conn)
    return sid

def update_signal_status(signal_id: str, new_status: str, *, conn: Optional[Connection] = None) -> None:
    stmt = sa.text("""
        UPDATE signals SET status = :new_status
         WHERE signal_id = :signal_id
    """)
    _exec(stmt, {"new_status": new_status, "signal_id": signal_id}, conn)

# 注文→約定（fills upsert + orders EXECUTED）も呼び出し側Txに参加可能に
def mark_order_executed_with_fill(
    order_id: str, executed_size: float, price: float, fee: Optional[float],
    executed_at: datetime, fill_raw: Optional[Dict[str, Any]] = None,
    order_raw: Optional[Dict[str, Any]] = None, *,
    # 追加: ordersを新規に立てるための最低限の情報
    symbol: Optional[str] = None,
    side: Optional[str] = None,      # "BUY"/"SELL"
    type_: str = "MARKET",
    size_hint: Optional[float] = None,
    requested_at: Optional[datetime] = None,
    placed_at: Optional[datetime] = None,
    conn: Optional[Connection] = None,
) -> None:
    def _do(c: Connection):
        # 1) orders を UPSERT（なければINSERT, あればEXECUTEDへUPDATE）
        c.execute(sa.text("""
            INSERT INTO orders (order_id, symbol, side, type, size, status, requested_at, placed_at, raw)
            VALUES (:oid, :symbol, :side, :type, :size, 'EXECUTED', :req, :plc, :raw)
            ON CONFLICT (order_id) DO UPDATE
            SET status     = 'EXECUTED',
                placed_at  = COALESCE(EXCLUDED.placed_at, orders.placed_at),
                requested_at = COALESCE(EXCLUDED.requested_at, orders.requested_at),
                raw        = COALESCE(EXCLUDED.raw, orders.raw)
        """), {
            "oid": order_id,
            "symbol": symbol,
            "side": side,
            "type": type_,
            "size": size_hint if size_hint is not None else executed_size,
            "req": _as_utc(requested_at) if requested_at else utcnow(),
            "plc": _as_utc(placed_at) if placed_at else utcnow(),
            "raw": _jsonable(order_raw),
        })

        # 2) fills をUPSERT（既存実装そのまま）
        upsert_fill(
            fill_id=None, order_id=order_id, price=price, size=executed_size,
            fee=fee, executed_at=_as_utc(executed_at) or utcnow(),
            raw=fill_raw, conn=c
        )

    if conn is not None: _do(conn)
    else:
        with begin() as c: _do(c)

def get_user_email_and_pw(user_id: int, *, decrypt: bool = False):
    """
    ユーザー1名分の email / email_enabled / (必要なら) 復号済みパスワード を取得
    戻り値: {'email': str|None, 'email_enabled': bool, 'email_password_encrypted': str|None, 'email_password_plain': str|None}
    """
    with begin() as conn:
        if decrypt:
            key = os.getenv("EMAIL_SMTP_SECRET_KEY", "")
            sql = """
                SELECT
                email,
                COALESCE(email_enabled, TRUE) AS email_enabled,
                email_password_encrypted,
                CASE
                    WHEN email_password_encrypted IS NOT NULL AND :key <> '' THEN
                    pgp_sym_decrypt(
                        decode(email_password_encrypted, 'base64'),
                        :key
                    )
                    ELSE NULL
                END AS email_password_plain
                FROM "user"
                WHERE id = :uid
            """
            row = conn.execute(text(sql), {"uid": user_id, "key": key}).mappings().first()
        else:
            sql = """
                SELECT
                  email,
                  COALESCE(email_enabled, TRUE) AS email_enabled,
                  email_password_encrypted
                FROM "user"
                WHERE id = :uid
            """
            row = conn.execute(text(sql), {"uid": user_id}).mappings().first()

    return dict(row) if row else {"email": None, "email_enabled": True, "email_password_encrypted": None, "email_password_plain": None}

def get_emails_for_users(user_ids: list[int], *, only_enabled: bool = True) -> list[str]:
    """
    複数ユーザーの通知先メールをリストで返す（NULL/無効は除外）
    """
    if not user_ids:
        return []
    with begin() as conn:
        sql = """
            SELECT email
            FROM "user"
            WHERE id = ANY(:uids)
            AND email IS NOT NULL
            AND (:only_enabled = FALSE OR COALESCE(email_enabled, TRUE) = TRUE)
        """
        rows = conn.execute(text(sql), {"uids": user_ids, "only_enabled": only_enabled}).fetchall()
    return [r[0] for r in rows]

# -----------------------------------------------------------------------------
# LINE関連（必要に応じて呼び出し側Txへ参加可のものは conn を増設）
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def upsert_user_integration_token(
    user_id: int,
    provider: str,
    token_enc: bytes,
    token_last4: str,
    status: str = "active",
    *, conn: Optional[Connection] = None,
) -> None:
    stmt = sa.text("""
        INSERT INTO user_integrations (user_id, provider, access_token_enc, token_last4, status)
        VALUES (:user_id, :provider, :access_token_enc, :token_last4, :status)
        ON CONFLICT (user_id, provider) DO UPDATE
        SET access_token_enc = EXCLUDED.access_token_enc,
            token_last4      = EXCLUDED.token_last4,
            status           = EXCLUDED.status,
            updated_at       = now()
    """)
    params = dict(
        user_id=user_id, provider=provider,
        access_token_enc=token_enc, token_last4=token_last4, status=status
    )
    _exec(stmt, params, conn)

def get_active_user_integration_token(
    user_id: int, provider: str, *, conn: Optional[Connection] = None,
) -> Optional[tuple[bytes, str]]:
    stmt = sa.text("""
        SELECT access_token_enc, token_last4
          FROM user_integrations
         WHERE user_id = :user_id
           AND provider = :provider
           AND status = 'active'
         LIMIT 1
    """)
    if conn is not None:
        row = conn.execute(stmt, {"user_id": user_id, "provider": provider}).fetchone()
    else:
        with engine.begin() as c:
            row = c.execute(stmt, {"user_id": user_id, "provider": provider}).fetchone()
    if not row:
        return None
    return row[0], row[1]

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def set_user_integration_status(
    user_id: int, provider: str, status: str, *, conn: Optional[Connection] = None,
) -> None:
    stmt = sa.text("""
        UPDATE user_integrations
           SET status = :status,
               updated_at = now()
         WHERE user_id = :user_id
           AND provider = :provider
    """)
    params = dict(user_id=user_id, provider=provider, status=status)
    _exec(stmt, params, conn)

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def upsert_line_endpoint(user_id: int, line_user_id: str, display_name: Optional[str] = None) -> None:
    stmt = sa.text("""
        INSERT INTO user_line_endpoints (user_id, line_user_id, display_name, status)
        VALUES (:user_id, :line_user_id, :display_name, 'active')
        ON CONFLICT (user_id) DO UPDATE
        SET line_user_id = EXCLUDED.line_user_id,
            display_name = COALESCE(EXCLUDED.display_name, user_line_endpoints.display_name),
            status       = 'active',
            updated_at   = now()
    """)
    _exec(stmt, {"user_id": user_id, "line_user_id": line_user_id, "display_name": display_name}, None)

def get_line_user_id(user_id: int) -> Optional[str]:
    stmt = sa.text("""
        SELECT line_user_id
          FROM user_line_endpoints
         WHERE user_id = :user_id AND status = 'active'
         LIMIT 1
    """)
    with engine.begin() as c:
        row = c.execute(stmt, {"user_id": user_id}).fetchone()
    return row[0] if row else None

def get_line_user_ids_for_users(user_ids: Sequence[int]) -> list[str]:
    if not user_ids:
        return []
    stmt = sa.text("""
        SELECT line_user_id
          FROM user_line_endpoints
         WHERE user_id = ANY(:ids) AND status = 'active'
    """)
    with engine.begin() as c:
        rows = c.execute(stmt, {"ids": list(user_ids)}).fetchall()
    return [r[0] for r in rows]

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def set_line_endpoint_status(user_id: int, status: str) -> None:
    stmt = sa.text("""
        UPDATE user_line_endpoints
           SET status = :status, updated_at = now()
         WHERE user_id = :user_id
    """)
    _exec(stmt, {"user_id": user_id, "status": status}, None)

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def upsert_line_channel_token(provider_key: str, access_token_enc: bytes, status: str = "active") -> None:
    stmt = sa.text("""
        INSERT INTO line_channels (provider_key, access_token_enc, status)
        VALUES (:k, :t, :s)
        ON CONFLICT (provider_key) DO UPDATE
        SET access_token_enc = EXCLUDED.access_token_enc,
            status = EXCLUDED.status,
            updated_at = now()
    """)
    _exec(stmt, {"k": provider_key, "t": access_token_enc, "s": status}, None)

def get_line_channel_token(provider_key: str) -> Optional[bytes]:
    stmt = sa.text("""
        SELECT access_token_enc
          FROM line_channels
         WHERE provider_key = :k AND status = 'active'
         LIMIT 1
    """)
    with engine.begin() as c:
        row = c.execute(stmt, {"k": provider_key}).fetchone()
    return row[0] if row else None

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def upsert_price_cache(
    symbol: str,
    price: float,
    ts: datetime,
    *, conn: Optional[Connection] = None,
) -> None:
    """
    現在価格の簡易キャッシュをUPSERT（symbol単位で最新値を保持）
    期待スキーマ:
        CREATE TABLE IF NOT EXISTS price_cache (
            symbol text PRIMARY KEY,
            price numeric,
            ts timestamptz NOT NULL
        );
    """
    stmt = sa.text("""
        INSERT INTO price_cache (symbol, price, ts)
        VALUES (:symbol, :price, :ts)
        ON CONFLICT (symbol) DO UPDATE
        SET price = EXCLUDED.price,
            ts    = EXCLUDED.ts
    """)
    params = dict(symbol=symbol, price=price, ts=_as_utc(ts) or utcnow())
    _exec(stmt, params, conn)

# -----------------------------------------------------------------------------
# ユーティリティ
# -----------------------------------------------------------------------------
def ping_db_once():
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("SELECT current_database() as db, current_user as usr, now() as ts"))
        insert_error("boot/ping", "ping ok", raw={"url": _redact_url(DATABASE_URL)})
    except Exception as e:
        insert_error("boot/ping", f"ping failed: {e}")
