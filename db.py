from __future__ import annotations
import os
import json
from contextlib import contextmanager
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any
from uuid import uuid4
from pathlib import Path  
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
import sqlalchemy as sa
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import IntegrityError
try:
    from psycopg2.extras import Json as _PsycoJson
except Exception:
    _PsycoJson = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # ←絶対パスで .env を読む
DATABASE_URL = os.getenv("DATABASE_URL")

JST = timezone(timedelta(hours=9))

def _as_utc(dt):
    """naive→UTC付与 / JST→UTC変換 / すでにtz付き→UTC変換"""
    if dt.tzinfo is None:
        # naive は UTC として扱う（必要ならここを JST 想定に変える）
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def get_trades_between(start_dt, end_dt, symbols=None, min_abs_pnl=None):
    """
    [start_dt, end_dt) に closed_at が入る trades を返却（辞書の配列）。
    - start_dt/end_dt: datetime（naive でも tz 付きでもOK）
    - symbols: 例 ["ltc_jpy","eth_jpy"]（None なら全件）
    - min_abs_pnl: 絶対値でこのPnLを下限にフィルタ（例: 0.0）
    """
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
    # dict のリストで返す
    return [dict(r) for r in rows]

def _json_param(v):
    if v is None:
        return None
    # psycopg2 の Json ラッパーが使えるなら最優先
    if _PsycoJson and isinstance(v, (dict, list)):
        return _PsycoJson(v)
    # フォールバック：文字列化
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v

def get_trades_for_day_jst(day_dt):
    """
    JST の 00:00〜24:00 を作り、その範囲の trades を返す（辞書配列）。
    day_dt: datetime（naive 可）— naive の場合は JST とみなす
    """
    day = day_dt
    if day.tzinfo is None:
        day = day.replace(tzinfo=JST)
    else:
        day = day.astimezone(JST)

    start_jst = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end_jst   = start_jst + timedelta(days=1)

    # UTC に直してからクエリ
    return get_trades_between(start_jst, end_jst)

# === 起動時に接続先をログ出力（パスワード隠し） ===
def _redact_url(u: str) -> str:
    # postgresql+psycopg2://user:***@host:5432/db という形にする
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

try:
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("DB target: %s", _redact_url(DATABASE_URL or "unset"))
except Exception:
    pass

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

try:
    import numpy as np
except Exception:
    np = None

def _to_plain(obj):
    """raw を JSON セーフな素の Python 型へ再帰変換"""
    # None / プリミティブ
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # 日付・日時
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # Decimal
    if isinstance(obj, Decimal):
        # 精度を保ちたいなら str(obj) でもOK
        return float(obj)

    # numpy 系
    if np is not None:
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

    # 辞書
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}

    # リスト/タプル/セット
    if isinstance(obj, (list, tuple, set)):
        return [_to_plain(v) for v in obj]

    # それ以外はとりあえず文字列化（最後の保険）
    return str(obj)

def _jsonable(v):
    from psycopg2.extras import Json as _PsycoJson  # 既にtry import済みなら不要
    # コンテナは plain 化してそのまま返す（Jsonラッパ or プレーン）
    if isinstance(v, (dict, list, tuple, set)):
        plain = _to_plain(v)
        return _PsycoJson(plain) if _PsycoJson else plain
    # スカラはプレーンに
    return _to_plain(v)
    
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

def ping_db_once():
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("SELECT current_database() as db, current_user as usr, now() as ts"))
        # ここで errors にも 1 行残して「ping OK」を見える化（同じDBに入るか確認用）
        insert_error("boot/ping", "ping ok", raw={"url": _redact_url(DATABASE_URL)})
    except Exception as e:
        insert_error("boot/ping", f"ping failed: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def upsert_user_integration_token(
    user_id: int,
    provider: str,
    token_enc: bytes,       # ← 暗号化済（Fernet等）。平文はここに渡さない
    token_last4: str,
    status: str = "active",
    conn: Optional[Connection] = None,
) -> None:
    """
    暗号化済みトークンを保存（既存があれば更新）。1ユーザー×1プロバイダ=1行で維持。
    """
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
    user_id: int,
    provider: str,
    conn: Optional[Connection] = None,
) -> Optional[tuple[bytes, str]]:
    """
    アクティブな暗号化済みトークンを取得（見つからなければ None）。
    戻り値: (token_enc, token_last4)
    """
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
    user_id: int,
    provider: str,
    status: str,   # 'revoked' | 'error' | 'active'
    conn: Optional[Connection] = None,
) -> None:
    """
    ステータス更新（失効/エラー時などに使用）。
    """
    stmt = sa.text("""
        UPDATE user_integrations
           SET status = :status,
               updated_at = now()
         WHERE user_id = :user_id
           AND provider = :provider
    """)
    params = dict(user_id=user_id, provider=provider, status=status)
    _exec(stmt, params, conn)

# 1) ユーザーごとの送信先（LINE userId=U...）を登録/更新
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

# 2) 単一ユーザーのLINE userId を取得（なければ None）
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

# 3) 複数ユーザー分のLINE userIdを配列で取得（multicast 用）
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

# 4) エンドポイントの状態更新（例: ブロック検出時に 'blocked' へ）
@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
def set_line_endpoint_status(user_id: int, status: str) -> None:
    stmt = sa.text("""
        UPDATE user_line_endpoints
           SET status = :status, updated_at = now()
         WHERE user_id = :user_id
    """)
    _exec(stmt, {"user_id": user_id, "status": status}, None)

# --- （任意）チャネル・トークンをDBで運用する場合 --------------------------
#  * 現状は .env の LINE_CHANNEL_ACCESS_TOKEN でOK。将来マルチテナント時に活用。

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