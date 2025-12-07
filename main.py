# app/main.py
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import sqlalchemy as sa
from db import engine
from fastapi.responses import StreamingResponse, JSONResponse
import io, csv
import json
from app.routers import public_performance, strategies, virtual, monitor
import logging

logger = logging.getLogger("uvicorn.error")

JST = timezone(timedelta(hours=9))

app = FastAPI(
    title="Signal Service API",
    version="0.1.0",
    description="Public endpoints for metrics and signals."
)

app.include_router(public_performance.router, prefix="/api")
app.include_router(strategies.router)
app.include_router(virtual.router)
app.include_router(monitor.router)

# --- CORS（必要に応じて調整） ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://198-13-61-186.sslip.io"],  # ← 自サイトのみ許可
    allow_credentials=False,  # Cookie/SameSite 等を使わないなら False でOK
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],  # OPTIONS も必須（プリフライト用）
    allow_headers=["Content-Type", "Authorization"],  # 必要なヘッダだけ許可
)

# --- 型: JSONに安全な形へ整形 ---
def _to_plain(x):
    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_plain(v) for v in x]
    return str(x)

# ---- 便利: ISO 8601 文字列をUTCのaware datetimeに変換 ----
def _parse_iso8601_or_422(s: str) -> datetime:
    try:
        x = s.strip()
        if x.endswith("Z"):
            x = x[:-1] + "+00:00"
        dt = datetime.fromisoformat(x)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid ISO8601 datetime: {s}")

# --- Pydantic モデル（出力用） ---
class SymbolsEntry(BaseModel):
    trades: int
    win_rate: Optional[float] = Field(None, description="0..1")
    avg_pnl_pct: Optional[float] = None

class PublicMetricsOut(BaseModel):
    period_start: datetime
    period_end: datetime
    total_trades: int
    win_rate: Optional[float] = None
    avg_pnl_pct: Optional[float] = None
    symbols: Dict[str, SymbolsEntry] = Field(default_factory=dict)

class RuleCreateIn(BaseModel):
    symbol: str
    timeframe: str
    score_col: str
    op: str
    v1: Optional[float] = None
    v2: Optional[float] = None
    target_side: str
    # 以下はクライアント指定があっても無視し、サーバ側で固定値を上書きします
    action: Optional[str] = None
    priority: Optional[int] = None
    version: Optional[str] = None
    user_id: Optional[str] = None
    strategy_id: Optional[str] = None
    active: Optional[bool] = None
    notes: Optional[str] = ""

class RuleUpdateIn(BaseModel):
    # 部分更新（nullable許容）
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    score_col: Optional[str] = None
    op: Optional[str] = None
    v1: Optional[float] = None
    v2: Optional[float] = None
    target_side: Optional[str] = None
    action: Optional[str] = None
    priority: Optional[int] = None
    active: Optional[bool] = None
    version: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    user_id: Optional[str] = None
    strategy_id: Optional[str] = None
    notes: Optional[str] = None

# --- クエリ（最新1件） ---
LATEST_SQL = sa.text("""
    SELECT
      period_start, period_end, total_trades, win_rate, avg_pnl_pct, symbols
    FROM public_metrics
    ORDER BY period_end DESC
    LIMIT 1
""")

# === signals 系（source='real' に限定） ===
LATEST_SIGNALS_SQL = sa.text("""
  SELECT symbol, side, price, generated_at, strength_score
  FROM signals
  WHERE source = 'real'
  ORDER BY generated_at DESC
  LIMIT :limit
""")

COUNT_SIGNALS_SQL = sa.text("""
    SELECT COUNT(*)::bigint AS cnt
    FROM signals
    WHERE source = 'real'
      AND (:symbol IS NULL OR lower(symbol) = lower(:symbol))
      AND (:since  IS NULL OR generated_at >= :since)
""")

FILTERED_SIGNALS_SQL = sa.text("""
    SELECT symbol, side, price, generated_at, strength_score
    FROM signals
    WHERE source = 'real'
      AND (:symbol IS NULL OR lower(symbol) = lower(:symbol))
      AND (:since  IS NULL OR generated_at >= :since)
    ORDER BY generated_at DESC
    LIMIT :limit OFFSET :offset
""")

# === 日次系列（メトリクスは従来どおり public_metrics_daily を参照） ===
DAILY_SERIES_SQL = sa.text("""
SELECT d AS metric_date,
       COALESCE(m.total_trades, 0) AS total_trades,
       m.win_rate,
       m.avg_pnl_pct
FROM generate_series(:start_date, :end_date, INTERVAL '1 day') AS d
LEFT JOIN public_metrics_daily m
  ON m.metric_date = d
 AND m.source = 'real'  -- ★実運用のみ
ORDER BY d ASC
""").bindparams(
    sa.bindparam("start_date", type_=sa.Date),
    sa.bindparam("end_date", type_=sa.Date),
)

# === signals のCSV（real 限定） ===
EXPORT_SIGNALS_SQL = sa.text("""
    SELECT symbol, side, price, generated_at, strength_score
    FROM signals
    WHERE source = 'real'
      AND (:symbol IS NULL OR lower(symbol) = lower(:symbol))
      AND (:since  IS NULL OR generated_at >= :since)
    ORDER BY generated_at DESC
    LIMIT :limit
""")

# 期間のdate列を作るSQL（既存の bindparam 版）
DAILY_SERIES_DATE_SQL = sa.text("""
SELECT d AS metric_date
FROM generate_series(:start_date, :end_date, INTERVAL '1 day') AS d
ORDER BY d ASC
""").bindparams(
    sa.bindparam("start_date", type_=sa.Date),
    sa.bindparam("end_date", type_=sa.Date),
)

# public_metrics_daily から日付範囲で rows を取る
FETCH_DAILY_ROWS_SQL = sa.text("""
SELECT metric_date, total_trades, symbols
FROM public_metrics_daily
WHERE metric_date BETWEEN :start_date AND :end_date
  AND source = 'real'  -- ★追加
ORDER BY metric_date ASC
""").bindparams(
    sa.bindparam("start_date", type_=sa.Date),
    sa.bindparam("end_date", type_=sa.Date),
)

# === シンボル候補（real 限定） ===
SYMBOLS_SQL = sa.text("""
SELECT DISTINCT LOWER(symbol) AS symbol
FROM signals
WHERE source = 'real'
  AND ((:days IS NULL) OR generated_at >= (NOW() AT TIME ZONE 'UTC') - (:days || ' days')::interval)
ORDER BY 1 ASC
""").bindparams(
    sa.bindparam("days", type_=sa.Integer)
)

# === オープンポジション（real 限定・未クローズ判定は trades.source='real' に限定） ===
OPEN_POSITIONS_SQL = sa.text("""
    SELECT
      p.position_id,
      p.symbol,
      p.side,
      p.size,
      p.avg_entry_price AS entry_price,
      p.opened_at::timestamptz AS entry_time
    FROM positions p
    WHERE
      p.source = 'real'
      AND COALESCE(p.size, 0) <> 0
      AND lower(p.side) IN ('long', 'short')
      AND NOT EXISTS (
        SELECT 1 FROM trades t
        WHERE t.entry_position_id = p.position_id
          AND t.source = 'real'
          AND (t.closed_at IS NOT NULL OR t.exit_price IS NOT NULL)
      )
      AND p.opened_at >= NOW() - INTERVAL '3 days'
      AND (:symbol IS NULL OR lower(p.symbol) = lower(:symbol))
    ORDER BY p.opened_at DESC
""")

OPEN_WITH_PRICE_SQL = sa.text("""
    SELECT
      p.position_id,
      p.symbol,
      p.side,
      p.size,
      p.avg_entry_price AS entry_price,
      p.opened_at::timestamptz AS entry_time,
      CASE
      WHEN pc.ts >= NOW() - INTERVAL '10 minutes' THEN pc.price
      ELSE NULL
      END AS current_price,
      CASE
      WHEN pc.ts < NOW() - INTERVAL '10 minutes' THEN NULL
      WHEN pc.price IS NULL OR p.avg_entry_price IS NULL OR p.size = 0 THEN NULL
      WHEN lower(p.side) = 'long'  THEN (pc.price / NULLIF(p.avg_entry_price,0) - 1) * 100
      WHEN lower(p.side) = 'short' THEN (NULLIF(p.avg_entry_price,0) / pc.price - 1) * 100
      ELSE NULL
      END AS unrealized_pnl_pct,
      pc.ts AS price_ts
    FROM positions p
    LEFT JOIN price_cache pc
      ON lower(btrim(pc.symbol)) = lower(btrim(p.symbol))
    WHERE
      p.source = 'real'
      AND COALESCE(p.size, 0) <> 0
      AND NOT EXISTS (
        SELECT 1 FROM trades t
        WHERE t.entry_position_id = p.position_id
          AND t.source = 'real'
          AND (t.closed_at IS NOT NULL OR t.exit_price IS NOT NULL)
      )
      AND (:symbol IS NULL OR lower(p.symbol) = lower(:symbol))
      AND p.opened_at >= (NOW() AT TIME ZONE 'UTC') - INTERVAL '3 days'
    ORDER BY p.opened_at DESC
""").bindparams(
    sa.bindparam("symbol", type_=sa.String)
)

# === KPI 7days（trades を real 限定で集計） ===
OPEN_KPI_SQL = sa.text("""
    SELECT
      COUNT(*)::int AS trade_count,
      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::int AS win_count,
      SUM(pnl)::numeric AS total_pnl,
      AVG(pnl_pct)::numeric AS avg_pnl_pct,
      AVG(holding_hours)::numeric AS avg_holding_hours
    FROM trades
    WHERE source = 'real'
      AND closed_at >= NOW() - INTERVAL '7 days'
      AND (:symbol IS NULL OR lower(symbol) = lower(:symbol))
""")

# === 本日の確定トレード（JST基準 / opened_at は算出） ===
TRADES_TODAY_SQL = sa.text("""
SELECT
  t.trade_id,
  t.symbol,
  t.side,
  t.entry_position_id,
  t.exit_order_id,
  t.entry_price,
  t.exit_price,
  t.size,
  t.pnl,
  t.pnl_pct,
  t.holding_hours,
  t.closed_at,
  t.strategy_id,
  t.source,
  t.user_id
FROM trades t
WHERE t.source = :source
  AND t.closed_at IS NOT NULL
  AND (t.closed_at AT TIME ZONE 'Asia/Tokyo')::date = :target_date
  AND (:symbol IS NULL OR LOWER(t.symbol) = LOWER(:symbol))
ORDER BY t.closed_at DESC
""")

# === real-only 日次（全体集計）: trades 由来 ===
DAILY_SERIES_FROM_TRADES_REAL = sa.text("""
WITH days AS (
  SELECT d::date AS metric_date
  FROM generate_series(:start_date, :end_date, INTERVAL '1 day') AS d
),
daily_real AS (
  SELECT
    t.closed_at::date                          AS d,
    COUNT(*)::int                              AS total_trades,
    SUM(t.pnl)::numeric                        AS total_pnl,        -- ★追加：損益合計
    AVG(CASE WHEN t.pnl > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
    AVG(t.pnl_pct)::numeric                    AS avg_pnl_pct
  FROM trades t
  WHERE t.closed_at IS NOT NULL
    AND t.source = :source
    AND t.closed_at::date BETWEEN :start_date AND :end_date
  GROUP BY t.closed_at::date
)
SELECT
  d.metric_date,
  COALESCE(r.total_trades, 0) AS total_trades,
  r.total_pnl,                    -- ★追加
  r.win_rate,
  r.avg_pnl_pct
FROM days d
LEFT JOIN daily_real r ON r.d = d.metric_date
ORDER BY d.metric_date ASC
""").bindparams(
    sa.bindparam("start_date", type_=sa.Date),
    sa.bindparam("end_date", type_=sa.Date),
    sa.bindparam("source", type_=sa.String),
)

# === real-only 日次（シンボル別）: trades 由来 ===
DAILY_BY_SYMBOL_FROM_TRADES_REAL = sa.text("""
WITH days AS (
  SELECT d::date AS metric_date
  FROM generate_series(:start_date, :end_date, INTERVAL '1 day') AS d
),
daily_real AS (
  SELECT
    t.closed_at::date                          AS d,
    LOWER(t.symbol)                            AS symbol,
    COUNT(*)::int                              AS trades,
    AVG(CASE WHEN t.pnl > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
    AVG(t.pnl_pct)::numeric                    AS avg_pnl_pct,
    SUM(t.pnl)::numeric                        AS total_pnl     -- ★ 追加
  FROM trades t
  WHERE t.closed_at IS NOT NULL
    AND t.source = :source                     -- 'real'
    AND t.closed_at::date BETWEEN :start_date AND :end_date
    AND (:symbol IS NULL OR LOWER(t.symbol) = LOWER(:symbol))
  GROUP BY t.closed_at::date, LOWER(t.symbol)
)
SELECT
  days.metric_date,
  daily_real.symbol,
  COALESCE(daily_real.trades, 0)   AS trades,
  daily_real.win_rate,
  daily_real.avg_pnl_pct,
  COALESCE(daily_real.total_pnl,0) AS total_pnl   -- ★ 追加
FROM days
LEFT JOIN daily_real
  ON daily_real.d = days.metric_date
ORDER BY days.metric_date ASC, daily_real.symbol NULLS LAST
""").bindparams(
    sa.bindparam("start_date", type_=sa.Date),
    sa.bindparam("end_date", type_=sa.Date),
    sa.bindparam("source", type_=sa.String),
    sa.bindparam("symbol", type_=sa.String),
)

RULES_LIST_SQL_BASE = """
SELECT
  id, symbol, timeframe, score_col, op, v1, v2,
  target_side, action, priority, active, version,
  valid_from, valid_to, user_id, strategy_id, notes
FROM signal_rule_thresholds
WHERE 1=1
  AND (:symbol     IS NULL OR LOWER(symbol) = LOWER(:symbol))
  AND (:timeframe  IS NULL OR timeframe = :timeframe)
  AND (:active_txt IS NULL OR (active = CASE WHEN :active_txt='true' THEN TRUE WHEN :active_txt='false' THEN FALSE ELSE active END))
  AND (:user_id    IS NULL OR user_id = :user_id)
  AND (:strategy_id IS NULL OR strategy_id = :strategy_id)
  AND (:version    IS NULL OR version = :version)
"""

INSERT_RULE_SQL = sa.text("""
INSERT INTO signal_rule_thresholds (
  symbol, timeframe, score_col, op, v1, v2, target_side,
  action, priority, active, version, valid_from, valid_to,
  user_id, strategy_id, notes
) VALUES (
  :symbol, :timeframe, :score_col, :op, :v1, :v2, :target_side,
  :action, :priority, :active, :version, NOW() AT TIME ZONE 'UTC', NULL,
  :user_id, :strategy_id, :notes
)
RETURNING id, symbol, timeframe, score_col, op, v1, v2, target_side, action, priority, active, version,
          valid_from, valid_to, user_id, strategy_id, notes
""")

UPDATE_RULE_SQL_BASE = """
UPDATE signal_rule_thresholds
SET {sets}
WHERE id = :id
RETURNING id, symbol, timeframe, score_col, op, v1, v2, target_side, action, priority, active, version,
          valid_from, valid_to, user_id, strategy_id, notes
"""

LOGICAL_DELETE_SQL = sa.text("""
UPDATE signal_rule_thresholds
SET valid_to = NOW() AT TIME ZONE 'UTC',
    active   = FALSE
WHERE id = :id
RETURNING id, symbol, timeframe, score_col, op, v1, v2, target_side, action, priority, active, version,
          valid_from, valid_to, user_id, strategy_id, notes
""")

CANDLES_15MIN_SQL = sa.text("""
SELECT
  ts,
  open,
  high,
  low,
  close,
  volume
FROM public.candles_15min
WHERE lower(symbol) = lower(:symbol)
  AND ts >= (NOW() AT TIME ZONE 'UTC') - (:days || ' days')::interval
ORDER BY ts ASC
""").bindparams(
    sa.bindparam("symbol", type_=sa.String),
    sa.bindparam("days", type_=sa.Integer),
)

SR_ZONES_SQL = sa.text("""
SELECT
  zone_id,
  user_id,
  symbol,
  timeframe,
  lookback_days,
  zone_type,
  price_center,
  price_low,
  price_high,
  touches,
  strength,
  last_touched_at,
  generated_at,
  source
FROM support_resistance_zones
WHERE
  source = 'sim'
  AND lower(symbol) = lower(:symbol)
  AND timeframe = :timeframe
  AND lookback_days = :lookback_days
ORDER BY price_center ASC
""").bindparams(
    sa.bindparam("symbol", type_=sa.String),
    sa.bindparam("timeframe", type_=sa.String),
    sa.bindparam("lookback_days", type_=sa.Integer),
)


# ------------------- エンドポイント -------------------

# q（簡易全文）条件を動的に足す
def _rules_q_clause():
    # 数値列は::text化が分岐要るが、PostgreSQLなら to_char でもOK
    return """ AND (
      LOWER(COALESCE(symbol,'')) LIKE :q
      OR LOWER(COALESCE(timeframe,'')) LIKE :q
      OR LOWER(COALESCE(score_col,'')) LIKE :q
      OR LOWER(COALESCE(op,'')) LIKE :q
      OR LOWER(COALESCE(target_side,'')) LIKE :q
      OR LOWER(COALESCE(action,'')) LIKE :q
      OR LOWER(COALESCE(version,'')) LIKE :q
      OR LOWER(COALESCE(user_id,'')) LIKE :q
      OR LOWER(COALESCE(strategy_id,'')) LIKE :q
      OR LOWER(COALESCE(notes,'')) LIKE :q
    )"""

# valid_to IS NULL のみ
def _rules_only_open_ended_clause():
    return " AND valid_to IS NULL "

def _rules_order_clause(sort: str):
    allowed = {
        "id": "id",
        "symbol": "symbol",
        "timeframe": "timeframe",
        "score_col": "score_col",
        "op": "op",
        "target_side": "target_side",
        "action": "action",
        "priority": "priority",
        "active": "active",
        "version": "version",
        "valid_from": "valid_from",
        "valid_to": "valid_to",
        "user_id": "user_id",
        "strategy_id": "strategy_id",
    }
    col = allowed.get(sort or "priority", "priority")
    # 既定は priority ASC, id ASC
    if col == "priority":
        return " ORDER BY priority ASC, id ASC "
    return f" ORDER BY {col} ASC, id ASC "

from fastapi import Query

@app.get("/admin/signal-rules")
def list_signal_rules(
    symbol: str | None = Query(None),
    timeframe: str | None = Query(None),
    active: str | None = Query(None, pattern="^(true|false)$"),
    user_id: str | None = Query(None),
    strategy_id: str | None = Query(None),
    version: str | None = Query(None),
    only_open_ended: bool = Query(False),
    q: str | None = Query(None, description="簡易全文検索（部分一致・小文字化）"),
    sort: str | None = Query("priority"),
    page: int = Query(1, ge=1),
    limit: int = Query(200, ge=1, le=1000),
):
    base = RULES_LIST_SQL_BASE
    params = {
        "symbol": symbol,
        "timeframe": timeframe,
        "active_txt": active,
        "user_id": user_id,
        "strategy_id": strategy_id,
        "version": version,
    }

    if q:
        base += _rules_q_clause()
        params["q"] = f"%{q.lower()}%"
    if only_open_ended:
        base += _rules_only_open_ended_clause()

    order = _rules_order_clause(sort)
    offset = (page - 1) * limit

    sql = sa.text(base + order + " LIMIT :limit OFFSET :offset ")
    with engine.begin() as conn:
        rows = conn.execute(sql, {**params, "limit": limit, "offset": offset}).mappings().all()

    # JSON整形
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "score_col": r["score_col"],
            "op": r["op"],
            "v1": float(r["v1"]) if r["v1"] is not None else None,
            "v2": float(r["v2"]) if r["v2"] is not None else None,
            "target_side": r["target_side"],
            "action": r["action"],
            "priority": r["priority"],
            "active": bool(r["active"]),
            "version": r["version"],
            "valid_from": r["valid_from"].isoformat() if r["valid_from"] else None,
            "valid_to": r["valid_to"].isoformat() if r["valid_to"] else None,
            "user_id": r.get("user_id"),
            "strategy_id": r.get("strategy_id"),
            "notes": r.get("notes"),
        })
    return out

@app.get("/public/metrics", response_model=PublicMetricsOut)
def get_public_metrics():
    try:
        with engine.begin() as conn:
            row = conn.execute(LATEST_SQL).mappings().first()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not row:
        raise HTTPException(status_code=404, detail="no public metrics found")

    payload = {
        "period_start": row["period_start"],
        "period_end": row["period_end"],
        "total_trades": row["total_trades"],
        "win_rate": row["win_rate"],
        "avg_pnl_pct": row["avg_pnl_pct"],
        "symbols": _to_plain(row["symbols"]) or {},
    }
    return payload

@app.get("/public/signals")
def get_public_signals(
    response: Response,
    symbol: Optional[str] = Query(None, min_length=1, max_length=50, description="例: BTCUSDT / sol_jpy"),
    since: Optional[str]  = Query(None, description="ISO8601 例: 2025-09-01T00:00:00Z"),
    page: int             = Query(1, ge=1),
    limit: int            = Query(50, ge=1, le=200),
):
    since_dt = _parse_iso8601_or_422(since) if since else None
    offset = (page - 1) * limit

    try:
        with engine.begin() as conn:
            total = conn.execute(COUNT_SIGNALS_SQL, {
                "symbol": symbol,
                "since": since_dt,
            }).scalar_one()

            rows = conn.execute(FILTERED_SIGNALS_SQL, {
                "symbol": symbol,
                "since": since_dt,
                "limit": limit,
                "offset": offset,
            }).mappings().all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Limit"] = str(limit)
    response.headers["X-Has-Next"] = "true" if (offset + len(rows)) < total else "false"

    return [{
        "symbol": r["symbol"],
        "side": r["side"],
        "price": float(r["price"]) if r["price"] is not None else None,
        "generated_at": r["generated_at"].isoformat() if r["generated_at"] else None,
        "strength_score": float(r["strength_score"]) if r["strength_score"] is not None else None,
    } for r in rows]

@app.get("/public/performance/daily")
def get_performance_daily(
    days: int = Query(30, ge=1, le=365),
    source: str = Query("real", pattern="^(real|virtual)$")  # ← 追加：将来の切り替え用、今は 'real' がデフォ
):
    today_jst = datetime.now(JST).date()
    start_date = today_jst - timedelta(days=days - 1)
    end_date = today_jst

    try:
        with engine.begin() as conn:
            rows = conn.execute(DAILY_SERIES_FROM_TRADES_REAL, {
                "start_date": start_date,
                "end_date": end_date,
                "source": source,
            }).mappings().all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return [{
        "date": r["metric_date"].isoformat(),
        "total_trades": int(r["total_trades"]) if r["total_trades"] is not None else 0,
        "total_pnl": float(r["total_pnl"]) if r["total_pnl"] is not None else 0.0,  # ★追加
        "win_rate": float(r["win_rate"]) if r["win_rate"] is not None else None,
        "avg_pnl_pct": float(r["avg_pnl_pct"]) if r["avg_pnl_pct"] is not None else None,
    } for r in rows]

@app.get("/public/export/performance/daily.csv")
def export_performance_daily_csv(
    days: int = Query(30, ge=1, le=365),
    source: str = Query("real", pattern="^(real|virtual)$")
):
    today_jst = datetime.now(JST).date()
    start_date = today_jst - timedelta(days=days - 1)
    end_date = today_jst

    with engine.begin() as conn:
        rows = conn.execute(DAILY_SERIES_FROM_TRADES_REAL, {
            "start_date": start_date,
            "end_date": end_date,
            "source": source,
        }).mappings().all()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date", "total_trades", "win_rate", "avg_pnl_pct"])
    for r in rows:
        w.writerow([
            r["metric_date"].isoformat(),
            int(r["total_trades"]) if r["total_trades"] is not None else 0,
            float(r["win_rate"]) if r["win_rate"] is not None else "",
            float(r["avg_pnl_pct"]) if r["avg_pnl_pct"] is not None else "",
        ])

    csv_bytes = buf.getvalue().encode("utf-8")
    filename = f"performance_daily_{source}_{days}d.csv"
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'}
    )

@app.get("/public/export/signals.csv")
def export_signals_csv(
    symbol: Optional[str] = Query(None, min_length=1, max_length=50),
    since: Optional[str]  = Query(None, description="ISO8601（例: 2025-09-01T00:00:00Z）"),
    limit: int            = Query(1000, ge=1, le=10000),
):
    since_dt = _parse_iso8601_or_422(since) if since else None

    with engine.begin() as conn:
        rows = conn.execute(EXPORT_SIGNALS_SQL, {
            "symbol": symbol,
            "since": since_dt,
            "limit": limit
        }).mappings().all()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["symbol", "side", "price", "generated_at", "strength_score"])
    for r in rows:
        gen = r["generated_at"].isoformat() if r["generated_at"] else ""
        w.writerow([
            r["symbol"],
            r["side"],
            float(r["price"]) if r["price"] is not None else "",
            gen,
            float(r["strength_score"]) if r["strength_score"] is not None else "",
        ])

    csv_bytes = buf.getvalue().encode("utf-8")
    filename = "signals_last.csv"
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'}
    )

def _pick_symbol_case_insensitive(symbols_obj: Dict[str, Any], symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    if not symbols_obj or not symbol:
        return None
    if symbol in symbols_obj:
        return symbols_obj[symbol]
    sym_l = symbol.lower()
    for k, v in symbols_obj.items():
        if isinstance(k, str) and k.lower() == sym_l:
            return v
    return None

@app.get("/public/performance/daily/by-symbol")
def get_performance_daily_by_symbol(
    symbol: Optional[str] = Query(None, min_length=1, max_length=50, description="省略時は全シンボル"),
    days: int = Query(30, ge=1, le=365),
    source: str = Query("real", pattern="^(real|virtual)$"),
):
    today_jst = datetime.now(JST).date()
    start_date = today_jst - timedelta(days=days - 1)
    end_date = today_jst

    with engine.begin() as conn:
        rows = conn.execute(DAILY_BY_SYMBOL_FROM_TRADES_REAL, {
            "start_date": start_date,
            "end_date": end_date,
            "symbol": symbol,
            "source": source,
        }).mappings().all()

    if symbol:
        out = []
        for r in rows:
            out.append({
                "date": r["metric_date"].isoformat(),
                "total_trades": int(r["trades"]) if r["trades"] is not None else 0,
                "win_rate": float(r["win_rate"]) if r["win_rate"] is not None else None,
                "avg_pnl_pct": float(r["avg_pnl_pct"]) if r["avg_pnl_pct"] is not None else None,
                "total_pnl": float(r["total_pnl"]) if r["total_pnl"] is not None else 0.0,  # ★ 追加
            })
        return out

    items_by_symbol: dict[str, list] = {}
    for r in rows:
        sym = r["symbol"]
        if sym is None:
            continue
        items_by_symbol.setdefault(sym, []).append({
            "date": r["metric_date"].isoformat(),
            "total_trades": int(r["trades"]) if r["trades"] is not None else 0,
            "win_rate": float(r["win_rate"]) if r["win_rate"] is not None else None,
            "avg_pnl_pct": float(r["avg_pnl_pct"]) if r["avg_pnl_pct"] is not None else None,
            "total_pnl": float(r["total_pnl"]) if r["total_pnl"] is not None else 0.0,  # ★ 追加
        })
    return {"days": days, "items_by_symbol": items_by_symbol}

@app.get("/public/export/performance/daily_by_symbol.csv")
def export_performance_daily_by_symbol_csv(
    symbol: Optional[str] = Query(None, min_length=1, max_length=50, description="省略時は全シンボル"),
    days: int = Query(30, ge=1, le=365),
    source: str = Query("real", pattern="^(real|virtual)$")
):
    today_jst = datetime.now(JST).date()
    start_date = today_jst - timedelta(days=days - 1)
    end_date = today_jst

    with engine.begin() as conn:
        rows = conn.execute(DAILY_BY_SYMBOL_FROM_TRADES_REAL, {
            "start_date": start_date,
            "end_date": end_date,
            "symbol": symbol,
            "source": source,
        }).mappings().all()

    buf = io.StringIO()
    w = csv.writer(buf)

    if symbol:
        w.writerow(["date", "trades", "win_rate", "avg_pnl_pct", "total_pnl"])
        for r in rows:
            w.writerow([
                r["metric_date"].isoformat(),
                int(r["trades"]) if r["trades"] is not None else 0,
                float(r["win_rate"]) if r["win_rate"] is not None else "",
                float(r["avg_pnl_pct"]) if r["avg_pnl_pct"] is not None else "",
                float(r["total_pnl"]) if r["total_pnl"] is not None else "",        # ★ 追加
            ])
        filename = f"performance_daily_{symbol}_{source}_{days}d.csv"
    else:
        w.writerow(["date", "symbol", "trades", "win_rate", "avg_pnl_pct", "total_pnl"])
        for r in rows:
            if r["symbol"] is None:
                continue
            w.writerow([
                r["metric_date"].isoformat(),
                r["symbol"],
                int(r["trades"]) if r["trades"] is not None else 0,
                float(r["win_rate"]) if r["win_rate"] is not None else "",
                float(r["avg_pnl_pct"]) if r["avg_pnl_pct"] is not None else "",
                float(r["total_pnl"]) if r["total_pnl"] is not None else "",        # ★ 追加
            ])
        filename = f"performance_daily_all_{source}_{days}d.csv"

    csv_bytes = buf.getvalue().encode("utf-8")
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'}
    )

@app.get("/public/symbols")
def get_symbols(days: int = Query(90, ge=1, le=365)):
    with engine.begin() as conn:
        rows = conn.execute(SYMBOLS_SQL, {"days": days}).mappings().all()
    return JSONResponse([r["symbol"] for r in rows])

@app.get("/public/positions/open")
def get_open_positions(symbol: Optional[str] = Query(None, description="例: btc_jpy")):
    try:
        with engine.begin() as conn:
            rows = conn.execute(OPEN_POSITIONS_SQL, {"symbol": symbol}).mappings().all()
    except Exception as e:
        logger.exception("GET /public/positions/open failed")
        raise HTTPException(status_code=500, detail=f"positions query failed: {e}")

    result = []
    for r in rows:
        t = r.get("entry_time")
        result.append({
            "position_id": r.get("position_id"),
            "symbol": r.get("symbol"),
            "side": r.get("side"),
            "size": float(r["size"]) if r.get("size") is not None else None,
            "entry_price": float(r["entry_price"]) if r.get("entry_price") is not None else None,
            "entry_time": t.isoformat() if hasattr(t, "isoformat") else t,
        })
    return result

@app.get("/public/positions/open_with_price")
def get_open_positions_with_price(symbol: str | None = Query(None)):
    with engine.begin() as conn:
        rows = conn.execute(OPEN_WITH_PRICE_SQL, {"symbol": symbol}).mappings().all()
    return [
        {
            "position_id": r["position_id"],
            "symbol": r["symbol"],
            "side": r["side"],
            "size": float(r["size"]) if r["size"] is not None else None,
            "entry_price": float(r["entry_price"]) if r["entry_price"] is not None else None,
            "entry_time": r["entry_time"].isoformat() if r["entry_time"] else None,
            "current_price": float(r["current_price"]) if r["current_price"] is not None else None,
            "unrealized_pnl_pct": float(r["unrealized_pnl_pct"]) if r["unrealized_pnl_pct"] is not None else None,
            "price_ts": r["price_ts"].isoformat() if r["price_ts"] else None,
        }
        for r in rows
    ]

# 統合版ヘルスチェック（DB & price_cache 鮮度）
@app.get("/healthz")
def healthz():
    try:
        with engine.begin() as conn:
            pong = conn.execute(sa.text("SELECT 1")).scalar_one()
            fresh_cnt = conn.execute(sa.text("""
                SELECT COUNT(*) FROM price_cache WHERE ts >= NOW() - INTERVAL '10 minutes'
            """)).scalar_one()
            return {
                "ok": True,
                "db": pong == 1,
                "price_cache_fresh_count": int(fresh_cnt),
                "ttl_minutes": 10
            }
    except Exception as e:
        logger.exception("healthz failed")
        raise HTTPException(status_code=500, detail=f"healthz error: {e}")

@app.get("/public/kpi/7days")
def get_kpi_7days(symbol: str | None = Query(None)):
    try:
        with engine.begin() as conn:
            row = conn.execute(OPEN_KPI_SQL, {"symbol": symbol}).mappings().first()
    except Exception as e:
        logger.exception("GET /public/kpi/7days failed")
        raise HTTPException(status_code=500, detail=f"kpi query failed: {e}")

    if not row:
        return {}

    def _f(v): return float(v) if v is not None else None

    return {
        "trade_count": row["trade_count"],
        "win_count": row["win_count"],
        "win_rate": (row["win_count"] / row["trade_count"] * 100.0) if row["trade_count"] else None,
        "total_pnl": _f(row["total_pnl"]),
        "avg_pnl_pct": _f(row["avg_pnl_pct"]),
        "avg_holding_hours": _f(row["avg_holding_hours"]),
    }

@app.get("/public/trades/today")
def get_trades_today(
    symbol: str | None = Query(None, description="例: btc_jpy"),
    source: str = Query("real", pattern="^(real|virtual)$")
):
    today_jst = datetime.now(JST).date()

    with engine.begin() as conn:
        rows = conn.execute(TRADES_TODAY_SQL, {
            "source": source,
            "symbol": symbol,
            "target_date": today_jst,
        }).mappings().all()

    def _f(v): return float(v) if v is not None else None
    out = []
    for r in rows:
        closed_at = r.get("closed_at")
        # opened_at は holding_hours から逆算（存在時のみ）
        opened_iso = None
        if closed_at is not None and r.get("holding_hours") is not None:
            try:
                opened_dt = closed_at - timedelta(hours=float(r["holding_hours"]))
                opened_iso = opened_dt.isoformat()
            except Exception:
                opened_iso = None
        out.append({
            "trade_id": r.get("trade_id"),
            "symbol": r.get("symbol"),
            "side": r.get("side"),
            "entry_position_id": r.get("entry_position_id"),
            "exit_order_id": r.get("exit_order_id"),
            "size": _f(r.get("size")),
            "entry_price": _f(r.get("entry_price")),
            "exit_price": _f(r.get("exit_price")),
            "pnl": _f(r.get("pnl")),
            "pnl_pct": _f(r.get("pnl_pct")),
            "opened_at": opened_iso,
            "closed_at": closed_at.isoformat() if closed_at else None,
            "holding_hours": _f(r.get("holding_hours")),
            "strategy_id": r.get("strategy_id"),
            "source": r.get("source"),
            "user_id": r.get("user_id"),
        })
    return out

@app.get("/admin/export/signal-rules.csv")
def export_signal_rules_csv(
    symbol: str | None = Query(None),
    timeframe: str | None = Query(None),
    active: str | None = Query(None, pattern="^(true|false)$"),
    user_id: str | None = Query(None),
    strategy_id: str | None = Query(None),
    version: str | None = Query(None),
    only_open_ended: bool = Query(False),
    q: str | None = Query(None),
    sort: str | None = Query("priority"),
    limit: int = Query(5000, ge=1, le=20000),
):
    base = RULES_LIST_SQL_BASE
    params = {
        "symbol": symbol,
        "timeframe": timeframe,
        "active_txt": active,
        "user_id": user_id,
        "strategy_id": strategy_id,
        "version": version,
    }
    if q:
        base += _rules_q_clause()
        params["q"] = f"%{q.lower()}%"
    if only_open_ended:
        base += _rules_only_open_ended_clause()

    order = _rules_order_clause(sort)
    sql = sa.text(base + order + " LIMIT :limit ")
    with engine.begin() as conn:
        rows = conn.execute(sql, {**params, "limit": limit}).mappings().all()

    buf = io.StringIO()
    w = csv.writer(buf)
    cols = ["id","symbol","timeframe","score_col","op","v1","v2","target_side","action","priority","active","version","valid_from","valid_to","user_id","strategy_id","notes"]
    w.writerow(cols)
    for r in rows:
        w.writerow([
            r["id"], r["symbol"], r["timeframe"], r["score_col"], r["op"],
            r["v1"], r["v2"], r["target_side"], r["action"], r["priority"], r["active"], r["version"],
            r["valid_from"].isoformat() if r["valid_from"] else "",
            r["valid_to"].isoformat() if r["valid_to"] else "",
            r.get("user_id") or "", r.get("strategy_id") or "", r.get("notes") or "",
        ])
    data = buf.getvalue().encode("utf-8")
    return StreamingResponse(iter([data]), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="signal_rule_thresholds.csv"'}
    )

@app.get("/public/support_resistance_zones")
def get_support_resistance_zones(
    symbol: str = Query(..., min_length=1, max_length=50, description="例: ltc_jpy"),
    timeframe: str = Query("1hour", description="SRを作成した足種 (例: 1hour / 15min)"),
    lookback_days: int = Query(30, ge=1, le=180),
):
    """
    support_resistance_zones テーブルから、指定シンボルのSRゾーンを返す。
    価格チャート上に水平ゾーンとして描画する用途。
    """
    try:
        with engine.begin() as conn:
            rows = conn.execute(SR_ZONES_SQL, {
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback_days": lookback_days,
            }).mappings().all()
    except Exception as e:
        logger.exception("GET /public/support_resistance_zones failed")
        raise HTTPException(status_code=500, detail=f"sr_zones query failed: {e}")

    out = []
    for r in rows:
        out.append({
            "zone_id": str(r["zone_id"]),
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "lookback_days": int(r["lookback_days"]),
            "zone_type": r["zone_type"],
            "price_center": float(r["price_center"]),
            "price_low": float(r["price_low"]),
            "price_high": float(r["price_high"]),
            "touches": int(r["touches"]),
            "strength": float(r["strength"]),
            "last_touched_at": r["last_touched_at"].isoformat() if r["last_touched_at"] else None,
            "generated_at": r["generated_at"].isoformat() if r["generated_at"] else None,
        })
    return out

def _row_to_json(r):
    return {
        "id": r["id"],
        "symbol": r["symbol"],
        "timeframe": r["timeframe"],
        "score_col": r["score_col"],
        "op": r["op"],
        "v1": float(r["v1"]) if r["v1"] is not None else None,
        "v2": float(r["v2"]) if r["v2"] is not None else None,
        "target_side": r["target_side"],
        "action": r["action"],
        "priority": r["priority"],
        "active": bool(r["active"]),
        "version": r["version"],
        "valid_from": r["valid_from"].isoformat() if r["valid_from"] else None,
        "valid_to": r["valid_to"].isoformat() if r["valid_to"] else None,
        "user_id": r.get("user_id"),
        "strategy_id": r.get("strategy_id"),
        "notes": r.get("notes"),
    }

@app.post("/admin/signal-rules")
def create_signal_rule(payload: RuleCreateIn):
    # ご指定の固定値を強制適用
    defaults = {
        "action": "disable",
        "priority": 1,
        "version": "v1",
        "user_id": "1",
        "strategy_id": "ST0001",
        "active": True,
        "notes": payload.notes or "",
    }
    params = {
        "symbol": payload.symbol,
        "timeframe": payload.timeframe,
        "score_col": payload.score_col,
        "op": payload.op,
        "v1": payload.v1,
        "v2": payload.v2,
        "target_side": payload.target_side,
        **defaults,
    }
    with engine.begin() as conn:
        row = conn.execute(INSERT_RULE_SQL, params).mappings().first()
    return _row_to_json(row)

@app.get("/public/candles_15min")
def get_candles_15min(
    symbol: str = Query(..., min_length=1, max_length=50, description="例: ltc_jpy"),
    days: int = Query(7, ge=1, le=30, description="取得する日数（最大30日）"),
):
    """
    candles_15min テーブルから、指定シンボルのローソク足（15分足）を返す。
    Frontendの PositionsTabsClient でチャート描画に利用。
    """
    try:
        with engine.begin() as conn:
            rows = conn.execute(CANDLES_15MIN_SQL, {
                "symbol": symbol,
                "days": days,
            }).all()
    except Exception as e:
        logger.exception("GET /public/candles_15min failed")
        raise HTTPException(status_code=500, detail=f"candles_15min query failed: {e}")

    out = []
    for ts, open_, high_, low_, close_, volume_ in rows:
        out.append({
            "time": ts.isoformat(),
            "open": float(open_),
            "high": float(high_),
            "low": float(low_),
            "close": float(close_),
            "volume": float(volume_) if volume_ is not None else None,
        })
    return out

JST = timezone(timedelta(hours=9))

@app.put("/admin/signal-rules/{id}")
def update_signal_rule(id: int, payload: RuleUpdateIn):
    data = payload.dict(exclude_unset=True)

    # ====== valid_from / valid_to を常に +9時間した上で JST(+09:00) 付きに変換 ======
    for key in ["valid_from", "valid_to"]:
        val = data.get(key)
        if isinstance(val, datetime):
            # ① タイムゾーンなし（naive） → UTC想定で +9h → JST 付与
            if val.tzinfo is None:
                jst_time = val + timedelta(hours=9)
                data[key] = jst_time.replace(tzinfo=JST)
            else:
                # ② タイムゾーン付き → UTC換算後 +9h → JST 付与
                jst_time = val.astimezone(timezone.utc) + timedelta(hours=9)
                data[key] = jst_time.replace(tzinfo=JST)

    # ====== 更新SET句構築 ======
    sets = []
    params = {"id": id}

    # valid_to は重複防止のため一旦除外
    for key, val in data.items():
        if key == "valid_to":
            continue
        sets.append(f"{key} = :{key}")
        params[key] = val

    # valid_toの扱い
    if data.get("active") is True and ("valid_to" in data) and (data["valid_to"] is not None):
        sets.append("valid_to = :valid_to")
        params["valid_to"] = None
    elif "valid_to" in data:
        sets.append("valid_to = :valid_to")
        params["valid_to"] = data["valid_to"]

    if not sets:
        raise HTTPException(status_code=400, detail="no fields to update")

    sql = sa.text(UPDATE_RULE_SQL_BASE.format(sets=", ".join(sets)))
    with engine.begin() as conn:
        row = conn.execute(sql, params).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="rule not found")
    return _row_to_json(row)

@app.delete("/admin/signal-rules/{id}")
def logical_delete_signal_rule(id: int):
    with engine.begin() as conn:
        row = conn.execute(LOGICAL_DELETE_SQL, {"id": id}).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="rule not found")
    return _row_to_json(row)
