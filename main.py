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

# ------------------- エンドポイント -------------------

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
