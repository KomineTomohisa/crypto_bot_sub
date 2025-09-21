# app/main.py
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from decimal import Decimal
import sqlalchemy as sa
from db import engine

app = FastAPI(
    title="Signal Service API",
    version="0.1.0",
    description="Public endpoints for metrics and signals."
)

# --- CORS（必要に応じて調整） ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # 公開するなら "*" でもOK。必要に応じてドメインを限定
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# --- 型: JSONに安全な形へ整形 ---
def _to_plain(x):
    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Decimal):
        # 小数はfloatへ（丸めが必要なら round(x, 4) などに変更）
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

LATEST_SIGNALS_SQL = sa.text("""
  SELECT symbol, side, price, generated_at, strength_score
  FROM signals
  ORDER BY generated_at DESC
  LIMIT :limit
""")

# ---- SQL（COUNT と ページング本体）----
COUNT_SIGNALS_SQL = sa.text("""
    SELECT COUNT(*)::bigint AS cnt
    FROM signals
    WHERE (:symbol IS NULL OR lower(symbol) = lower(:symbol))
      AND (:since  IS NULL OR generated_at >= :since)
""")

FILTERED_SIGNALS_SQL = sa.text("""
    SELECT symbol, side, price, generated_at, strength_score
    FROM signals
    WHERE (:symbol IS NULL OR lower(symbol) = lower(:symbol))
      AND (:since  IS NULL OR generated_at >= :since)
    ORDER BY generated_at DESC
    LIMIT :limit OFFSET :offset
""")

@app.get("/public/metrics", response_model=PublicMetricsOut)
def get_public_metrics():
    try:
        with engine.begin() as conn:
            row = conn.execute(LATEST_SQL).mappings().first()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not row:
        # スナップショットがまだ無い（aggregate_public_metrics.py が未実行など）
        raise HTTPException(status_code=404, detail="no public metrics found")

    # symbols は JSONB → dict として返ってくる想定（psycopg2）
    # Decimal などを含む可能性があるので plain 化してから Pydantic へ
    payload = {
        "period_start": row["period_start"],
        "period_end": row["period_end"],
        "total_trades": row["total_trades"],
        "win_rate": row["win_rate"],
        "avg_pnl_pct": row["avg_pnl_pct"],
        "symbols": _to_plain(row["symbols"]) or {},
    }
    return payload

@app.get("/healthz")
def healthz():
    # 簡易ヘルスチェック
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unhealthy: {e}")

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

    # メタ情報はヘッダで
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Limit"] = str(limit)
    response.headers["X-Has-Next"] = "true" if (offset + len(rows)) < total else "false"

    # 互換のため配列返却（フロントはそのまま動く）
    return [{
        "symbol": r["symbol"],
        "side": r["side"],
        "price": float(r["price"]) if r["price"] is not None else None,
        "generated_at": r["generated_at"].isoformat() if r["generated_at"] else None,
        "strength_score": float(r["strength_score"]) if r["strength_score"] is not None else None,
    } for r in rows]