from typing import Optional, Dict, List, Any, Iterable
from datetime import date, timedelta
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import csv
import io

# 既存の日次集計取得関数をそのまま利用（戻り値例:
# [{"metric_date": date, "symbols": {"eth_jpy": {"total_trades": 0, "win_rate": None, "avg_pnl_pct": None}, ...}}, ...]
from db import get_public_metrics_daily

router = APIRouter(prefix="/public", tags=["public"])

def _date_range_days(days: int) -> tuple[date, date]:
    end = date.today()              # JST運用なら必要に応じてJST化
    start = end - timedelta(days=days)
    return start, end

@router.get("/performance/daily/by-symbol")
def get_daily_by_symbol(
    symbol: Optional[str] = Query(None, description="例: eth_jpy。省略時は全シンボルを返す"),
    days: int = Query(30, ge=1, le=180),
):
    """
    - symbol 指定あり: 互換維持のため **配列** を返す（既存と同じフォーマット）
      [ {date, total_trades, win_rate, avg_pnl_pct}, ... ]
    - symbol 省略時: 全シンボルをまとめて返す
      { "days": <int>, "items_by_symbol": { "<sym>": [ {date, total_trades, ...}, ... ], ... } }
    """
    start, end = _date_range_days(days)
    rows: List[Dict[str, Any]] = get_public_metrics_daily(start, end)

    def row_item(day_row: Dict[str, Any], one_sym: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 既存キー名に合わせる（total_trades / win_rate / avg_pnl_pct）
        return {
            "date": (day_row["metric_date"].isoformat()
                     if hasattr(day_row["metric_date"], "isoformat")
                     else day_row["metric_date"]),
            "total_trades": payload.get("total_trades", payload.get("trades", 0)),
            "win_rate": payload.get("win_rate"),
            "avg_pnl_pct": payload.get("avg_pnl_pct"),
        }

    if symbol:
        items: List[Dict[str, Any]] = []
        for r in rows:
            payload = (r.get("symbols") or {}).get(symbol)
            if payload:
                items.append(row_item(r, symbol, payload))
        return items  # ★単一シンボル時は従来どおり「配列」を返す

    # symbol 未指定 → 全シンボル分
    items_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        syms = r.get("symbols") or {}
        for sym, payload in syms.items():
            items_by_symbol.setdefault(sym, []).append(row_item(r, sym, payload))
    return {"days": days, "items_by_symbol": items_by_symbol}

@router.get("/export/performance/daily_by_symbol.csv")
def export_daily_by_symbol_csv(
    symbol: Optional[str] = Query(None, description="例: eth_jpy。省略時は全シンボルを縦持ちで出力"),
    days: int = Query(30, ge=1, le=180),
):
    """
    CSV は以下の互換性を維持:
    - symbol 指定あり: 既存どおり 4列 → date,trades,win_rate,avg_pnl_pct
    - symbol なし   : 全シンボル縦持ち → date,symbol,trades,win_rate,avg_pnl_pct
    """
    start, end = _date_range_days(days)
    rows: List[Dict[str, Any]] = get_public_metrics_daily(start, end)

    # 正規化
    flat_rows: List[Dict[str, Any]] = []

    if symbol:
        for r in rows:
            payload = (r.get("symbols") or {}).get(symbol)
            if not payload:
                continue
            flat_rows.append({
                "date": (r["metric_date"].isoformat()
                         if hasattr(r["metric_date"], "isoformat")
                         else r["metric_date"]),
                # CSV互換: trades カラム名で出す（既存が date,trades,win_rate,avg_pnl_pct）
                "trades": payload.get("total_trades", payload.get("trades", 0)),
                "win_rate": payload.get("win_rate"),
                "avg_pnl_pct": payload.get("avg_pnl_pct"),
            })
        fieldnames = ["date", "trades", "win_rate", "avg_pnl_pct"]
    else:
        for r in rows:
            syms = r.get("symbols") or {}
            for sym, payload in syms.items():
                flat_rows.append({
                    "date": (r["metric_date"].isoformat()
                             if hasattr(r["metric_date"], "isoformat")
                             else r["metric_date"]),
                    "symbol": sym,
                    "trades": payload.get("total_trades", payload.get("trades", 0)),
                    "win_rate": payload.get("win_rate"),
                    "avg_pnl_pct": payload.get("avg_pnl_pct"),
                })
        fieldnames = ["date", "symbol", "trades", "win_rate", "avg_pnl_pct"]

    def _iter_csv() -> Iterable[bytes]:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            # 欠損は空欄で
            for k in fieldnames:
                row.setdefault(k, "")
            writer.writerow(row)
        yield buf.getvalue().encode("utf-8")

    filename = ("daily_by_symbol.csv"
                if not symbol else f"daily_by_symbol_{symbol}.csv")
    return StreamingResponse(
        _iter_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
