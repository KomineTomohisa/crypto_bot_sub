#!/usr/bin/env python3
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import argparse
import math
import sys
from sqlalchemy import text
from psycopg2.extras import Json

JST = timezone(timedelta(hours=9))

def safe_float(x):
    try:
        if x is None:
            return None
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None

def compute_metrics(trades: List[dict], *, win_threshold: float) -> Dict[str, Any]:
    total = 0
    wins = 0
    pnl_pct_vals = []
    per_symbol: Dict[str, Dict[str, Any]] = {}

    for t in trades:
        sym = t.get("symbol")
        if not sym:
            continue
        pnl = safe_float(t.get("pnl"))
        pnl_pct = safe_float(t.get("pnl_pct"))

        total += 1
        if pnl is not None and pnl > win_threshold:
            wins += 1
        if pnl_pct is not None:
            pnl_pct_vals.append(pnl_pct)

        d = per_symbol.setdefault(sym, {"trades": 0, "wins": 0, "pnl_pct_vals": []})
        d["trades"] += 1
        if pnl is not None and pnl > win_threshold:
            d["wins"] += 1
        if pnl_pct is not None:
            d["pnl_pct_vals"].append(pnl_pct)

    win_rate = (wins / total) if total > 0 else None
    avg_pnl_pct = (sum(pnl_pct_vals) / len(pnl_pct_vals)) if pnl_pct_vals else None

    symbols_summary: Dict[str, Dict[str, Any]] = {}
    for sym, d in per_symbol.items():
        t = d["trades"]
        w = d["wins"]
        vr = (w / t) if t > 0 else None
        ap = (sum(d["pnl_pct_vals"]) / len(d["pnl_pct_vals"])) if d["pnl_pct_vals"] else None
        symbols_summary[sym] = {
            "trades": t,
            "win_rate": None if vr is None else round(vr, 4),
            "avg_pnl_pct": None if ap is None else round(ap, 4),
        }

    return {
        "total_trades": total,
        "win_rate": None if win_rate is None else round(win_rate, 4),
        "avg_pnl_pct": None if avg_pnl_pct is None else round(avg_pnl_pct, 4),
        "symbols": symbols_summary,
    }

def align_dt(dt: datetime, mode: str) -> datetime:
    if mode == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    if mode == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt

def main():
    parser = argparse.ArgumentParser(description="Aggregate last N days of trades into public_metrics (upsert).")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days (default: 30)")
    parser.add_argument("--code-dir", type=str, default=".", help="Directory containing db.py (default: current dir)")
    parser.add_argument("--align", choices=["none", "hour", "day"], default="none",
                        help="Align period_end to the top of hour/day (default: none)")
    parser.add_argument("--min_abs_pnl", type=float, default=None,
                        help="Ignore trades whose |pnl| < this value (noise filter)")
    parser.add_argument("--win_threshold", type=float, default=0.0,
                        help="pnl > threshold is counted as win (default: 0.0)")
    args = parser.parse_args()

    sys.path.insert(0, args.code_dir)
    try:
        from db import get_trades_between, engine
        from sqlalchemy import text
    except Exception as e:
        print(f"[ERROR] failed to import db helpers: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        now_jst = align_dt(datetime.now(JST), args.align)
        start_jst = now_jst - timedelta(days=args.days)

        # pull trades
        trades = get_trades_between(start_jst, now_jst)

        # optional noise filter
        if args.min_abs_pnl is not None:
            def keep(t):
                v = safe_float(t.get("pnl"))
                return (v is not None) and (abs(v) >= args.min_abs_pnl)
            trades = [t for t in trades if keep(t)]

        # compute
        m = compute_metrics(trades, win_threshold=args.win_threshold)

        # define period (UTC)
        start_utc = start_jst.astimezone(timezone.utc)
        end_utc = now_jst.astimezone(timezone.utc)

        # upsert
        sql = text("""
            INSERT INTO public_metrics (
                period_start, period_end, total_trades, win_rate, avg_pnl_pct, symbols
            ) VALUES (
                :period_start, :period_end, :total_trades, :win_rate, :avg_pnl_pct, :symbols
            )
            ON CONFLICT (period_start, period_end) DO UPDATE
            SET total_trades = EXCLUDED.total_trades,
                win_rate     = EXCLUDED.win_rate,
                avg_pnl_pct  = EXCLUDED.avg_pnl_pct,
                symbols      = EXCLUDED.symbols
        """)

        params = {
            "period_start": start_utc,
            "period_end": end_utc,
            "total_trades": int(m["total_trades"]),
            "win_rate": float(m["win_rate"]) if m["win_rate"] is not None else None,
            "avg_pnl_pct": float(m["avg_pnl_pct"]) if m["avg_pnl_pct"] is not None else None,
            # ここを Json(...) に
            "symbols": Json(m["symbols"]),
        }

        with engine.begin() as conn:
            conn.execute(sql, params)

        # logs
        print(f"[OK] upserted public_metrics {start_utc.isoformat()} → {end_utc.isoformat()}")
        print(json.dumps(m, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"[ERROR] aggregation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
