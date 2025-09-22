#!/usr/bin/env python3
import json
import math
from datetime import datetime, timedelta, timezone, date, time as dtime
from typing import Dict, Any, List, Optional
import argparse

JST = timezone(timedelta(hours=9))

def safe_float(x):
    try:
        if x is None:
            return None
        xf = float(x)
        if math.isfinite(xf):
            return xf
        return None
    except Exception:
        return None

def compute_metrics(trades: List[dict]) -> Dict[str, Any]:
    total = len(trades)
    wins = 0
    pnl_pct_vals = []
    per_symbol: Dict[str, Dict[str, Any]] = {}

    for t in trades:
        sym = t.get("symbol")
        pnl = safe_float(t.get("pnl"))
        pnl_pct = safe_float(t.get("pnl_pct"))

        if pnl is not None and pnl > 0:
            wins += 1
        if pnl_pct is not None:
            pnl_pct_vals.append(pnl_pct)

        if sym not in per_symbol:
            per_symbol[sym] = {"trades": 0, "wins": 0, "pnl_pct_vals": []}
        per_symbol[sym]["trades"] += 1
        if pnl is not None and pnl > 0:
            per_symbol[sym]["wins"] += 1
        if pnl_pct is not None:
            per_symbol[sym]["pnl_pct_vals"].append(pnl_pct)

    win_rate = (wins / total) if total > 0 else None
    avg_pnl_pct = (sum(pnl_pct_vals) / len(pnl_pct_vals)) if pnl_pct_vals else None

    symbols_summary: Dict[str, Dict[str, Any]] = {}
    for sym, d in per_symbol.items():
        t = d["trades"]
        w = d["wins"]
        vr = (w / t) if t > 0 else None
        ap = (sum(d["pnl_pct_vals"]) / len(d["pnl_pct_vals"])) if d["pnl_pct_vals"] else None
        symbols_summary[sym] = {"trades": t, "win_rate": vr, "avg_pnl_pct": ap}

    return {
        "total_trades": total,
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl_pct,
        "symbols": symbols_summary,
    }

def main():
    parser = argparse.ArgumentParser(description="Upsert daily public metrics for last N days (JST days).")
    parser.add_argument("--days", type=int, default=30, help="How many days back including today (default: 30)")
    parser.add_argument("--code-dir", type=str, default="/root/crypto_bot", help="Directory containing db.py")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, args.code_dir)

    from db import get_trades_between, engine  # DATABASE_URL 必須
    from sqlalchemy import text

    sql = text("""
        INSERT INTO public_metrics_daily (
            metric_date, total_trades, win_rate, avg_pnl_pct, symbols
        ) VALUES (
            :metric_date, :total_trades, :win_rate, :avg_pnl_pct, :symbols
        )
        ON CONFLICT (metric_date) DO UPDATE
        SET total_trades = EXCLUDED.total_trades,
            win_rate     = EXCLUDED.win_rate,
            avg_pnl_pct  = EXCLUDED.avg_pnl_pct,
            symbols      = EXCLUDED.symbols,
            created_at   = now()
    """)

    today_jst = datetime.now(JST).date()
    start_date = today_jst - timedelta(days=args.days - 1)

    upserts = []
    with engine.begin() as conn:
        for i in range(args.days):
            day = start_date + timedelta(days=i)
            day_start_jst = datetime.combine(day, dtime(0,0,0, tzinfo=JST))
            day_end_jst   = day_start_jst + timedelta(days=1)

            trades = get_trades_between(day_start_jst, day_end_jst)
            m = compute_metrics(trades)

            params = {
                "metric_date": day,  # JST基準の日付をそのまま保存
                "total_trades": int(m["total_trades"]),
                "win_rate": float(m["win_rate"]) if m["win_rate"] is not None else None,
                "avg_pnl_pct": float(m["avg_pnl_pct"]) if m["avg_pnl_pct"] is not None else None,
                "symbols": json.dumps(m["symbols"], ensure_ascii=False),  # ← キャストなしでOK
            }
            conn.execute(sql, params)
            upserts.append({"metric_date": str(day), **m})

    print(json.dumps({"upserted_days": len(upserts), "items": upserts}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
