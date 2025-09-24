from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import math

JST = timezone(timedelta(hours=9), name="JST")

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _fmt_amount(yen: float) -> str:
    return f"{yen:,.0f}円"

def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"

def _day_bounds_jst(day: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    # [00:00, 24:00) JST の日付範囲
    day = (day or datetime.now(JST)).astimezone(JST)
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end

def _is_exit_like(t: Dict) -> bool:
    # DB: exit系 → closed_at/pnl 存在
    if "closed_at" in t or "pnl" in t:
        return True
    # メモリの trade_log_exit 形式
    needed = ("exit_price","exit_time","profit")
    return all(k in t for k in needed)

def _extract_exit_time_jst(t: Dict) -> Optional[datetime]:
    for key in ("closed_at", "exit_time", "time"):
        v = t.get(key)
        if isinstance(v, datetime):
            return v.astimezone(JST) if v.tzinfo else v.replace(tzinfo=JST)
        if isinstance(v, str):
            s = v.strip().replace("Z", "+00:00")
            # ISO8601 with tz
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=JST)
                return dt.astimezone(JST)
            except Exception:
                pass
            # "YYYY-mm-dd HH:MM:SS.sss +0900" 形式対応
            try:
                dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f %z")
                return dt.astimezone(JST)
            except Exception:
                continue
    return None

def _pick_number(d: Dict, keys: List[str], default=0.0) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            return _safe_float(d[k], default)
    return default

def build_daily_report_message(trades: List[Dict], day: Optional[datetime] = None) -> Tuple[str, str]:
    """
    trades: 1日分の「クローズ済」トレード辞書（DB/メモリどちらの形式でもOK）
    return: (subject, message)
    """
    start, end = _day_bounds_jst(day)
    day_str = start.strftime("%Y-%m-%d")

    # --- 対象日のクローズ済みだけに絞る ---
    closed = []
    for t in trades:
        if not _is_exit_like(t):
            continue
        et = _extract_exit_time_jst(t)
        if et is None:
            continue
        if start <= et < end:
            closed.append(t)

    total_trades = len(closed)
    if total_trades == 0:
        subject = f"Daily Report {day_str}"
        message = (
            f"【日次まとめ】{day_str}\n"
            f"本日のクローズ済みトレードはありません。\n"
            f"{start.strftime('%Y-%m-%d %H:%M')} 〜 {end.strftime('%Y-%m-%d %H:%M')} JST"
        )
        return subject, message

    # --- 集計 ---
    wins, losses = 0, 0
    pnl_sum = 0.0
    hold_sum = 0.0

    symbol_stats = {}
    best = {"pnl": -1e18, "t": None}
    worst = {"pnl": 1e18, "t": None}

    for t in closed:
        # 収益（メモリ=profit / DB=pnl）
        pnl = _pick_number(t, ["profit","pnl"], 0.0)
        pnl_sum += pnl

        # 勝敗（0 は負け扱いでも良いが、ここは ± で）
        if pnl > 0:
            wins += 1
        else:
            losses += 1

        # 保有時間（hour単位）
        holding = _pick_number(t, ["holding_hours"], None)
        if not math.isnan(holding) if isinstance(holding, float) else holding is not None:
            hold_sum += float(holding)

        # シンボル別
        symbol = t.get("symbol") or t.get("pair") or "UNKNOWN"
        ss = symbol_stats.setdefault(symbol, {"trades":0,"win":0,"loss":0,"pnl":0.0})
        ss["trades"] += 1
        ss["pnl"] += pnl
        if pnl > 0: ss["win"] += 1
        else: ss["loss"] += 1

        # ベスト/ワースト
        if pnl > best["pnl"]:
            best = {"pnl": pnl, "t": t}
        if pnl < worst["pnl"]:
            worst = {"pnl": pnl, "t": t}

    win_rate = (wins / total_trades) * 100.0
    avg_hold = (hold_sum / total_trades) if total_trades > 0 and hold_sum else None

    # 平均勝ち/負け、Profit Factor
    win_pnls = [_pick_number(t, ["profit","pnl"]) for t in closed if _pick_number(t, ["profit","pnl"]) > 0]
    loss_pnls = [_pick_number(t, ["profit","pnl"]) for t in closed if _pick_number(t, ["profit","pnl"]) <= 0]
    avg_win = sum(win_pnls)/len(win_pnls) if win_pnls else 0.0
    avg_loss_abs = abs(sum(loss_pnls)/len(loss_pnls)) if loss_pnls else 0.0
    gross_profit = sum(win_pnls)
    gross_loss_abs = abs(sum(loss_pnls))
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else float('inf')

    # --- メッセージ生成（Markdown互換） ---
    lines = []
    lines.append(f"【日次まとめ】{day_str}")
    lines.append(f"期間: {start.strftime('%Y-%m-%d %H:%M')} 〜 {end.strftime('%Y-%m-%d %H:%M')} JST")
    lines.append("")
    lines.append(f"総トレード数: {total_trades}  |  勝ち: {wins} / 負け: {losses}  |  勝率: {_fmt_pct(win_rate)}")
    lines.append(f"損益合計: {_fmt_amount(pnl_sum)}  |  PF: {profit_factor:.2f}  |  平均保有時間: {avg_hold:.2f}h" if avg_hold is not None else
                 f"損益合計: {_fmt_amount(pnl_sum)}  |  PF: {profit_factor:.2f}")
    lines.append("")
    lines.append("— シンボル別 —")
    for sym, s in sorted(symbol_stats.items(), key=lambda kv: kv[1]["pnl"], reverse=True):
        wr = (s["win"]/s["trades"]*100.0) if s["trades"] else 0.0
        lines.append(f"{sym.upper()}: {s['trades']}回  勝率 {_fmt_pct(wr)}  損益 {_fmt_amount(s['pnl'])}")

    # ベスト/ワースト（あれば）
    def _brief(t: Dict) -> str:
        if not t: return "-"
        side = t.get("type") or t.get("side") or "-"
        ep = _pick_number(t, ["entry_price","price"], None)
        xp = _pick_number(t, ["exit_price"], None)
        pnl = _pick_number(t, ["profit","pnl"], 0.0)
        return f"{t.get('symbol','?').upper()} {side}  PnL {_fmt_amount(pnl)}  (in {ep:.4f} → out {xp:.4f})" if ep and xp else \
               f"{t.get('symbol','?').upper()} {side}  PnL {_fmt_amount(pnl)}"

    lines.append("")
    lines.append("— Best —")
    lines.append(_brief(best["t"]))
    lines.append("— Worst —")
    lines.append(_brief(worst["t"]))

    # 参考：今日の学び・メモ欄
    lines.append("")
    lines.append("メモ: ADXが高い時間帯はTPヒット率↑ / ATR拡大時はSLをやや広めに　(例)")
    lines.append("")

    subject = f"Daily Report {day_str}"
    message = "\n".join(lines)
    return subject, message
