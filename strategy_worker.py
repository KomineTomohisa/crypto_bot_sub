from __future__ import annotations
import os, json, time, smtplib, ssl
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta
from uuid import uuid5, NAMESPACE_URL

from decimal import Decimal
import pandas as pd
from sqlalchemy import text

# 15m は既存のローダーを使用。5m/1h は未実装ならスキップ。
from app.market_loader import load_ohlcv_15m
from app.indicators import add_indicators
from app.virtual_engine import EntryRule, ExitRule, _row_entry_ok
from db import engine, insert_signal, upsert_position, insert_trade

POLL_SEC = int(os.getenv("WORKER_POLL_SEC", "30"))
EXIT_USE_HL = os.getenv("EXIT_USE_HL", "1") == "1"  # True: TP/SL を High/Low で評価

# ===== 共通ユーティリティ =====
def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        # Decimal / numpy なども float に統一
        return float(x)
    except Exception:
        return None

def _to_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(x))
    except Exception:
        return None

def _parse_symbols(raw: Any) -> List[str]:
    """user_strategies.symbols が JSONB でも text[] でも吸収"""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    s = str(raw)
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
        return [x.strip().strip('"') for x in inner.split(",") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",")]
    return [s]

# ===== メール送信（必要に応じて環境変数でON） =====
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER or "noreply@example.com")
MAIL_TO_DEFAULT = os.getenv("NOTIFY_EMAIL_DEFAULT")

def _send_email(subject: str, body: str, to_addr: Optional[str] = None) -> None:
    to_addr = to_addr or MAIL_TO_DEFAULT
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and to_addr):
        print(f"[email-skip] {subject}")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = MAIL_FROM
    msg["To"] = to_addr
    ctx = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls(context=ctx)
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(MAIL_FROM, [to_addr], msg.as_string())

# ===== DBユーティリティ =====
def _fetch_enabled_strategies() -> List[Dict[str, Any]]:
    sql = text("""
        SELECT user_id, strategy_id, name, symbols, timeframe,
               atr_min, atr_pct_min, adx_min, rsi_long_min,
               macd_cross, ema_cross_req, cross_mode,
               take_profit_pct, stop_loss_pct, max_holding_min,
               use_dynamic_levels, trade_size_jpy, is_enabled
          FROM user_strategies
         WHERE is_enabled = true
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        d["symbols"] = _parse_symbols(d.get("symbols"))
        out.append(d)
    return out

def _get_position(position_id: str) -> Optional[Dict[str, Any]]:
    sql = text("""
        SELECT position_id, symbol, side, size, avg_entry_price, opened_at, updated_at
          FROM positions
         WHERE position_id = :pid
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {"pid": position_id}).mappings().first()
    return dict(row) if row else None

def _stable_signal_id(user_id: int, strategy_id: str, symbol: str, ts: datetime, kind: str) -> str:
    """kind: 'entry' | 'exit'。UUID v5で安定IDを生成"""
    t = ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    name = f"virt:{user_id}:{strategy_id}:{symbol}:{t}:{kind}"
    return str(uuid5(NAMESPACE_URL, name))

def _jst(dt: datetime) -> datetime:
    return dt.astimezone(timezone(timedelta(hours=9)))

# ===== OHLCV + インジ計算 =====
def _load_last_row(symbol: str, tf: str, lookback_min: int = 180) -> Optional[pd.Series]:
    tf = (tf or "15m").lower()
    if tf == "15m":
        df = load_ohlcv_15m(symbol, lookback_min=lookback_min)
    elif tf in ("5m", "1h", "60m"):
        print(f"[worker] skip unsupported timeframe loader: {tf} sym={symbol}")
        return None
    else:
        print(f"[worker] skip unknown timeframe: {tf} sym={symbol}")
        return None

    if df is None or df.empty:
        return None
    df = add_indicators(df.tail(40))
    if df is None or df.empty:
        return None
    df2 = df.dropna()
    if df2.empty:
        return None
    return df2.tail(1).iloc[0]

# ===== ルール生成（Decimal -> float/int に正規化） =====
def _entry_rule_from_strategy(s: Dict[str, Any]) -> EntryRule:
    return EntryRule(
        atr_min=_to_float(s.get("atr_min")),
        atr_pct_min=_to_float(s.get("atr_pct_min")),
        adx_min=_to_float(s.get("adx_min")),
        rsi_long_min=_to_float(s.get("rsi_long_min")),
        macd_cross=bool(s.get("macd_cross")) if s.get("macd_cross") is not None else None,
        ema_cross_req=bool(s.get("ema_cross_req")) if s.get("ema_cross_req") is not None else None,
        cross_mode=(s.get("cross_mode") or "any"),
    )

def _exit_rule_from_strategy(s: Dict[str, Any]) -> ExitRule:
    return ExitRule(
        take_profit_pct=_to_float(s.get("take_profit_pct")),
        stop_loss_pct=_to_float(s.get("stop_loss_pct")),
        max_holding_min=_to_int(s.get("max_holding_min")),
        use_dynamic_levels=bool(s.get("use_dynamic_levels")),
    )

# ===== ENTRY / EXIT 判定 & 通知 =====
def _maybe_notify_entry(s: Dict[str, Any], sym: str, tf: str, row: pd.Series) -> None:
    user_id = s["user_id"]; sid = s["strategy_id"]
    ts: datetime = row["timestamp"]
    price = float(row["close"])
    pos_id = f"virt:{user_id}:{sid}:{sym}"

    # 既に仮想ポジションがあればスキップ
    pos = _get_position(pos_id)
    if pos and pos["size"] and float(pos["size"]) > 0.0:
        return

    # エントリー条件判定
    er = _entry_rule_from_strategy(s)
    entry_ok = _row_entry_ok(row, er)

    # 診断ログ
    print(f"[worker] check user={user_id} strat={str(sid)[:8]} sym={sym} tf={tf} "
          f"close={price} rsi={row.get('RSI')} adx={row.get('ADX')} atr={row.get('ATR')} "
          f"ema20={row.get('EMA20')} ema50={row.get('EMA50')} macd_cross={row.get('MACD_cross_up')} "
          f"ema_cross={row.get('EMA_cross_up')} -> entry_ok={entry_ok}")

    if not entry_ok:
        return

    # ポジションを開く（ロングのみ）
    size_jpy = _to_float(s.get("trade_size_jpy")) or 10000.0
    size = size_jpy / price
    upsert_position(
        position_id=pos_id,
        symbol=sym,
        side="long",
        size=float(size),
        avg_entry_price=price,
        opened_at=ts,
        updated_at=ts,
        raw={"kind": "virtual"},
        strategy_id=sid,
        user_id=user_id,
        source="virtual",
    )

    # シグナル保存（重複防止のため安定IDを使う）
    sig_id = _stable_signal_id(user_id, sid, sym, ts, "entry")
    insert_signal(
        signal_id=sig_id, user_id=user_id, symbol=sym, timeframe=tf, side="long",
        rsi=_to_float(row.get("RSI")),
        adx=_to_float(row.get("ADX")),
        atr=_to_float(row.get("ATR")),
        ema_fast=_to_float(row.get("EMA20")),
        ema_slow=_to_float(row.get("EMA50")),
        price=price, generated_at=ts, strategy_id=sid, version="v1", status="entry",
        raw={"rule": er.__dict__},
        source="virtual",
    )

    # 通知（JST表示）
    ts_jst = _jst(ts).strftime("%Y-%m-%d %H:%M")
    cross_txt = ("MACD or EMA" if (s.get("macd_cross") and s.get("ema_cross_req"))
                 else ("MACD" if s.get("macd_cross") else ("EMA" if s.get("ema_cross_req") else "-")))
    body = f"""[ENTRY] {sym} {tf}
Time(JST): {ts_jst}
Price    : {price:,.0f} JPY
Strategy : {s.get('name')}
Filters  : ADX>={s.get('adx_min')} RSI>={s.get('rsi_long_min')} ATR%>={s.get('atr_pct_min')}
Trigger  : {cross_txt}
"""
    _send_email(subject=f"[ENTRY] {sym} {tf} #{sid}", body=body)

def _tf_minutes(tf_str: str) -> int:
    tf_str = (tf_str or "15m").lower()
    if tf_str.endswith("m"):
        return int(tf_str[:-1])
    if tf_str.endswith("h"):
        return int(tf_str[:-1]) * 60
    return 15

def _maybe_notify_exit(s: Dict[str, Any], sym: str, tf: str, row: pd.Series) -> None:
    """open中の仮想ポジションについて TP/SL/max_holding を判定"""
    user_id = s["user_id"]; sid = s["strategy_id"]
    ts: datetime = row["timestamp"]
    price = float(row["close"])
    pos_id = f"virt:{user_id}:{sid}:{sym}"
    pos = _get_position(pos_id)
    if not pos or not pos["size"] or float(pos["size"]) <= 0.0:
        return

    entry_px = _to_float(pos["avg_entry_price"]) or price
    size = _to_float(pos["size"]) or 0.0
    opened_at = pos["opened_at"]
    if isinstance(opened_at, str):
        opened_at = datetime.fromisoformat(opened_at)
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)

    ex = _exit_rule_from_strategy(s)

    # --- TP/SL 評価（High/Low or Close） ---
    hi = _to_float(row.get("high")) or price
    lo = _to_float(row.get("low")) or price
    # %は float 化済み（_exit_rule_from_strategy）
    px_tp = entry_px * (1.0 + (ex.take_profit_pct or 0.0))
    px_sl = entry_px * (1.0 + (ex.stop_loss_pct or 0.0))
    ref_up = hi if EXIT_USE_HL else price
    ref_dn = lo if EXIT_USE_HL else price
    tp_hit = (ex.take_profit_pct is not None) and (ref_up >= px_tp)
    sl_hit = (ex.stop_loss_pct  is not None) and (ref_dn <= px_sl)

    # --- max_holding（分 & バー数） ---
    mh_hit = False
    if ex.max_holding_min is not None:
        age_ok = (ts - opened_at) >= timedelta(minutes=ex.max_holding_min)
        bar_min = _tf_minutes(tf)
        bars_needed = max(1, (ex.max_holding_min + bar_min - 1) // bar_min)
        bars_elapsed = max(0, int((ts - opened_at).total_seconds() // (bar_min * 60)))
        bars_ok = bars_elapsed >= bars_needed
        mh_hit = age_ok or bars_ok

    # 目視ログ
    print(f"[worker-exit-check] sym={sym} tf={tf} entry={entry_px:.4f} "
          f"close={price:.4f} hi={hi:.4f} lo={lo:.4f} "
          f"px_tp={px_tp:.4f} px_sl={px_sl:.4f} "
          f"tp_hit={tp_hit} sl_hit={sl_hit} mh_hit={mh_hit} "
          f"opened_at={opened_at} now={ts}")

    if not (tp_hit or sl_hit or mh_hit):
        return

    # 取引記録（トレード確定）
    pnl = (price - entry_px) * size
    pnl_pct = (price / entry_px - 1.0) * 100.0
    holding_h = (ts - opened_at).total_seconds() / 3600.0
    insert_trade(
        trade_id=None, symbol=sym, side="long",
        entry_position_id=pos_id, exit_order_id=None,
        entry_price=entry_px, exit_price=price, size=size,
        pnl=pnl, pnl_pct=pnl_pct, holding_hours=holding_h,
        closed_at=ts,
        raw={"reason": "virtual-exit", "tp": tp_hit, "sl": sl_hit, "mh": mh_hit},
        strategy_id=sid,
        user_id=user_id,
        source="virtual",
    )

    # ポジションをクローズ（size=0）
    upsert_position(
        position_id=pos_id, symbol=sym, side="long", size=0.0,
        avg_entry_price=entry_px, opened_at=opened_at, updated_at=ts, raw={"closed": True}
    )

    # シグナル保存（EXIT）
    sig_id = _stable_signal_id(user_id, sid, sym, ts, "exit")
    insert_signal(
        signal_id=sig_id,
        user_id=user_id,
        symbol=sym,
        timeframe=tf,
        side="flat",
        price=price,
        generated_at=ts,
        strategy_id=sid,
        version="v1",
        status="exit",
        raw={"reason": "virtual-exit", "tp": tp_hit, "sl": sl_hit, "mh": mh_hit},
        source="virtual",
    )

    # 通知
    ts_jst = _jst(ts).strftime("%Y-%m-%d %H:%M")
    reason = "TP" if tp_hit else ("SL" if sl_hit else "MAX_HOLD")
    body = f"""[EXIT-{reason}] {sym} {tf}
Time(JST): {ts_jst}
Price    : {price:,.0f} JPY
PnL      : {pnl:,.2f} JPY  ({pnl_pct:.2f}%)
Strategy : {s.get('name')}
"""
    _send_email(subject=f"[EXIT-{reason}] {sym} {tf} #{sid}", body=body)

# ===== メインループ =====
def main_loop():
    print("[worker] start")
    while True:
        try:
            strategies = _fetch_enabled_strategies()
            print(f"[worker] tick strategies={len(strategies)}")
            if not strategies:
                time.sleep(POLL_SEC); continue

            # 対象 (tf, sym) をユニーク化して先に読み込む
            pairs = sorted(set(((s.get("timeframe") or "15m").lower(), sym)
                               for s in strategies for sym in (s.get("symbols") or [])))

            last_rows: Dict[tuple, pd.Series] = {}
            for tf, sym in pairs:
                row = _load_last_row(sym, tf, lookback_min=180)
                if row is not None and "close" in row:
                    last_rows[(tf, sym)] = row

            # 各戦略×銘柄で Exit→Entry の順に判定
            for s in strategies:
                tf = (s.get("timeframe") or "15m").lower()
                for sym in s.get("symbols") or []:
                    row = last_rows.get((tf, sym))
                    if row is None:
                        continue
                    _maybe_notify_exit(s, sym, tf, row)
                    _maybe_notify_entry(s, sym, tf, row)
        except Exception as e:
            print("[worker-error]", e)
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main_loop()