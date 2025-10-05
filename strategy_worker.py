from __future__ import annotations
import os, json, time, smtplib, ssl, math
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import pandas as pd
from sqlalchemy import text

from app.market_loader import load_ohlcv_15m
from app.indicators import add_indicators
from app.virtual_engine import EntryRule, ExitRule, Variant, _row_entry_ok  # _row_entry_ok を再利用
from uuid import uuid5, NAMESPACE_URL
from db import engine, insert_signal, upsert_position, insert_trade

POLL_SEC = int(os.getenv("WORKER_POLL_SEC", "30"))
TIMEFRAME = "15m"

# ====== 通知（メール） ======
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER or "noreply@example.com")
MAIL_TO_DEFAULT = os.getenv("NOTIFY_EMAIL_DEFAULT")  # ユーザーごとの宛先は後述の TODO

def _send_email(subject: str, body: str, to_addr: Optional[str] = None) -> None:
    to_addr = to_addr or MAIL_TO_DEFAULT
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and to_addr):
        # 送信設定が無ければスキップ（ログだけ）
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

# ====== DB ユーティリティ ======
def _parse_symbols(raw: Any) -> List[str]:
    """user_strategies.symbols が JSONB でも text[] でも吸収"""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    s = str(raw)
    # JSON ならそのまま読む
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    # {eth_jpy,xrp_jpy} 形式（text[]）の簡易パース
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
        return [x.strip().strip('"') for x in inner.split(",") if x.strip()]
    # カンマ区切り
    if "," in s:
        return [x.strip() for x in s.split(",")]
    return [s]

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
    """
    kind: "entry" / "exit"
    UUID v5（名前ベース）で安定IDを生成（DB列がUUID型でもOK）。
    """
    t = ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    name = f"virt:{user_id}:{strategy_id}:{symbol}:{t}:{kind}"
    return str(uuid5(NAMESPACE_URL, name))

def _jst(dt: datetime) -> datetime:
    return dt.astimezone(timezone(timedelta(hours=9)))

# ====== ロジック本体 ======
def _last_row_with_indicators(symbol: str, lookback_min: int = 180) -> Optional[pd.Series]:
    df = load_ohlcv_15m(symbol, lookback_min=lookback_min)
    if df is None or df.empty:
        return None
    df = add_indicators(df.tail(40))
    if df is None or df.empty:
        return None
    return df.dropna().tail(1).iloc[0]

def _entry_rule_from_strategy(s: Dict[str, Any]) -> EntryRule:
    return EntryRule(
        atr_min=s.get("atr_min"),
        atr_pct_min=s.get("atr_pct_min"),
        adx_min=s.get("adx_min"),
        rsi_long_min=s.get("rsi_long_min"),
        macd_cross=s.get("macd_cross"),
        ema_cross_req=s.get("ema_cross_req"),
        cross_mode=s.get("cross_mode") or "any",
    )

def _exit_rule_from_strategy(s: Dict[str, Any]) -> ExitRule:
    return ExitRule(
        take_profit_pct=s.get("take_profit_pct"),
        stop_loss_pct=s.get("stop_loss_pct"),
        max_holding_min=s.get("max_holding_min"),
        use_dynamic_levels=bool(s.get("use_dynamic_levels")),
    )

def _maybe_notify_entry(s: Dict[str, Any], sym: str, row: pd.Series) -> None:
    user_id = s["user_id"]; sid = s["strategy_id"]
    ts: datetime = row["timestamp"]
    price = float(row["close"])
    pos_id = f"virt:{user_id}:{sid}:{sym}"

    # 既に仮想ポジションがあればスキップ
    pos = _get_position(pos_id)
    if pos and pos["size"] and pos["size"] > 0:
        return

    # エントリー条件判定
    er = _entry_rule_from_strategy(s)
    if not _row_entry_ok(row, er):
        return

    # ポジションを開く（ロングのみ）
    size = float(s.get("trade_size_jpy") or 10000) / price
    upsert_position(
        position_id=pos_id,
        symbol=sym,
        side="long",
        size=size,
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
        signal_id=sig_id, user_id=user_id, symbol=sym, timeframe=TIMEFRAME, side="long",
        rsi=float(row.get("RSI")) if pd.notna(row.get("RSI")) else None,
        adx=float(row.get("ADX")) if pd.notna(row.get("ADX")) else None,
        atr=float(row.get("ATR")) if pd.notna(row.get("ATR")) else None,
        ema_fast=float(row.get("EMA20")) if pd.notna(row.get("EMA20")) else None,
        ema_slow=float(row.get("EMA50")) if pd.notna(row.get("EMA50")) else None,
        price=price, generated_at=ts, strategy_id=sid, version="v1", status="entry",
        raw={"rule": er.__dict__},
        source="virtual",
    )

    # 通知（JST表示）
    ts_jst = _jst(ts).strftime("%Y-%m-%d %H:%M")
    subject = f"[ENTRY] {sym} {TIMEFRAME} #{sid}"
    body = f"""[ENTRY] {sym} {TIMEFRAME}
Time(JST): {ts_jst}
Price    : {price:,.0f} JPY
Strategy : {s.get('name')}
Filters  : ADX>={s.get('adx_min')} RSI>={s.get('rsi_long_min')} ATR%>={s.get('atr_pct_min')}
Trigger  : {'MACD or ' if s.get('macd_cross') and s.get('ema_cross_req') else ('MACD' if s.get('macd_cross') else 'EMA')}
"""
    _send_email(subject, body)

def _maybe_notify_exit(s: Dict[str, Any], sym: str, row: pd.Series) -> None:
    """open中の仮想ポジションについて TP/SL/max_holding を判定"""
    user_id = s["user_id"]; sid = s["strategy_id"]
    ts: datetime = row["timestamp"]
    price = float(row["close"])
    pos_id = f"virt:{user_id}:{sid}:{sym}"
    pos = _get_position(pos_id)
    if not pos or not pos["size"] or pos["size"] <= 0:
        return

    entry_px = float(pos["avg_entry_price"]); size = float(pos["size"])
    opened_at = pos["opened_at"]; opened_at = opened_at.replace(tzinfo=timezone.utc) if opened_at.tzinfo is None else opened_at

    ex = _exit_rule_from_strategy(s)
    tp_hit = (ex.take_profit_pct is not None) and (price >= entry_px * (1 + ex.take_profit_pct))
    sl_hit = (ex.stop_loss_pct  is not None) and (price <= entry_px * (1 + ex.stop_loss_pct))
    mh_hit = (ex.max_holding_min is not None) and ((ts - opened_at) >= timedelta(minutes=ex.max_holding_min))

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
    upsert_position(position_id=pos_id, symbol=sym, side="long", size=0.0,
                    avg_entry_price=entry_px, opened_at=opened_at, updated_at=ts, raw={"closed":True})

    # シグナル保存（EXIT）
    sig_id = _stable_signal_id(user_id, sid, sym, ts, "exit")
    insert_signal(
        signal_id=sig_id,
        user_id=user_id,
        symbol=sym,
        timeframe=TIMEFRAME,
        side="flat",
        price=price,
        generated_at=ts,
        strategy_id=sid,
        version="v1",
        status="exit",
        raw={"reason": "virtual-exit", "tp": tp_hit, "sl": sl_hit, "mh": mh_hit},
        source="virtual",
    )

    # 通知（JST表示）
    ts_jst = _jst(ts).strftime("%Y-%m-%d %H:%M")
    subject = f"[EXIT]  {sym} {TIMEFRAME} #{sid}"
    reason = "TP" if tp_hit else ("SL" if sl_hit else "MAX_HOLD")
    body = f"""[EXIT-{reason}] {sym} {TIMEFRAME}
Time(JST): {ts_jst}
Price    : {price:,.0f} JPY
PnL      : {pnl:,.2f} JPY  ({pnl_pct:.2f}%)
Strategy : {s.get('name')}
"""
    _send_email(subject, body)

def main_loop():
    print("[worker] start")
    while True:
        try:
            strategies = _fetch_enabled_strategies()
            if not strategies:
                time.sleep(POLL_SEC); continue

            # 対象シンボルをユニーク化
            all_syms = sorted(set(sym for s in strategies for sym in (s.get("symbols") or [])))

            # 最新バー取得（銘柄毎）
            last_rows: Dict[str, pd.Series] = {}
            for sym in all_syms:
                row = _last_row_with_indicators(sym, lookback_min=180)
                if row is not None and "close" in row:
                    last_rows[sym] = row

            # 各戦略×銘柄で Entry/Exit 判定
            for s in strategies:
                for sym in s.get("symbols") or []:
                    row = last_rows.get(sym)
                    if row is None:
                        continue
                    _maybe_notify_exit(s, sym, row)   # 先にEXIT判定（クローズ優先）
                    _maybe_notify_entry(s, sym, row)  # 次にENTRY判定
        except Exception as e:
            print("[worker-error]", e)
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main_loop()
