from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime

@dataclass
class IndicatorSnapshot:
    rsi: Optional[float] = None
    adx: Optional[float] = None
    atr: Optional[float] = None      # 絶対値（価格同一単位）
    sma_fast: Optional[float] = None # 例: SMA20
    sma_slow: Optional[float] = None # 例: SMA50
    price: Optional[float] = None
    timeframe: str = "5m"

@dataclass
class SignalContext:
    symbol: str
    side: str                     # "BUY" | "SELL" | "EXIT-LONG" | "EXIT-SHORT"
    reason_tags: Optional[list] = None  # ["rsi_oversold","adx_trend","sma_cross",...]
    tp: Optional[float] = None
    sl: Optional[float] = None
    score: Optional[float] = None       # あれば（内部スコア）

# ---- 小さなユーティリティ（説明テキスト生成） ----
def _qual_rsi(rsi: Optional[float]) -> str:
    if rsi is None: return "RSI: データなし"
    if rsi <= 30:   return f"RSI {rsi:.1f}（売られすぎ傾向）"
    if rsi >= 70:   return f"RSI {rsi:.1f}（買われすぎ傾向）"
    return f"RSI {rsi:.1f}（中立域）"

def _qual_adx(adx: Optional[float]) -> str:
    if adx is None: return "ADX: データなし"
    if adx < 20:    return f"ADX {adx:.1f}（トレンド弱）"
    if adx < 25:    return f"ADX {adx:.1f}（やや弱いトレンド）"
    if adx < 35:    return f"ADX {adx:.1f}（トレンドあり）"
    if adx < 50:    return f"ADX {adx:.1f}（強いトレンド）"
    return f"ADX {adx:.1f}（非常に強いトレンド）"

def _qual_atr_pct(atr: Optional[float], price: Optional[float]) -> str:
    if atr is None or not price: return "ATR: データなし"
    atr_pct = 100.0 * atr / price
    if atr_pct < 0.8:   band = "低ボラ"
    elif atr_pct < 1.6: band = "中ボラ"
    else:               band = "高ボラ"
    return f"ATR {atr:.4f}（{atr_pct:.2f}%・{band}）"

def _qual_sma(sma_fast: Optional[float], sma_slow: Optional[float]) -> str:
    if sma_fast is None or sma_slow is None:
        return "SMA: データなし"
    if sma_fast > sma_slow:
        return f"SMA {sma_fast:.2f}/{sma_slow:.2f}（短期>長期＝上向き）"
    if sma_fast < sma_slow:
        return f"SMA {sma_fast:.2f}/{sma_slow:.2f}（短期<長期＝下向き）"
    return f"SMA {sma_fast:.2f}/{sma_slow:.2f}（短期≈長期）"

def _side_ja(side: str) -> str:
    return {
        "BUY": "エントリー（買い）",
        "SELL": "エントリー（売り）",
        "EXIT-LONG": "ロング決済",
        "EXIT-SHORT": "ショート決済",
    }.get(side, side)

def _reason_line(reason_tags: Optional[list]) -> str:
    if not reason_tags: return ""
    jp = {
        "rsi_oversold": "RSI売られすぎ",
        "rsi_overbought": "RSI買われすぎ",
        "adx_trend": "ADXでトレンド有",
        "sma_cross": "SMAゴールデン/デッドクロス",
        "atr_wide": "ATR拡大（高ボラ）",
        "confluence": "複数根拠の合流",
    }
    mapped = [jp.get(t, t) for t in reason_tags]
    return "根拠: " + ", ".join(mapped)

# ---- 本体：通知本文を生成 ----
def compose_signal_message(ctx: SignalContext, ind: IndicatorSnapshot, *, locale: str = "ja") -> str:
    """
    LINE/メール兼用のテキスト（Markdown互換）を返す。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"【{ctx.symbol.upper()} / {ind.timeframe}】{_side_ja(ctx.side)}"
    price_line = f"現値: {ind.price:.4f}" if ind.price else "現値: -"
    score_line = f"スコア: {ctx.score:.2f}" if ctx.score is not None else ""
    tp_sl = []
    if ctx.tp: tp_sl.append(f"TP: {ctx.tp:.4f}")
    if ctx.sl: tp_sl.append(f"SL: {ctx.sl:.4f}")
    tpsl_line = " / ".join(tp_sl)

    # 指標の短文解説
    lines = [
        title,
        f"{price_line}   {tpsl_line}   {score_line}".strip(),
        _reason_line(ctx.reason_tags),
        "",
        "— 指標サマリ —",
        _qual_rsi(ind.rsi),
        _qual_adx(ind.adx),
        _qual_atr_pct(ind.atr, ind.price),
        _qual_sma(ind.sma_fast, ind.sma_slow),
        "",
        "※ RSI: オシレーター（買われ/売られの過熱）",
        "※ ADX: トレンドの強弱",
        "※ ATR: 変動幅（ボラティリティ）",
        "※ SMA: トレンド方向（短期/長期の位置）",
        f"\n{now} JST",
    ]
    return "\n".join([l for l in lines if l is not None and l != ""])

@dataclass
class ExitPerf:
    position: str              # "long" | "short"
    entry_price: float
    exit_price: float
    size: float
    profit_jpy: float
    profit_pct: float          # 例: +2.34 は +2.34%
    holding_hours: float
    reason: Optional[str] = None  # "TP", "SL", "時間経過", など任意

def compose_exit_message(symbol: str, timeframe: str, perf: ExitPerf, *, locale: str="ja") -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pos_ja = "ロング" if perf.position == "long" else "ショート"
    title = f"【{symbol.upper()} / {timeframe}】{pos_ja}決済"

    # 1行目: PnL をドン！
    pnl_line = f"損益: {perf.profit_jpy:,.0f} 円（{perf.profit_pct:+.2f}%）"

    # 2行目: 価格とサイズの内訳
    detail_line = (
        f"Entry: {perf.entry_price:,.0f} / Exit: {perf.exit_price:,.0f} / "
        f"Size: {perf.size:g}"
    )

    # 3行目: 保有時間・理由
    hold_reason = f"保有: {perf.holding_hours:.1f} 時間"
    if perf.reason:
        hold_reason += f" / 理由: {perf.reason}"

    lines = [
        title,
        pnl_line,
        detail_line,
        hold_reason,
        "",
        f"{now} JST",
    ]
    return "\n".join(lines)