# app/indicators.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    必須列: ['timestamp','open','high','low','close','volume']
    返却: 指標列を追加した DataFrame（timestamp昇順・重複排除済）
    """
    df = df.copy()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # === ATR(14) ===
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    df["TR"] = tr
    df["ATR"] = _ema(tr, 14)

    # === DMI/ADX(14) ===
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = df["ATR"].replace(0, np.nan)

    df["+DM"] = _ema(pd.Series(plus_dm, index=df.index), 14)
    df["-DM"] = _ema(pd.Series(minus_dm, index=df.index), 14)

    df["DI+"] = (df["+DM"] / atr) * 100.0
    df["DI-"] = (df["-DM"] / atr) * 100.0

    dx = ((df["DI+"].abs() - df["DI-"].abs()).abs() / (df["DI+"].abs() + df["DI-"].abs()).replace(0, np.nan)) * 100.0
    df["ADX"] = _ema(dx, 14)

    # === RSI(14) ===
    chg = close.diff()
    gain = chg.clip(lower=0.0)
    loss = -chg.clip(upper=0.0)
    avg_gain = _ema(gain, 14)
    avg_loss = _ema(loss, 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # === MACD(12,26,9) ===
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    df["MACD"] = macd
    df["MACD_signal"] = signal
    df["MACD_hist"] = macd - signal
    df["MACD_cross_up"] = (df["MACD"].shift(1) <= df["MACD_signal"].shift(1)) & (df["MACD"] > df["MACD_signal"])
    df["MACD_cross_down"] = (df["MACD"].shift(1) >= df["MACD_signal"].shift(1)) & (df["MACD"] < df["MACD_signal"])

    # === EMA(20/50) & ゴールデンクロス ===
    df["EMA20"] = _ema(close, 20)
    df["EMA50"] = _ema(close, 50)
    df["EMA_cross_up"] = (df["EMA20"].shift(1) <= df["EMA50"].shift(1)) & (df["EMA20"] > df["EMA50"])
    df["EMA_cross_down"] = (df["EMA20"].shift(1) >= df["EMA50"].shift(1)) & (df["EMA20"] < df["EMA50"])

    df["ATR_PCT"] = (df["ATR"] / df["close"]) * 100.0

    return df
