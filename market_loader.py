# app/market_loader.py
from __future__ import annotations
import os, json
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional, Tuple
import pandas as pd

CACHE_DIR = os.getenv("MARKET_CACHE_DIR", os.path.join(os.getcwd(), "cache"))

def _to_ts(v: Any) -> Optional[pd.Timestamp]:
    """多様なタイムスタンプ表現を pandas Timestamp(UTC) に変換。失敗時 None。"""
    if v is None:
        return None
    # 文字列（ISO）
    if isinstance(v, str):
        # よくある ISO / "YYYY-mm-ddTHH:MM:SS+09:00"
        ts = pd.to_datetime(v, errors="coerce", utc=True)
        if ts is not None and not pd.isna(ts):
            return ts
        # 万一 "YYYY/mm/dd HH:MM:SS" などでも…
        try:
            return pd.to_datetime(v, utc=True)
        except Exception:
            return None
    # 数値（秒 or ミリ秒）
    try:
        x = float(v)
        if x > 1e12:   # ms
            return pd.to_datetime(int(x), unit="ms", utc=True)
        if x > 1e9:    # sec
            return pd.to_datetime(int(x), unit="s", utc=True)
    except Exception:
        pass
    return None

def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def _parse_ohlcv_list(row: Iterable[Any]) -> Optional[dict]:
    """
    row が配列のときの頑健パーサ。
    想定パターン:
      A) [ts, open, high, low, close, volume]
      B) [open, high, low, close, volume, ts]   ← よくある入れ替わり
    いずれも追加の余剰列があっても先頭/末尾から解決できるようにする。
    """
    row = list(row)
    if len(row) < 6:
        return None

    # ts 候補を先頭・末尾から判定
    ts_head = _to_ts(row[0])
    ts_tail = _to_ts(row[-1])

    # Aパターン: 先頭が ts
    if ts_head is not None:
        open_, high, low, close, vol = (_as_float(row[1]), _as_float(row[2]),
                                        _as_float(row[3]), _as_float(row[4]),
                                        _as_float(row[5]))
        return {
            "timestamp": ts_head,
            "open": open_, "high": high, "low": low, "close": close, "volume": vol
        }

    # Bパターン: 末尾が ts
    if ts_tail is not None:
        open_, high, low, close, vol = (_as_float(row[0]), _as_float(row[1]),
                                        _as_float(row[2]), _as_float(row[3]),
                                        _as_float(row[4]))
        return {
            "timestamp": ts_tail,
            "open": open_, "high": high, "low": low, "close": close, "volume": vol
        }

    # どちらでもない場合は諦める
    return None

def _parse_ohlcv_dict(d: dict) -> Optional[dict]:
    """
    dict のときのパーサ。キーのゆらぎに対応。
    許容例:
      {"timestamp": "...", "open": "...", "high": "...", "low": "...", "close": "...", "volume": "..."}
      {"openTime": 1728105600000, "open": "...", ...}
      {"time": "...", "o": "...", "h": "...", "l": "...", "c": "...", "v": "..."}
    """
    # 時刻キー候補
    t = d.get("timestamp") or d.get("openTime") or d.get("time") or d.get("t")
    ts = _to_ts(t)
    if ts is None:
        return None

    # 価格キー候補
    open_  = d.get("open")  or d.get("o")
    high   = d.get("high")  or d.get("h")
    low    = d.get("low")   or d.get("l")
    close  = d.get("close") or d.get("c")
    volume = d.get("volume") or d.get("v")

    return {
        "timestamp": ts,
        "open": _as_float(open_),
        "high": _as_float(high),
        "low":  _as_float(low),
        "close":_as_float(close),
        "volume": _as_float(volume),
    }

def _read_json_ohlcv(path: str) -> Optional[pd.DataFrame]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    rows = []

    # GMO風 {"data":{"candlestick":[{"ohlcv":[ [...], ... ]}]}}
    if isinstance(data, dict) and data.get("data") and isinstance(data["data"].get("candlestick"), list):
        for c in data["data"]["candlestick"]:
            for r in c.get("ohlcv") or []:
                if isinstance(r, dict):
                    parsed = _parse_ohlcv_dict(r)
                else:
                    parsed = _parse_ohlcv_list(r)
                if parsed:
                    rows.append(parsed)

    # フラット配列
    elif isinstance(data, list):
        for r in data:
            if isinstance(r, dict):
                parsed = _parse_ohlcv_dict(r)
            else:
                parsed = _parse_ohlcv_list(r)
            if parsed:
                rows.append(parsed)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # UTC に正規化 → 表示や比較は必要に応じてローカル変換してください
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # 欠損掃除
    df = df.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp")
    # 直列で重複排除
    df = df.drop_duplicates(subset=["timestamp"])
    return df[["timestamp","open","high","low","close","volume"]]

def load_ohlcv_15m(symbol: str, lookback_min: int = 24*60) -> Optional[pd.DataFrame]:
    """
    cache_dir/{symbol}_15min_YYYYMMDD.json を今日/昨日/一昨日の最大3ファイル連結
    """
    today = datetime.now(tz=timezone.utc)
    frames = []
    for d in (0,1,2):
        dt = today - timedelta(days=d)
        fn = f"{symbol}_15min_{dt.strftime('%Y%m%d')}.json"
        path = os.path.join(CACHE_DIR, fn)
        if os.path.exists(path):
            df = _read_json_ohlcv(path)
            if df is not None and not df.empty:
                frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # 直近 lookback_min 分のみ
    cutoff = today - timedelta(minutes=lookback_min + 5)
    df = df[df["timestamp"] >= cutoff]

    # 数値化の安全化
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)
