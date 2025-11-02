import requests
import pandas as pd
import numpy as np
from datetime import datetime,date, timedelta, timezone
import time
import json
import os
import logging
import smtplib
import shutil
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import traceback
import sys
import argparse
import hmac
import hashlib
import random  # ランダムな待機時間のために追加
import uuid    # SIM注文ID発行用（SIM-...）
import concurrent.futures
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket
from db import (
    begin, text,
    insert_order, upsert_fill,                 # ★ upsert_fill を追加
    upsert_position, insert_trade, insert_balance_snapshot, utcnow, insert_error
)
from db import get_line_user_id, get_line_user_ids_for_users  # 既に追加済みのDB関数
from db import get_trades_between
from db import insert_signal
from db import update_signal_status
from db import upsert_price_cache
from db import mark_order_executed_with_fill
from uuid import UUID

is_backtest_mode = "backtest" in sys.argv

if not is_backtest_mode:
    from notifiers.line_messaging import LineMessaging
    from notifications.message_templates import compose_exit_message, ExitPerf
    from notifications.message_templates import compose_signal_message, IndicatorSnapshot, SignalContext
else:
    class LineMessaging:
        def send_text(self, *args, **kwargs): return True
        def send_text_bulk(self, *args, **kwargs): return True
    def compose_exit_message(*args, **kwargs): return "Exit message"
    class ExitPerf: pass
    def compose_signal_message(*args, **kwargs): return "Signal message"
    class IndicatorSnapshot: pass
    class SignalContext: pass
try:
    from reports.daily_report import build_daily_report_message
except ImportError:
    from daily_report import build_daily_report_message
from typing import Optional, List, Dict, Tuple, Any
from email.header import Header
import datetime as dt
JST = timezone(timedelta(hours=9))

def now_jst() -> dt.datetime:
    """常にJSTのaware datetimeを返す"""
    return dt.datetime.now(JST)

# 既存のimport文の後に追加
try:
    from excel_report_generator import ExcelReportGenerator
    EXCEL_REPORT_AVAILABLE = True
    print("✅ Excel自動評価レポート機能が利用可能です")
except ImportError:
    EXCEL_REPORT_AVAILABLE = False
    print("⚠️ Excel自動評価レポート機能が利用できません（excel_report_generator.pyが見つかりません）")

def _num(x):
    """NUMERIC列に安全に入れられる値だけ返す: None/NaN/Inf は None に"""
    try:
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None

def _is_sim_run(self) -> bool:
    try:
        return not self.exchange_settings_gmo.get("live_trade", False)
    except Exception:
        return True

def _make_session():
    s = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(['GET', 'POST']),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

HTTP_SESSION = _make_session()

def _is_dns_temp_failure(err: Exception) -> bool:
    """
    DNS一時失敗（EAI_AGAINなど）を検出
    """
    # requests -> urllib3 -> socket.gaierror が cause に入ることがある
    e = err
    while e:
        if isinstance(e, socket.gaierror):
            # Linux系で Errno -3 が "Temporary failure in name resolution"
            return True
        e = getattr(e, "__cause__", None)
    return False

def to_iso8601(v):
    """文字列やdatetimeをISO8601(+TZ)に正規化"""
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=JST)  # naiveはJST前提
        return v.isoformat()
    if isinstance(v, str):
        s = v.strip()
        # "YYYY-mm-dd HH:MM:SS.sss +0900" を許可
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f %z")
            return dt.isoformat()
        except Exception:
            pass
        # naiveなISO文字列 "2025-09-24T12:34:56.123456" → JST付与
        if "T" in s and ("+" not in s and "Z" not in s):
            return s + "+09:00"
    return v

def row_to_report_trade(row):
    """DBから取った行を日次レポート用に整形"""
    return {
        "symbol": row.get("symbol") or row.get("pair"),
        "side": row.get("type") or row.get("position_type"),
        "exit_time": to_iso8601(row.get("closed_at") or row.get("exit_time")),
        "exit_price": row.get("exit_price"),
        "pnl": row.get("pnl") if row.get("pnl") is not None else row.get("profit"),
        "profit": row.get("profit") if row.get("profit") is not None else row.get("pnl"),
        "entry_price": row.get("entry_price"),
        "holding_hours": row.get("holding_hours"),
    }

def to_uuid_or_none(v: str | None) -> str | None:
    if not v:
        return None
    try:
        # 正規化した文字列表現（ハイフン付き）に変換
        return str(UUID(str(v)))
    except Exception:
        return None

def default_source_from_env(self) -> str:
    # liveならreal、バックテストならbacktest、なければsim
    if getattr(self, "is_backtest_mode", False):
        return "backtest"
    if self.exchange_settings_gmo.get("live_trade"):
        return "real"
    return "sim"

class PriceCacheRefresher:
    def __init__(self, symbols, get_price_fn, interval_sec=30, logger=None):
        self.symbols = symbols
        self.get_price = get_price_fn
        self.interval = interval_sec
        self.logger = logger

    def run_forever(self):
        import time
        while True:
            start = time.time()
            for sym in self.symbols:
                try:
                    px = self.get_price(sym)
                    if px and px > 0:
                        upsert_price_cache(sym.lower(), px, dt.datetime.now(timezone.utc))
                        if self.logger:
                            self.logger.debug(f"[price_cache] {sym} = {px}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"[price_cache] {sym} failed: {e}")
            elapsed = time.time() - start
            time.sleep(max(0, self.interval - elapsed))

class GMOCoinAPI:
    """GMOコインの信用取引APIラッパー"""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coin.z.com/private"
        self.public_url = "https://api.coin.z.com/public"
        
    def _sign(self, method, timestamp, path, reqBody=""):
        """リクエストの署名を作成"""
        text = timestamp + method + path + (reqBody if reqBody else "")
        return hmac.new(
            self.api_secret.encode('utf-8'),
            text.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method, path, params=None, data=None, *, timeout=8):
        """
        APIリクエスト共通処理（署名は送信直前に毎回生成。リトライごとにtimestampを更新）
        """
        base_url = self.base_url + path

        # GETクエリをURLへ付与（署名には含めない：pathのみでOK）
        url = base_url
        if method == "GET" and params:
            q = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{base_url}?{q}"

        body_str = json.dumps(data, separators=(',', ':')) if data else ""

        # 最大3回リトライ（HTTP_SESSION側にもRetryはあるが、ここではtimestamp再生成のため明示）
        for attempt in range(3):
            ts = str(int(time.time() * 1000))  # ← 毎回作り直す（Timestamp too late対策）
            headers = {
                "API-KEY": self.api_key,
                "API-TIMESTAMP": ts,
                "API-SIGN": self._sign(method, ts, path, body_str),
            }
            if method != "GET":
                headers["Content-Type"] = "application/json"

            try:
                if method == "GET":
                    with HTTP_SESSION.get(url, headers=headers, timeout=timeout) as r:
                        r.raise_for_status()
                        return r.json()
                else:
                    with HTTP_SESSION.post(url, headers=headers, data=body_str, timeout=timeout) as r:
                        r.raise_for_status()
                        return r.json()

            except requests.exceptions.RequestException as e:
                # DNS一時失敗は少し待って再試行
                if _is_dns_temp_failure(e):
                    time.sleep(1.2 * (attempt + 1))
                    continue

                # GMO特有の "Timestamp for this request is too late." は再署名して再送
                msg = str(e)
                if "Timestamp for this request is too late" in msg or "status code: 400" in msg:
                    time.sleep(0.8 * (attempt + 1))
                    continue

                # それ以外は最後にJSON化エラーに備えて安全に返す
                if attempt == 2:
                    return {"status": -1, "messages": [{"message_string": f"{type(e).__name__}: {e}"}]}
                time.sleep(0.6 * (attempt + 1))
                continue

            except Exception as e:
                if attempt == 2:
                    return {"status": -1, "messages": [{"message_string": f"{type(e).__name__}: {e}"}]}
                time.sleep(0.6 * (attempt + 1))

        # ここには基本来ない
        return {"status": -1, "messages": [{"message_string": "unknown error"}]}

    def get_margin_positions(self, symbol):
        """信用取引のポジション情報を取得
        
        Parameters:
        symbol (str): 必須。通貨ペアシンボル（例: 'BTC_JPY'）
        """
        path = "/v1/openPositions"
        params = {"symbol": symbol}
        
        return self._request("GET", path, params=params)
    
    def order_margin(self, symbol, side, position_type, size, order_type="MARKET"):
        """信用取引の注文"""
        path = "/v1/order"
        data = {
            "symbol": symbol,
            "side": side,  # BUY or SELL
            "executionType": order_type,  # MARKET or LIMIT
            "settlePosition": position_type,  # OPEN or CLOSE
            "size": str(size)
        }
        
        # 信用取引の場合は leverage type を追加
        if position_type == "OPEN":
            data["marginTradeType"] = "SHORT" if side == "SELL" else "LONG"
        
        return self._request("POST", path, data=data)
    
    def get_balance(self):
        """資産情報を取得"""
        path = "/v1/account/assets"
        return self._request("GET", path)

    def close_position(self, symbol, position_id, size, side, position_type="limit", price=None):
        """ポジションを決済する（公式例準拠）
        
        Parameters:
        symbol (str): 通貨ペアシンボル
        position_id (int): ポジションID
        size (str): 決済サイズ
        side (str): 取引サイド（BUY or SELL）
        position_type (str): 注文タイプ（LIMIT or MARKET）
        price (str): 指値価格（LIMIT注文の場合）
        """
        path = "/v1/closeOrder"
        
        data = {
            "symbol": symbol,
            "side": side,
            "executionType": position_type.upper(),
            "settlePosition": [
                {
                    "positionId": position_id,
                    "size": str(size)
                }
            ]
        }
        
        # LIMIT注文の場合は価格とタイムインフォースを追加
        if position_type.upper() == "LIMIT" and price:
            data["price"] = str(price)
            data["timeInForce"] = "FAK"  # Fill and Kill
        
        return self._request("POST", path, data=data)

class CryptoTradingBot:
    def _is_live_mode(self) -> bool:
        """
        ランタイム状態から LIVE かどうかを堅牢に判定する。
        優先順位:
        1) self.mode が "live" / ("paper","sim","simulation","backtest")
        2) self.exchange_settings_gmo.get("live_trade")
        3) self.live_trade (bool フラグ)
        """
        # 1) 明示 mode 優先
        mode = getattr(self, "mode", None)
        if isinstance(mode, str):
            m = mode.lower()
            if m == "live":
                return True
            if m in ("paper", "sim", "simulation", "backtest"):
                return False

        # 2) 設定での明示フラグ
        es = getattr(self, "exchange_settings_gmo", {}) or {}
        if isinstance(es, dict) and "live_trade" in es:
            # True/False が入っている前提。None などはスキップ
            val = es.get("live_trade")
            if isinstance(val, bool):
                return val

        # 3) 後方互換: 単独フラグ
        return bool(getattr(self, "live_trade", False))


    def _is_sim_mode(self) -> bool:
        """LIVE でなければ SIM/PAPER/BACKTEST とみなす簡易判定"""
        return not self._is_live_mode()

    def _prepare_trade_logs_for_excel_report(self, trade_logs):
        """Excel レポート用に取引ログを準備"""
        try:
            if not trade_logs:
                return pd.DataFrame()
            
            # DataFrameに変換
            df = pd.DataFrame(trade_logs)
            
            # 必要な列の名前変更・追加
            if 'type' in df.columns:
                df['position'] = df['type']  # ExcelReportGeneratorが期待する列名
            
            # entry_timeをdatetime型に変換
            if 'entry_time' in df.columns:
                df['entry_time'] = pd.to_datetime(df['entry_time'])
            
            # holding_period列の追加
            if 'holding_hours' in df.columns:
                df['holding_period'] = df['holding_hours']
            
            # 数値列の型確認
            numeric_columns = ['profit', 'profit_pct', 'entry_price', 'exit_price', 'size']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # ATR, ADXなどのテクニカル指標列を追加（存在しない場合はダミー値）
            if 'ATR' not in df.columns:
                df['ATR'] = np.random.uniform(0.5, 2.0, len(df))  # ダミー値
            if 'ADX' not in df.columns:
                df['ADX'] = np.random.uniform(15, 40, len(df))  # ダミー値
            
            # buy_score, sell_scoreがない場合はダミー値を設定
            # buy_score, sell_scoreが存在しない場合は NaN
            if 'buy_score' not in df.columns:
                df['buy_score'] = np.nan
            if 'sell_score' not in df.columns:
                df['sell_score'] = np.nan

            
            return df
            
        except Exception as e:
            self.logger.error(f"Excel用データ準備エラー: {str(e)}")
            return pd.DataFrame()

    def _current_strategy_version(self):
        base = {
            "date": now_jst().strftime("%Y-%m-%d"),
            "thresholds": self.entry_thresholds,
        }
        h = hashlib.sha1(json.dumps(base, sort_keys=True).encode()).hexdigest()[:7]
        return f"{base['date']}+{h}"

    def _generate_excel_report_from_trade_logs(self, trade_logs, days_to_test):
        """取引ログからExcelレポートを生成"""
        try:
            if not EXCEL_REPORT_AVAILABLE:
                self.logger.warning("Excel自動評価レポート機能が利用できません")
                return None
            
            # データ準備
            df = self._prepare_trade_logs_for_excel_report(trade_logs)
            
            if df.empty:
                self.logger.warning("Excel レポート用のデータが空です")
                return None
            
            # レポートジェネレーターを作成
            report_generator = ExcelReportGenerator(df)
            
            # 各シートを追加
            report_generator.add_summary_sheet()
            report_generator.add_score_analysis_sheet()
            report_generator.add_indicator_tables_like_scores()
            report_generator.add_market_condition_sheet()
            report_generator.add_charts_sheet()
            report_generator.add_symbol_sheets()
            
            # ファイル保存
            timestamp = now_jst().strftime('%Y%m%d_%H%M%S')
            report_filename = f"excel_evaluation_report_{timestamp}.xlsx"
            report_path = os.path.join(self.log_dir, report_filename)
            
            report_generator.save(report_path)
            
            self.logger.info(f"Excel評価レポートが生成されました: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Excel評価レポート生成エラー: {str(e)}", exc_info=True)
            return None

    def _get_open_positions_from_db(self, gmo_symbol: str | None = None):
        """
        SIM/PAPER 用。positions テーブルから size>0 のオープンポジションを取得し、
        GMOの openPositions 互換っぽいフォーマット {"status": 0, "data": {"list": [...]}} に整形して返す。
        - strategy_id は DB 側が UUID 型を想定。UUID に変換できた場合のみ絞り込みに使う。
        """
        import os
        from uuid import UUID

        def _to_uuid_or_none(v):
            try:
                if v is None:
                    return None
                return str(UUID(str(v)))
            except Exception:
                return None

        try:
            # source は SIM 固定（live_trade=False の前提）
            src = "real" if self.exchange_settings_gmo.get("live_trade", False) else "sim"
            # strategy_id を UUID に正規化できた場合のみ使う
            sid = _to_uuid_or_none(getattr(self, "strategy_id", None))
            uid = getattr(self, "user_id", None)

            from db import begin, text  # 既存の db.py のヘルパーを利用
            rows = []
            with begin() as conn:
                params = {"src": src}
                where_extra = ""

                # ユーザーID
                if uid is not None:
                    where_extra += " AND user_id = :uid"
                    params["uid"] = uid

                # strategy_id（UUIDのみ許可。文字列IDなら条件を外す）
                if sid is not None:
                    where_extra += " AND strategy_id = :sid"
                    params["sid"] = sid

                # シンボル（"LTC_JPY" → "ltc_jpy" に合わせて保存されている想定）
                if gmo_symbol:
                    bot_symbol = gmo_symbol.lower()
                    where_extra += " AND symbol = :sym"
                    params["sym"] = bot_symbol

                sql = f"""
                    SELECT position_id, symbol, side, size, avg_entry_price, opened_at
                    FROM positions
                    WHERE size > 0
                    AND source = :src
                    {where_extra}
                """
                res = conn.execute(text(sql), params)
                rows = [dict(r._mapping) for r in res]

            lst = []
            for r in rows:
                bot_symbol = r["symbol"]  # 例: "xrp_jpy"
                lst.append({
                    "positionId": r["position_id"],
                    "symbol": self.symbol_mapping.get(bot_symbol, bot_symbol.upper().replace("_", "")),
                    "side": "BUY" if r["side"] == "long" else "SELL",
                    "size": float(r["size"]),
                    "price": float(r.get("avg_entry_price") or 0.0),
                    "openedAt": r.get("opened_at"),
                })
            return {"status": 0, "data": {"list": lst}}

        except Exception as e:
            self.logger.error(f"[SIM] DBからポジション取得に失敗: {e}", exc_info=True)
            return {"status": -1, "messages": [{"message_string": str(e)}]}
            
    def _positions_filepath(self, source: str | None = None) -> str:
        """モードに応じた positions ファイルパスを返す"""
        src = source or getattr(self, "source", None) or "real"
        base = getattr(self, "data_dir", ".")
        # real は従来名を維持、sim/backtest はサフィックスで分離
        name = "positions.json" if src == "real" else f"positions_{src}.json"
        return os.path.join(base, name)

    def _get_margin_positions_safe(self, gmo_symbol: str):
        """
        liveモードかつ self.gmo_api がある場合のみ実APIを呼び、
        それ以外（paperモードや未初期化）は空リストでフォールバックする。
        戻り値フォーマットは既存利用箇所に合わせて data: [] を基本形とする。
        """
        if not self.exchange_settings_gmo.get("live_trade", False):
            return self._get_open_positions_from_db(gmo_symbol)

        is_live = bool(self.exchange_settings_gmo.get("live_trade", False))
        if is_live and self.gmo_api is not None:
            return self.gmo_api.get_margin_positions(gmo_symbol)
        # ← SIM/PAPER はDBから取得
        return self._get_open_positions_from_db(gmo_symbol)

    def _record_signal(self, *, symbol: str, timeframe: str, side: str, price: float,
                   strength_score: float | None,
                   indicators: dict[str, float] | None,
                   strategy_id: str | None = None,
                   version: str | None = None,
                   status: str = "new",
                   raw: dict | None = None,
                   signal_id: str | None = None) -> str:
        try:
            rsi = indicators.get("RSI") if indicators else None
            adx = indicators.get("ADX") if indicators else None
            atr = indicators.get("ATR") if indicators else None
            di_p = indicators.get("DI+") if indicators else None
            di_m = indicators.get("DI-") if indicators else None
            ema_fast = indicators.get("EMA_fast") if indicators else None
            ema_slow = indicators.get("EMA_slow") if indicators else None

            sid = insert_signal(
                user_id=getattr(self, "user_id", None),
                symbol=symbol,
                timeframe=timeframe,
                side=side,
                strength_score=strength_score,
                rsi=rsi, adx=adx, atr=atr,
                di_plus=di_p, di_minus=di_m,
                ema_fast=ema_fast, ema_slow=ema_slow,
                price=price,
                strategy_id=strategy_id, version=version,
                status=status,
                raw=raw,
                signal_id=signal_id,
                source=default_source_from_env(self),  # ★追加
            )
            self.logger.info(f"✅ シグナル記録: signal_id={sid} {symbol} {timeframe} {side} price={price}")
            return sid
        except Exception as e:
            self.logger.error(f"シグナル記録エラー: {e}", exc_info=True)
            return ""

    def get_total_balance(self):
        """
        GMOコインの総資産額を取得する（現物＋信用取引評価額）
        
        Returns:
        float: 総資産額（JPY）
        """
        try:
            # paper(=sim) モードでは API を使わず初期残高でフォールバック
            if not self.exchange_settings_gmo.get("live_trade", False):
                init = float(getattr(self, "paper_initial_balance", getattr(self, "initial_capital", 1_000_000.0)))
                self.logger.info(f"Paper mode: 総資産額は初期残高でフォールバック: {init:,.0f}円")
                return init
            
            # GMO APIが利用可能か確認
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません。総資産額を取得できません。")
                return 0.0
            
            total_balance = 0.0
            
            # 1. 信用取引の証拠金情報を取得
            margin_info_response = self.gmo_api._request("GET", "/v1/account/margin")
            if margin_info_response.get("status") == 0:
                margin_data = margin_info_response.get("data", {})
                
                # 証拠金残高（これが基本的な口座残高）
                available_amount = float(margin_data.get("availableAmount", 0))
                margin_amount = float(margin_data.get("marginAmount", 0))
                profit_loss = float(margin_data.get("profitLoss", 0))
                
                # 総資産 = 証拠金 + 評価損益
                total_balance = available_amount + profit_loss
                
                self.logger.info(f"証拠金残高: {available_amount:,.0f}円")
                self.logger.info(f"必要証拠金: {margin_amount:,.0f}円")
                self.logger.info(f"評価損益: {profit_loss:,.0f}円")
            else:
                error_msg = margin_info_response.get("messages", [{"message_string": "不明なエラー"}])[0].get("message_string", "不明なエラー")
                self.logger.error(f"GMOコイン証拠金情報取得エラー: {error_msg}")
                
                # 証拠金情報が取得できない場合、現物資産情報を取得
                assets_response = self.gmo_api.get_balance()
                
                if assets_response.get("status") == 0:
                    assets = assets_response.get("data", [])
                    for asset in assets:
                        symbol = asset.get("symbol", "").lower()
                        
                        # JPY現物
                        if symbol == "jpy":
                            available = float(asset.get("available", 0))
                            total_balance += available
                        else:
                            # 暗号資産現物
                            available = float(asset.get("available", 0))
                            if available > 0:
                                # 現在価格を取得
                                current_price = self.get_current_price(f"{symbol}_jpy")
                                if current_price > 0:
                                    asset_value = available * current_price
                                    total_balance += asset_value
            
            # 2. 個別のポジション情報を取得（詳細ログ用）
            try:
                margin_positions_response = self.gmo_api.get_margin_positions()
                
                if margin_positions_response.get("status") == 0:
                    positions = margin_positions_response.get("data", {}).get("list", [])
                    
                    for position in positions:
                        symbol = position.get("symbol")
                        size = float(position.get("size", 0))
                        price = float(position.get("price", 0))
                        side = position.get("side")
                        position_id = position.get("positionId")
                        
                        # 現在価格を取得
                        bb_symbol = None
                        for bb_sym, gmo_sym in self.symbol_mapping.items():
                            if gmo_sym == symbol:
                                bb_symbol = bb_sym
                                break
                        
                        if bb_symbol:
                            current_price = self.get_current_price(bb_symbol)
                            if current_price > 0:
                                # ポジションの評価損益を計算（ログ用）
                                if side == "BUY":
                                    profit = (current_price - price) * size
                                else:  # SELL
                                    profit = (price - current_price) * size
                                
            except Exception as e:
                self.logger.debug(f"個別ポジション情報取得時のエラー（無視）: {e}")
            
            self.logger.info(f"GMOコイン総資産額: {total_balance:,.0f}円")
            return total_balance
            
        except Exception as e:
            self.logger.error(f"GMOコイン総資産額取得エラー: {e}", exc_info=True)
            return 0.0

    def __init__(self, initial_capital=200000, test_mode=True, user_id=None):
        """
        トレーディングボットの初期化
        
        Parameters:
        initial_capital (float): 初期資金 (円)
        test_mode (bool): テストモード（少額取引）の場合はTrue
        """

        self.user_id = user_id  # ← DBから通知設定/トークンを引くための文脈

        # ディレクトリ設定（最初に設定する必要あり）
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(self.base_dir, 'data_cache')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.backup_dir = os.path.join(self.base_dir, 'backups')
        
        # ディレクトリが存在しない場合は作成
        for directory in [self.cache_dir, self.log_dir, self.backup_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # 基本設定
        self.initial_capital = initial_capital
        self.test_mode = test_mode
        self.symbols = ["ltc_jpy", "xrp_jpy", "sol_jpy", "doge_jpy", "eth_jpy", "bcc_jpy", "ada_jpy"]
        self.positions = {symbol: None for symbol in self.symbols}  # None, 'long', 'short'
        self.entry_prices = {symbol: 0 for symbol in self.symbols}
        self.entry_times = {symbol: None for symbol in self.symbols}
        self.entry_sizes = {symbol: 0 for symbol in self.symbols}  # 注文サイズ保存用
        self.entry_scores = {symbol: {} for symbol in self.symbols}
        
        # 運用資金増加の管理用変数
        self.monthly_increase = 500  # 月間増資額
        self.last_increase_date = self._load_last_increase_date()  # 前回の増資日
        
        # ログ設定
        self.setup_logging()
        
        # 累積利益の読み込み
        self.total_profit = self._load_total_profit()
        
        # 取引サイズ設定
        if self.test_mode:
            self.TRADE_SIZE = 20000  # テストモード
        else:
            self.TRADE_SIZE = 20000  # 通常取引額
            
        # API呼び出し制限管理
        self.last_api_call = time.time() - 1
        self.rate_limit_delay = 0.5  # 0.5秒に増加（APIレート制限対策）
        
        # 通知設定（メール）
        self.notification_settings = {
            'enabled': True,
            'email': {
                'smtp_server': "smtp.gmail.com",
                'smtp_port'  : 587,
                # ★KeyError回避：getenvで取得（未設定なら空文字）
                'sender'     : os.getenv("EMAIL_ADDRESS_PIP", ""),
                'password'   : os.getenv("EMAIL_PASSWORD_PIP", ""),
                'recipient'  : "tomokomi1107@gmail.com",
            },
            'send_on_entry': True,
            'send_on_exit' : True,
            'send_on_error': True,
            'daily_report' : True,
        }

        # ★SIM対応：環境変数は get で取得（未設定でもエラーにしない）
        _env_api_key = os.getenv("GMO_API_KEY", "")
        _env_api_secret = os.getenv("GMO_API_SECRET", "")
        self.exchange_settings_gmo = {
            'api_key': _env_api_key,
            'api_secret': _env_api_secret,
            # ★初期値は False に変更（後で mode に応じて上書き）
            'live_trade': False,
        }
        # ★APIは lazy init（run_live等でlive_trade=Trueが確定してから作る）
        self.gmo_api = None

        # ポジションID管理を追加
        self.position_ids = {symbol: None for symbol in self.symbols}  # ポジションIDを保存


        # 通貨ペアマッピング（BitBank形式からGMO形式へ）
        self.symbol_mapping = {
            'ltc_jpy': 'LTC_JPY',
            'xrp_jpy': 'XRP_JPY', 
            'sol_jpy': 'SOL_JPY',
            'doge_jpy': 'DOGE_JPY',
            'eth_jpy': 'ETH_JPY',
            'bcc_jpy': 'BCH_JPY',
            'ada_jpy': 'ADA_JPY'  # 追加
        }
        
        # バックアップスケジュール
        self.last_backup_time = time.time()
        self.backup_interval = 24 * 60 * 60  # 24時間ごとにバックアップ
        
        # 最新の有効なデータ日付を記録
        self.valid_dates = {}
        
        # 市場センチメントを初期化
        self.sentiment = {
            'bullish': 50,  # 強気度合い（0-100）
            'bearish': 50,  # 弱気度合い（0-100）
            'volatility': 50,  # ボラティリティ（0-100）
            'trend_strength': 50,  # トレンド強度（0-100）
        }
        
        # 通貨ペアごとの最小注文量
        self.min_order_sizes = {
            'ltc_jpy': 1,       # LTC
            'xrp_jpy': 10,      # XRP
            'sol_jpy': 0.1,     # SOL
            'doge_jpy': 10,     # DOGE
            'eth_jpy': 0.01,    # ETH
            'bcc_jpy': 0.1,     # BCH
            'ada_jpy': 10      # ADA
        }
        self.entry_thresholds = {
            "adx_trend_min": 25.0,
            "rsi_long_min": 58.0,
            "rsi_short_max": 42.0,
            "score_long_min": 0.55,
            "score_short_min": 0.55,
            "ema_cross_required": True,
        }

        self.strategy_id = getattr(self, "strategy_id", "v1_weighted_signals")

        self.reentry_block_until = {symbol: None for symbol in self.symbols}
        
        # まずは暫定のファイルパス（source 未確定でも呼べる）
        self.positions_path = self._positions_filepath()

        self.logger.info(f"=== ボット初期化完了 （初期資金: {initial_capital:,}円, テストモード: {test_mode}) ===")

    def setup_logging(self):
        """ロギング設定（毎日 0:00 に日次ローテーション、JST想定）"""
        import logging
        from logging.handlers import TimedRotatingFileHandler
        import os
        import time as _time

        # JSTで回したい場合の保険（Linux）
        try:
            os.environ.setdefault("TZ", "Asia/Tokyo")
            if hasattr(_time, "tzset"):
                _time.tzset()
        except Exception:
            pass

        self.logger = logging.getLogger("crypto_bot")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 二重出力防止

        # 既存ハンドラをクリア（多重追加防止）
        for h in list(self.logger.handlers):
            try:
                h.close()
            except Exception:
                pass
            self.logger.removeHandler(h)

        # ログディレクトリ作成
        os.makedirs(self.log_dir, exist_ok=True)

        # ログファイルパス
        # ログを source(real/sim/backtest) ごとに分離
        try:
            _src = getattr(self, "source", None)
        except Exception:
            _src = None
        if not _src:
            try:
                _src = default_source_from_env(self)
            except Exception:
                _src = "unknown"
        base_log_path = os.path.join(self.log_dir, f"crypto_bot_{_src}.log")

        # 日次ローテーション（ローカルタイムの真夜中）
        self.file_handler = TimedRotatingFileHandler(
            filename=base_log_path,
            when="midnight",
            interval=1,
            backupCount=7,     # 古いログは20個まで保持
            encoding="utf-8",
            utc=False           # JSTに設定済みならJSTで回る
        )

        # フォーマット設定
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        self.file_handler.setFormatter(formatter)
        self.file_handler.setLevel(logging.INFO)

        # コンソール出力も保持
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(formatter)
        self.console_handler.setLevel(logging.INFO)

        # ハンドラ登録
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)


    
    def send_notification(self, subject, message, notification_type='info'):
        """通知を送信（メールとLINE）
        
        Parameters:
        subject (str): 通知の件名
        message (str): 通知の本文
        notification_type (str): 通知の種類 ('info', 'entry', 'exit', 'error')
        """
        if not self.notification_settings['enabled']:
            return
            
        # 通知タイプに基づいて送信するかどうかを判断
        should_send = False
        if notification_type == 'entry' and self.notification_settings['send_on_entry']:
            should_send = True
        elif notification_type == 'exit' and self.notification_settings['send_on_exit']:
            should_send = True
        elif notification_type == 'error' and self.notification_settings['send_on_error']:
            should_send = True
        elif notification_type == 'daily_report' and self.notification_settings['daily_report']:
            should_send = True
        elif notification_type == 'info':
            should_send = True
        
        if not should_send:
            return
            
        try:
            # 1) メール
            self._send_email(subject, message)
            # 2) LINE（Messaging API）
            self._send_line(subject, message)
            self.logger.info(f"通知送信完了: {subject}")
        except Exception as e:
            self.logger.error(f"通知送信エラー: {e}")

    def _send_daily_report(self, stats: dict) -> None:
        """
        その日のトレードを集計して日次レポート（メール/LINE）を送る。
        run_live 内の 0:00 トリガーで呼ばれる想定。
        """
        try:
            # --- 対象日のJST範囲を決定（force_day_start を優先） ---
            force_start = (stats or {}).get("force_day_start")
            if force_start:
                start = force_start
            else:
                jst_now = dt.datetime.now(JST)
                start = jst_now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)

            # --- DBから当日分のクローズ済みトレードを取得 ---
            try:
                from db import get_trades_between
                trades = get_trades_between(start, end) or []
            except Exception as e:
                self.logger.warning(f"DBからのトレード取得に失敗。メモリ上のログにフォールバック: {e}")
                trades = (stats.get("today_trade_logs", []) if isinstance(stats, dict) else []) or []
                trades += getattr(self, "daily_exit_logs", [])

            # --- 重複除外 ---
            seen, deduped = set(), []
            for t in trades:
                key = (
                    t.get("exit_order_id"),
                    t.get("closed_at") or t.get("exit_time") or t.get("time"),
                    t.get("symbol") or t.get("pair"),
                    t.get("entry_price"), t.get("exit_price"),
                    t.get("size"),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(t)
            trades = deduped

            # --- 正規化（外出し済みユーティリティを使用） ---
            normalized_trades = [row_to_report_trade(t) for t in trades]

            # デバッグに役立つ簡易ログ（必要なら残す）
            self.logger.info(f"[daily] raw={len(deduped)}, normalized={len(normalized_trades)}")
            if normalized_trades:
                s = normalized_trades[0]
                self.logger.info(f"[daily] sample exit_time={s.get('exit_time')} pnl={s.get('pnl')} profit={s.get('profit')} symbol={s.get('symbol')}")

            # --- メッセージ生成（JSTの start を day に渡す） ---
            subject, message = build_daily_report_message(normalized_trades, day=start)

            # --- 通知送信 ---
            self.send_notification(
                subject=subject,
                message=message,
                notification_type="daily_report",
            )
            self.logger.info("✅ 日次レポート送信完了")

        except Exception as e:
            self.logger.error(f"日次レポート送信エラー: {e}", exc_info=True)


    def _notify_signal(self, *args, **kwargs):
        """
        互換レイヤー（旧/新どちらの呼び方も吸収）
        受け付ける形:
        1) _notify_signal(subject, message)
        2) _notify_signal(kind, subject, message)  # kind='ENTRY'|'EXIT'|'ERROR' 等
        3) _notify_signal(**kwargs)  # 例: side, symbol, price, tp, sl, score, reason_tags, ...
            ※ df_5min など巨大データは通知文から除外
        """
        notification_type = kwargs.pop("notification_type", None)

        # --- 旧: 位置引数系 ---
        if args:
            if len(args) == 3:
                kind, subject, message = args
                nt = notification_type or (str(kind).lower() if isinstance(kind, str) else "info")
                return self.send_notification(subject, message, notification_type=nt)
            if len(args) == 2:
                subject, message = args
                nt = notification_type or "info"
                return self.send_notification(subject, message, notification_type=nt)
            raise TypeError(f"_notify_signal unexpected signature: args={args}, kwargs={kwargs}")

        # --- 新: キーワード引数系（EXIT側はこちらで来る想定） ---
        # 期待キー: side, symbol, price, tp, sl, score, reason_tags など
        side    = kwargs.get("side") or kwargs.get("kind") or "SIGNAL"
        symbol  = kwargs.get("symbol", "-")
        price   = kwargs.get("price")
        tp      = kwargs.get("tp")
        sl      = kwargs.get("sl")
        score   = kwargs.get("score")
        reasons = kwargs.get("reason_tags") or kwargs.get("reasons") or []

        # 件名
        subject = f"{side} {symbol}".upper()

        # 本文（巨大オブジェクトは含めない: df_5min 等は無視）
        lines = []
        if price is not None:
            try:
                lines.append(f"Price: {price:,.0f}")
            except Exception:
                lines.append(f"Price: {price}")
        if tp is not None:
            try:
                lines.append(f"TP: {tp:,.0f}")
            except Exception:
                lines.append(f"TP: {tp}")
        if sl is not None:
            try:
                lines.append(f"SL: {sl:,.0f}")
            except Exception:
                lines.append(f"SL: {sl}")
        if score is not None:
            lines.append(f"Score: {score}")
        if reasons:
            lines.append("Reasons: " + ", ".join(map(str, reasons)))

        message = "\n".join(lines) if lines else "(no details)"

        # 通知タイプ（未指定なら side から推測）
        if notification_type:
            nt = notification_type
        else:
            s = str(side).upper()
            if "EXIT" in s:
                nt = "exit"
            elif "ENTRY" in s:
                nt = "entry"
            elif "ERROR" in s or "FAIL" in s:
                nt = "error"
            else:
                nt = "info"

        return self.send_notification(subject, message, notification_type=nt)

    def _send_line(self, subject: str, body: str) -> None:
        """LINE Messaging API 送信（宛先は DB→ENV の順で解決。なければ黙ってスキップ）"""
        try:
            # 宛先解決：DB（user_id 紐づけ）→ 環境変数 LINE_DEFAULT_USER_ID
            line_user_id = None
            if getattr(self, "user_id", None) is not None:
                line_user_id = get_line_user_id(self.user_id)  # U で始まる 32桁のhex
            if not line_user_id:
                line_user_id = os.getenv("LINE_DEFAULT_USER_ID")
            if not line_user_id:
                self.logger.info("LINE宛先が未設定のため送信スキップ")
                return

            client = LineMessaging()  # ENV の LINE_CHANNEL_ACCESS_TOKEN を利用
            ok = client.send_text(line_user_id, f"{subject}\n{body}")
            if ok:
                self.logger.info("LINE送信成功")
            else:
                self.logger.warning("LINE送信失敗（詳細は直前のログ参照）")
        except Exception as e:
            # 失敗してもボットは止めない
            self.logger.warning(f"LINE送信エラー: {e}")

    # （将来用）複数ユーザー配信のヘルパ
    def _send_line_multicast(self, app_user_ids: list[int], text: str) -> None:
        try:
            ids = get_line_user_ids_for_users(app_user_ids)
            if not ids:
                self.logger.info("LINE宛先が空（multicast）")
                return
            client = LineMessaging()
            ok = client.send_text_bulk(ids, text)
            if not ok:
                self.logger.warning(f"LINEマルチキャスト失敗 count={len(ids)}")
        except Exception as e:
            self.logger.warning(f"LINEマルチキャスト エラー: {e}")

    def _resolve_recipients_for_mode(self) -> list[str]:
        """
        live: 既存の notification_settings['email']['recipient']（ENV/Service由来）を使用
        sim : DBの user.email を優先。なければ従来の recipient/ENV フォールバック
        """
        recipients: list[str] = []
        is_live = self._is_live_mode()

        if is_live:
            # live は従来通り（service/ENV）だけを見る
            try:
                rcpt = (self.notification_settings or {}).get("email", {}).get("recipient")
                if rcpt:
                    recipients.append(rcpt)
            except Exception:
                pass
        else:
            # sim：DB優先
            try:
                if getattr(self, "user_id", None) is not None:
                    from db import get_user_email_and_pw
                    info = get_user_email_and_pw(self.user_id)
                    if info.get("email_enabled", True):
                        db_mail = info.get("email")
                        if db_mail:
                            recipients.append(db_mail)
            except Exception as e:
                self.logger.debug(f"SIM宛先DB取得エラー: {e}")

            # 念のためのフォールバック（従来設定やENV）
            try:
                fallback = (self.notification_settings or {}).get("email", {}).get("recipient")
                if fallback and fallback not in recipients:
                    recipients.append(fallback)
            except Exception:
                pass
            env_fallback = os.getenv("EMAIL_FALLBACK_RECIPIENT", "")
            if env_fallback and env_fallback not in recipients:
                recipients.append(env_fallback)

        return recipients

    def _resolve_smtp_password_for_mode(self, default_pw: str) -> str:
        if self._is_live_mode():
            return default_pw
        try:
            if getattr(self, "user_id", None) is not None:
                from db import get_user_email_and_pw
                info = get_user_email_and_pw(self.user_id, decrypt=True)  # ★復号オン
                pw_plain = (info or {}).get("email_password_plain")
                if pw_plain:
                    return pw_plain
                enc = (info or {}).get("email_password_encrypted")
                if enc:
                    # 復号鍵未設定などで平文が取れない場合の後方互換（従来の暫定）
                    return enc
        except Exception as e:
            self.logger.debug(f"SMTPパスワード解決エラー: {e}")
        return default_pw

    def notify_users_by_email(self, user_ids: list[int], subject: str, body: str) -> int:
        """
        user_ids の email に一括送信（enabled=trueのみ）。戻り値は送信件数。
        - LIVE/SIM問わず利用可能（SIMでは件名に [SIM] を付与すると安全）
        """
        try:
            from db import get_emails_for_users
            recipients = list(dict.fromkeys(get_emails_for_users(user_ids, only_enabled=True)))  # 重複除去
            if not recipients:
                self.logger.info("notify_users_by_email: 宛先なし")
                return 0

            email_settings = (self.notification_settings or {}).get('email', {}) or {}
            required = ['smtp_server', 'smtp_port', 'sender', 'password']
            if any(not email_settings.get(k) for k in required):
                self.logger.warning(f"メール設定不足: {required}")
                return 0

            # SIM時は件名タグ & パスワード復号
            _subject = f"[SIM] {subject}" if self._is_sim_mode() else subject
            smtp_password = self._resolve_smtp_password_for_mode(email_settings['password'])

            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            sent = 0
            with smtplib.SMTP(email_settings['smtp_server'], email_settings['smtp_port'], timeout=20) as server:
                server.starttls()
                server.login(email_settings['sender'], smtp_password)

                for rcpt in recipients:
                    msg = MIMEMultipart()
                    msg['From'] = email_settings['sender']
                    msg['To'] = rcpt
                    msg['Subject'] = str(Header(_subject, 'utf-8'))
                    msg.attach(MIMEText(body, 'plain', 'utf-8'))
                    server.sendmail(email_settings['sender'], rcpt, msg.as_string())
                    sent += 1

            self.logger.info(f"notify_users_by_email: 送信成功 {sent}件")
            return sent
        except Exception as e:
            self.logger.error(f"notify_users_by_email エラー: {e}")
            return 0
    
    def _send_email(self, subject, body):
        if not (self.notification_settings or {}).get('enabled', False):
            return
        try:
            email_settings = (self.notification_settings or {}).get('email', {}) or {}
            required_settings = ['smtp_server', 'smtp_port', 'sender', 'password']
            missing = [s for s in required_settings if not email_settings.get(s)]
            if missing:
                self.logger.warning(f"メール設定が不完全: {missing}")
                return

            recipients = self._resolve_recipients_for_mode()
            if not recipients:
                self.logger.info("宛先なしのためスキップ")
                return

            # ★SIMタグ & 復号パスワード
            _subject = f"[SIM] {subject}" if self._is_sim_mode() else subject
            smtp_password = self._resolve_smtp_password_for_mode(email_settings['password'])

            import smtplib, time
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            for attempt in range(3):
                try:
                    with smtplib.SMTP(email_settings['smtp_server'], email_settings['smtp_port'], timeout=20) as server:
                        server.starttls()
                        server.login(email_settings['sender'], smtp_password)
                        for rcpt in recipients:
                            msg = MIMEMultipart()
                            msg['From'] = email_settings['sender']
                            msg['To'] = rcpt
                            msg['Subject'] = str(Header(_subject, 'utf-8'))
                            msg.attach(MIMEText(body, 'plain', 'utf-8'))
                            server.sendmail(email_settings['sender'], rcpt, msg.as_string())
                    self.logger.info(f"メール送信成功（{len(recipients)}件）")
                    return
                except Exception as e:
                    self.logger.warning(f"SMTP送信失敗({attempt+1}/3): {e}")
                    time.sleep(2 ** attempt)

            self.logger.error("メール送信リトライ上限に到達し失敗")
        except Exception as e:
            self.logger.error(f"メール送信エラー: {e}")

    def _send_exit_detail(self, symbol: str, exit_price: float, *, timeframe: str = "5m", reason: str | None = None) -> None:
        """
        イグジット時の“利益中心の詳細通知”を作成して送るヘルパー。
        - 件名: EXIT-LONG / EXIT-SHORT を自動判定
        - 本文: 先頭に損益（円・％）、次に Entry/Exit/Size、保有時間、理由、JSTタイムスタンプ

        引数:
            symbol      : "eth_jpy" など
            exit_price  : 決済価格（約定平均が取れればそれを渡す。無ければ current_price を渡す）
            timeframe   : 表示用の時間足ラベル（既定 "5m"）
            reason      : "利益確定" / "損切り" / "長時間保有" などの理由文字列
        """
        try:
            position    = self.positions.get(symbol)        # "long" | "short" | None
            entry_price = float(self.entry_prices.get(symbol, 0) or 0)
            size        = float(self.entry_sizes.get(symbol, 0) or 0)
            entry_time  = self.entry_times.get(symbol)

            # side（件名用）
            if position in ("long", "short"):
                side = "EXIT-LONG" if position == "long" else "EXIT-SHORT"
            else:
                side = "EXIT"

            # データ不整合時はシンプル通知にフォールバック（無通知を避ける）
            if entry_price <= 0 or size <= 0 or not entry_time or exit_price is None:
                self._notify_signal(
                    side=side,
                    symbol=symbol,
                    price=float(exit_price) if exit_price is not None else None,
                    reason_tags=[reason] if reason else None,
                    notification_type="exit",
                )
                return

            # PnL 計算
            if position == "long":
                profit_jpy = (exit_price - entry_price) * size
                profit_pct = (exit_price / entry_price - 1.0) * 100.0
            else:  # short
                profit_jpy = (entry_price - exit_price) * size
                profit_pct = (entry_price / exit_price - 1.0) * 100.0

            # 保有時間（時間）
            holding_hours = (now_jst() - entry_time).total_seconds() / 3600.0

            # 本文生成（利益最上段）
            perf = ExitPerf(
                position=position,
                entry_price=entry_price,
                exit_price=float(exit_price),
                size=size,
                profit_jpy=profit_jpy,
                profit_pct=profit_pct,
                holding_hours=holding_hours,
                reason=reason,
            )
            body = compose_exit_message(symbol, timeframe, perf)  # 先頭に「損益: ○円（±x.xx%）」を出す

            # 送信（件名＋本文）: _notify_signal の 2引数版 → send_notification 経由でメール/LINEに配送
            subject = f"{side} {symbol}".upper()
            self._notify_signal(subject, body, notification_type="exit")  # 2引数版OK:contentReference[oaicite:3]{index=3}

        except Exception as e:
            # 想定外エラー時も最低限の通知は出す
            self.logger.exception(f"_send_exit_detail エラー: {e}")
            self._notify_signal(
                side="EXIT",
                symbol=symbol,
                price=float(exit_price) if exit_price is not None else None,
                reason_tags=[str(e)],
                notification_type="exit",
            )

    def _load_total_profit(self):
        """累積利益の読み込み"""
        profit_file = os.path.join(self.base_dir, 'total_profit.json')
        
        if os.path.exists(profit_file):
            try:
                with open(profit_file, 'r') as f:
                    data = json.load(f)
                    return data.get('total_profit', 0)
            except Exception as e:
                self.logger.error(f"利益データ読み込みエラー: {e}")
                return 0
        else:
            # ファイルが存在しない場合は0から始める
            return 0
    
    def _save_total_profit(self):
        """累積利益の保存"""
        profit_file = os.path.join(self.base_dir, 'total_profit.json')
        
        try:
            with open(profit_file, 'w') as f:
                json.dump({'total_profit': self.total_profit, 'updated_at': now_jst().isoformat()}, f)
        except Exception as e:
            self.logger.error(f"利益データ保存エラー: {e}")
    
    def _load_last_increase_date(self):
        """前回の増資日を読み込む"""
        increase_file = os.path.join(self.base_dir, 'last_increase.json')
        
        if os.path.exists(increase_file):
            try:
                with open(increase_file, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data.get('last_increase_date', now_jst().isoformat()))
            except Exception as e:
                self.logger.error(f"増資日データ読み込みエラー: {e}")
                return now_jst()
        else:
            # ファイルが存在しない場合は現在の日付を設定
            self._save_last_increase_date(now_jst())
            return now_jst()

    def _save_last_increase_date(self, date):
        """増資日を保存"""
        increase_file = os.path.join(self.base_dir, 'last_increase.json')
        
        try:
            with open(increase_file, 'w') as f:
                json.dump({'last_increase_date': date.isoformat()}, f)
        except Exception as e:
            self.logger.error(f"増資日データ保存エラー: {e}")

    def check_monthly_increase(self):
        """月間の運用資金増加をチェック"""
        current_date = now_jst()
        last_date = self.last_increase_date
        
        # 年月が変わったかをチェック
        if (current_date.year > last_date.year) or (current_date.year == last_date.year and current_date.month > last_date.month):
            # 月数の差を計算
            months_diff = (current_date.year - last_date.year) * 12 + (current_date.month - last_date.month)
            
            # 増資額を計算
            increase_amount = months_diff * self.monthly_increase
            
            # 資金を増やす
            self.initial_capital += increase_amount
            
            # 増資記録の更新
            self.last_increase_date = current_date
            self._save_last_increase_date(current_date)
            
            self.logger.info(f"月間増資: {increase_amount:,}円を追加しました。新しい初期資金: {self.initial_capital:,}円")
            self.send_notification("月間増資", f"{increase_amount:,}円を運用資金に追加しました。\n現在の運用資金: {self.initial_capital:,}円", "info")
    
    def save_positions(self):
        """現在のポジション情報をファイルに保存（ポジションID含む・アトミック書き込み・モード別ファイル対応）"""
        import os, json, tempfile

        src = getattr(self, "source", "real")
        filename = "positions.json" if src == "real" else f"positions_{src}.json"
        positions_file = os.path.join(self.base_dir, filename)

        # 保存用データを組み立て（ポジションがあるシンボルのみ）
        positions_data = {}
        for symbol in self.symbols:
            pos = self.positions.get(symbol)
            if pos is None:
                continue
            positions_data[symbol] = {
                "position": pos,
                "entry_price": self.entry_prices.get(symbol),
                "entry_time": self.entry_times[symbol].isoformat() if self.entry_times.get(symbol) else None,
                "entry_size": self.entry_sizes.get(symbol, 0.0),
                "position_id": self.position_ids.get(symbol),
            }

        # ディレクトリ確保
        os.makedirs(self.base_dir, exist_ok=True)

        # アトミック書き込み
        fd, tmp_path = tempfile.mkstemp(prefix=".positions_", suffix=".tmp", dir=self.base_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(positions_data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, positions_file)
            self.logger.info("ポジション情報を保存しました: %s", positions_file)
        except Exception as e:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            self.logger.error("ポジション情報の保存エラー（%s）: %s", positions_file, e)

    def load_positions(self):
        """保存されたポジション情報を読み込む（ポジションID含む・モード別ファイル対応）"""
        import os, json
        from datetime import datetime

        # モードに応じたパス（data_dir/base_dir の揺れを避ける）
        src = getattr(self, "source", "real")
        positions_file = self._positions_filepath(src)
        self.logger.info("[positions] source=%s file=%s", src, positions_file)

        # 初期化（存在しないキーへの備え）
        for sym in self.symbols:
            self.positions.setdefault(sym, None)
            self.entry_prices.setdefault(sym, None)
            self.entry_sizes.setdefault(sym, 0.0)
            self.position_ids.setdefault(sym, None)
            self.entry_times.setdefault(sym, None)

        # 1) モード別ファイルがあればそのまま読む
        if os.path.exists(positions_file):
            try:
                with open(positions_file, "r", encoding="utf-8") as f:
                    positions_data = json.load(f)

                for symbol, data in positions_data.items():
                    if symbol not in self.symbols:
                        continue
                    self.positions[symbol] = data.get("position")
                    self.entry_prices[symbol] = data.get("entry_price")
                    self.entry_sizes[symbol] = data.get("entry_size", 0.0)
                    self.position_ids[symbol] = data.get("position_id")
                    ts = data.get("entry_time")
                    if ts:
                        t = datetime.fromisoformat(ts)
                        # tzなし→JST付与、別TZ→JSTへ変換
                        if t.tzinfo is None:
                            t = t.replace(tzinfo=JST)
                        else:
                            t = t.astimezone(JST)
                        self.entry_times[symbol] = t
                    else:
                        self.entry_times[symbol] = None

                self.logger.info("保存されたポジション情報を読み込みました: %s", positions_file)
                self.print_positions_info()
                return
            except Exception as e:
                self.logger.error("ポジション情報の読み込みエラー（%s）: %s", positions_file, e)

        # 2) なければ real の positions.json を初回コピー元にする（SIM/BTの分岐スタート）
        if src != "real":
            real_file = self._positions_filepath("real")
            if os.path.exists(real_file):
                try:
                    with open(real_file, "r", encoding="utf-8") as rf:
                        positions_data = json.load(rf)

                    for symbol, data in positions_data.items():
                        if symbol not in self.symbols:
                            continue
                        self.positions[symbol] = data.get("position")
                        self.entry_prices[symbol] = data.get("entry_price")
                        self.entry_sizes[symbol] = data.get("entry_size", 0.0)
                        self.position_ids[symbol] = data.get("position_id")
                        ts = data.get("entry_time")
                        if ts:
                            t = datetime.fromisoformat(ts)
                            # tzなし→JST付与、別TZ→JSTへ変換
                            if t.tzinfo is None:
                                t = t.replace(tzinfo=JST)
                            else:
                                t = t.astimezone(JST)
                            self.entry_times[symbol] = t
                        else:
                            self.entry_times[symbol] = None

                    # 自分のモードファイル名で保存して以後は分離運用
                    self.save_positions()  # save_positions 内も _positions_filepath を使うことを推奨
                    self.logger.info("初回作成: %s（ソース: %s）", positions_file, real_file)
                    self.print_positions_info()
                    return
                except Exception as e:
                    self.logger.warning("初回コピー元（%s）読み込みに失敗: %s", real_file, e)

        # 3) どちらも無ければ空で作成
        try:
            self.save_positions()
            self.logger.info("positions ファイルが無かったため新規作成: %s", positions_file)
        except Exception as e:
            self.logger.error("ポジション情報の初期保存エラー（%s）: %s", positions_file, e)
    
    def print_positions_info(self):
        """現在のポジション情報を表示"""
        self.logger.info("=== 現在のポジション情報 ===")
        has_positions = False
        
        for symbol in self.symbols:
            if self.positions[symbol] is not None:
                has_positions = True
                position_type = "買い(ロング)" if self.positions[symbol] == 'long' else "売り(ショート)"
                self.logger.info(f"通貨: {symbol}")
                self.logger.info(f"  ポジション: {position_type}")
                self.logger.info(f"  エントリー価格: {self.entry_prices[symbol]}")
                self.logger.info(f"  エントリー時間: {self.entry_times[symbol]}")
                self.logger.info(f"  取引量: {self.entry_sizes[symbol]:.6f}")
                
                # 保有時間を計算
                if self.entry_times[symbol]:
                    holding_time = now_jst() - self.entry_times[symbol]
                    hours = holding_time.total_seconds() / 3600
                    self.logger.info(f"  保有時間: {hours:.1f}時間")
                    
                # 現在の利益を表示（概算）
                try:
                    current_price = self.get_current_price(symbol)
                    if current_price > 0:
                        if self.positions[symbol] == 'long':
                            profit_pct = (current_price / self.entry_prices[symbol] - 1) * 100
                            profit_amount = (current_price - self.entry_prices[symbol]) * self.entry_sizes[symbol]
                        else:  # 'short'
                            profit_pct = (self.entry_prices[symbol] / current_price - 1) * 100
                            profit_amount = (self.entry_prices[symbol] - current_price) * self.entry_sizes[symbol]
                            
                        if profit_pct > 0:
                            status = "利益中"
                        else:
                            status = "損失中"
                            
                        self.logger.info(f"  現在の状態: {status} {profit_pct:.2f}% ({profit_amount:.2f}円)")
                except Exception as e:
                    self.logger.debug(f"現在価格取得エラー: {e}")
        
        if not has_positions:
            self.logger.info("現在ポジションはありません")

    def get_current_price(self, symbol):
        try:
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper())
            url = f'https://api.coin.z.com/public/v1/ticker?symbol={gmo_symbol}'
            headers = {'User-Agent': 'Mozilla/5.0'}

            # レート制限ウェイト
            elapsed = time.time() - self.last_api_call
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)

            # 最大3回の明示再試行（HTTP_SESSION側のRetryとは別にDNS対策）
            last_exc = None
            for attempt in range(3):
                try:
                    with HTTP_SESSION.get(url, headers=headers, timeout=10) as response:
                        self.last_api_call = time.time()
                        data = response.json()
                        if data.get('status') == 0 and 'data' in data:
                            ticker_data = data['data']
                            if isinstance(ticker_data, list):
                                for item in ticker_data:
                                    if item.get('symbol') == gmo_symbol:
                                        last_price = float(item.get('last', 0))
                                        self.logger.info(f"{symbol} 現在価格: {last_price}")
                                        return last_price
                                self.logger.warning(f"{symbol}（{gmo_symbol}）の価格データが見つかりません")
                                return 0.0
                            else:
                                return float(ticker_data.get('last', 0))
                        else:
                            err = data.get('messages', [{"message_string": "不明なエラー"}])[0].get("message_string", "不明なエラー") if data.get('messages') else "不明なエラー"
                            self.logger.error(f"GMOコイン価格取得エラー: {err}")
                            return 0.0
                except requests.exceptions.RequestException as e:
                    last_exc = e
                    if _is_dns_temp_failure(e):
                        time.sleep(1.2 * (attempt + 1))
                        continue
                    time.sleep(0.6 * (attempt + 1))
                    continue

            if last_exc:
                self.logger.error(f"価格取得APIリクエストエラー: {last_exc}")
            return 0.0

        except ValueError as e:
            self.logger.error(f"価格取得JSONパースエラー: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"価格取得未知のエラー: {e}")
            return 0.0

    
    def create_backup(self):
        """データのバックアップを作成"""
        try:
            # バックアップ時間を更新
            self.last_backup_time = time.time()
            
            # バックアップディレクトリ名
            backup_time = now_jst().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.backup_dir, f"backup_{backup_time}")
            os.makedirs(backup_dir, exist_ok=True)
            
            # キャッシュデータのバックアップ
            cache_backup = os.path.join(backup_dir, 'data_cache')
            if os.path.exists(self.cache_dir):
                shutil.copytree(self.cache_dir, cache_backup)
            
            # 利益データのバックアップ
            profit_file = os.path.join(self.base_dir, 'total_profit.json')
            if os.path.exists(profit_file):
                shutil.copy2(profit_file, backup_dir)
            
            # ポジションデータのバックアップ
            positions_file = os.path.join(self.base_dir, 'positions.json')
            if os.path.exists(positions_file):
                shutil.copy2(positions_file, backup_dir)
            
            # 増資データのバックアップ
            increase_file = os.path.join(self.base_dir, 'last_increase.json')
            if os.path.exists(increase_file):
                shutil.copy2(increase_file, backup_dir)
            
            # ログファイルのバックアップ
            logs_backup = os.path.join(backup_dir, 'logs')
            os.makedirs(logs_backup, exist_ok=True)
            for log_file in os.listdir(self.log_dir):
                if log_file.endswith('.log'):
                    shutil.copy2(os.path.join(self.log_dir, log_file), logs_backup)
            
            # 古いバックアップを削除（10個以上あれば）
            all_backups = sorted([d for d in os.listdir(self.backup_dir) if d.startswith('backup_')])
            if len(all_backups) > 10:
                for old_backup in all_backups[:-10]:
                    old_path = os.path.join(self.backup_dir, old_backup)
                    if os.path.isdir(old_path):
                        shutil.rmtree(old_path)
            
            self.logger.info(f"バックアップ作成完了: {backup_dir}")
            return True
        except Exception as e:
            self.logger.error(f"バックアップエラー: {e}")
            return False
    
    def check_backup_needed(self):
        """バックアップが必要かどうかをチェック"""
        if time.time() - self.last_backup_time > self.backup_interval:
            return self.create_backup()
        return False
        
    def verify_positions(self):
        """GMOコインの信用取引建玉を使用してポジションを検証する"""
        # SIM（paper）では取引所照合はしないが、DBのオープンポジションを反映して JSON を更新する
        if not self.exchange_settings_gmo.get("live_trade", False):
            self.logger.info("Paper mode: positions_* をロードし、DBのオープンポジションで上書きします")
            try:
                # 1) まず現在の positions_sim.json を読み込み（存在すれば）
                self.load_positions()
            except Exception:
                self.logger.exception("Paper mode での positions ロードに失敗")

            try:
                # 2) DBから source='sim' のオープンポジションを取得（内部で live/paper を見て src を切替）
                resp = self._get_open_positions_from_db(None)  # 全シンボル対象
                if resp.get("status") == 0:
                    lst = resp.get("data", {}).get("list", [])
                    # いったん全通貨をクリア
                    prev_entry_times = dict(self.entry_times)
                    for sym in self.symbols:
                        self.positions[sym] = None
                        self.entry_prices[sym] = 0.0
                        self.entry_sizes[sym] = 0.0
                        self.position_ids[sym] = None
                        # entry_times はここでクリアしない（openedAt で上書き or 既存温存）

                    # GMO表記 → bot表記へ逆引きして反映（逆引きマップを用意）
                    reverse_map = {v: k for k, v in self.symbol_mapping.items()}
                    # _get_open_positions_from_db は "symbol" に GMO表記（例: "XRPJPY"）を返す実装
                    # なので symbol_mapping を逆引きして bot表記（例: "xrp_jpy"）に戻す
                    for p in lst:
                        gmo_symbol = p.get("symbol")
                        bot_symbol = reverse_map.get(gmo_symbol)
                        if not bot_symbol or bot_symbol not in self.symbols:
                            # フォールバック（"XRPJPY" → "xrp_jpy" 的に近似復元）
                            try:
                                s = (gmo_symbol or "").upper()
                                if s.endswith("JPY"):
                                    bot_symbol = s[:-3].lower() + "_jpy"
                                else:
                                    bot_symbol = s.lower()
                            except Exception:
                                continue
                            if bot_symbol not in self.symbols:
                                continue

                        side = "long" if p.get("side") == "BUY" else "short"
                        self.positions[bot_symbol]     = side
                        self.entry_prices[bot_symbol]  = float(p.get("price") or 0.0)
                        self.entry_sizes[bot_symbol]   = float(p.get("size") or 0.0)
                        self.position_ids[bot_symbol]  = p.get("positionId")
                        # entry_time を openedAt から復元。無ければ既存値を温存。
                        opened_at = p.get("openedAt")
                        if opened_at:
                            import pandas as pd
                            ts = pd.to_datetime(opened_at, utc=False)
                            # pandas.Timestamp → JST aware に正規化
                            if getattr(ts, "tz", None) is None:
                                ts = ts.tz_localize("Asia/Tokyo")
                            else:
                                ts = ts.tz_convert("Asia/Tokyo")
                            self.entry_times[bot_symbol] = ts.to_pydatetime()
                        else:
                            self.entry_times[bot_symbol] = prev_entry_times.get(bot_symbol)

                    # 3) 反映済みのメモリ状態を SIM 用ファイルに保存
                    self.save_positions()
                    self.logger.info("Paper mode: DBのオープンポジションを positions_sim.json へ反映しました")
            except Exception as e:
                self.logger.warning(f"Paper mode: DB→JSON反映に失敗: {e}")
            return False

        # ローカルポジション情報のバックアップ
        local_positions_backup = {
            'positions': self.positions.copy(),
            'entry_prices': self.entry_prices.copy(),
            'entry_times': {k: v for k, v in self.entry_times.items()},
            'entry_sizes': self.entry_sizes.copy()
        }

        inconsistencies = 0
        resolved = 0

        for symbol in self.symbols:
            try:
                time.sleep(0.5)  # APIレート制限対策

                coin = symbol.split('_')[0]
                position_details = self.get_position_details(coin)

                has_actual_position = False
                actual_side = None
                actual_size = 0.0

                if position_details and position_details["positions"]:
                    net_size = position_details["net_size"]
                    if abs(net_size) >= self.min_order_sizes.get(symbol, 0.001):
                        has_actual_position = True
                        actual_side = "long" if net_size > 0 else "short"
                        actual_size = abs(net_size)

                self.logger.info(
                    f"{symbol}: ボット状態={self.positions[symbol]}, "
                    f"実際の建玉={actual_side if has_actual_position else 'なし'}"
                )

                if self.positions[symbol] is not None:  # ボットではポジションあり
                    if not has_actual_position:
                        self.logger.warning(f"{symbol}: ボットではポジション有りだが、実際には建玉が見つかりませんでした。")
                        self.positions[symbol] = None
                        self.entry_prices[symbol] = 0
                        self.entry_times[symbol] = None
                        self.entry_sizes[symbol] = 0
                        inconsistencies += 1
                        resolved += 1
                    else:
                        if self.positions[symbol] != actual_side:
                            self.logger.warning(
                                f"{symbol}: ポジションタイプの不一致。記録: {self.positions[symbol]}, 実際: {actual_side}"
                            )
                            self.positions[symbol] = actual_side
                            inconsistencies += 1
                            resolved += 1

                        # サイズの誤差チェック
                        size_diff_pct = abs(self.entry_sizes[symbol] - actual_size) / max(self.entry_sizes[symbol], actual_size) * 100
                        if size_diff_pct > 5:
                            self.logger.warning(
                                f"{symbol}: 記録上のポジションサイズ({self.entry_sizes[symbol]:.6f})と"
                                f"実際のサイズ({actual_size:.6f})に{size_diff_pct:.1f}%の差があります。更新します。"
                            )
                            self.entry_sizes[symbol] = actual_size
                            inconsistencies += 1
                            resolved += 1

                        # 価格は常に建玉から更新
                        side_key = "BUY" if actual_side == "long" else "SELL"
                        prices = [
                            pos["price"] for pos in position_details["positions"]
                            if pos["side"] == side_key
                        ]
                        if prices:
                            avg_price = sum(prices) / len(prices)
                            self.logger.info(
                                f"{symbol}: 建玉平均価格を更新します。旧値: {self.entry_prices[symbol]:.2f} → 新値: {avg_price:.2f}"
                            )
                            self.entry_prices[symbol] = avg_price
                            inconsistencies += 1
                            resolved += 1



                elif has_actual_position:  # ボットではポジションなし、実際にはポジションあり
                    self.logger.warning(
                        f"{symbol}: ボットではポジションなしだが、実際には{actual_size:.6f}の{actual_side}建玉が見つかりました。"
                    )

                    # 建玉から取得した平均価格を使用（該当サイドのみ）
                    avg_price = 0.0
                    side_key = "BUY" if actual_side == "long" else "SELL"
                    prices = [
                        pos["price"] for pos in position_details["positions"]
                        if pos["side"] == side_key
                    ]
                    if prices:
                        avg_price = sum(prices) / len(prices)

                    self.positions[symbol] = actual_side
                    self.entry_prices[symbol] = avg_price
                    self.entry_times[symbol] = now_jst() - timedelta(hours=1)  # 1時間前からの保有と仮定
                    self.entry_sizes[symbol] = actual_size

                    inconsistencies += 1
                    resolved += 1
                    self.logger.info(f"{symbol}: ポジション情報を更新しました（推定値, 価格={avg_price}）。")

            except Exception as e:
                self.logger.error(f"{symbol}のポジション検証中にエラー: {e}", exc_info=True)

        if inconsistencies > 0:
            self.save_positions()
            self.logger.info(f"ポジション情報の検証完了: {inconsistencies}件の不整合を検出、{resolved}件を解決しました")
        else:
            self.logger.info("ポジション情報の検証完了: 不整合はありませんでした")

            
    def get_balance(self, coin):
        """GMOコインから信用取引の建玉情報を取得
        
        Parameters:
        coin (str): 通貨名（例: 'ltc', 'xrp'）
        
        Returns:
        float: 該当通貨の建玉サイズの絶対値
        """
        try:
            # GMO APIが利用可能か確認
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません")
                return 0.0
            
            symbol = f"{coin}_jpy"
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
            # 信用取引の建玉情報を取得（symbolパラメータは使わない）
            margin_response = self._get_margin_positions_safe(gmo_symbol)
            # ステータスコードを確認
            status = margin_response.get("status")
            
            if status == 0:
                # データ形式を確認
                data = margin_response.get("data", {})
                
                # データが辞書の場合はlistを取得、リストの場合はそのまま使用
                if isinstance(data, dict):
                    positions = data.get("list", [])
                else:
                    positions = data
                
                # 該当通貨の建玉を集計
                total_position_size = 0.0
                
                # 通貨ペアに変換
                symbol = f"{coin}_jpy"
                gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
                
                for position in positions:
                    if position.get("symbol") == gmo_symbol:
                        size = float(position.get("size", 0))
                        side = position.get("side")
                        
                        # 買い建玉はプラス、売り建玉はマイナスとして扱う
                        if side == "BUY":
                            total_position_size += size
                        else:  # SELL
                            total_position_size -= size
                
                # 絶対値を返す
                result = abs(total_position_size)
                return result
            else:
                error_messages = margin_response.get("messages", [])
                error_msg = error_messages[0].get("message_string", "不明なエラー") if error_messages else "不明なエラー"
                self.logger.error(f"建玉情報取得エラー: {error_msg}")
                return 0.0
            
        except Exception as e:
            self.logger.error(f"建玉取得エラー: {str(e)}", exc_info=True)
            return 0.0

    def get_position_details(self, coin):
        """より詳細な建玉情報を取得する（オプション）
        
        Parameters:
        coin (str): 通貨名（例: 'ltc', 'xrp'）
        
        Returns:
        dict: 建玉情報の詳細、またはNone
        """
        if not self.exchange_settings_gmo.get("live_trade", False):
            symbol = f"{coin.lower()}_jpy"
            side = self.positions.get(symbol)
            size = float(self.entry_sizes.get(symbol) or 0.0)
            gmo_symbol = self.symbol_mapping.get(symbol, coin.upper() + "JPY")

            if not side or size <= 0:
                return {"positions": [], "net_size": 0.0}

            pos = {
                "symbol": gmo_symbol,
                "side": ("BUY" if side == "long" else "SELL"),
                "positionId": self.position_ids.get(symbol),
                "size": size,
                "price": float(self.entry_prices.get(symbol) or 0.0),
                "openedAt": (self.entry_times.get(symbol).isoformat()
                            if self.entry_times.get(symbol) else None),
            }
            return {
                "positions": [pos],
                "net_size": (size if side == "long" else -size),
            }

        try:
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません")
                return None
            
            # 通貨ペアに変換
            symbol = f"{coin}_jpy"
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
            
            # 建玉情報を取得
            margin_response = self._get_margin_positions_safe(gmo_symbol)
            
            if margin_response.get("status") == 0:
                # データ形式を確認
                data = margin_response.get("data", {})
                
                # データが辞書の場合はlistを取得、リストの場合はそのまま使用
                if isinstance(data, dict):
                    positions = data.get("list", [])
                else:
                    positions = data
                
                position_details = {
                    "long_size": 0.0,
                    "short_size": 0.0,
                    "net_size": 0.0,
                    "positions": []
                }

                # 取得した全建玉から該当通貨のものをフィルタリング
                for position in positions:
                    if position.get("symbol") == gmo_symbol:
                        size = float(position.get("size", 0))
                        side = position.get("side")
                        price = float(position.get("price", 0))
                        position_id = position.get("positionId")
                        timestamp = position.get("timestamp")
                        
                        # 価格 × サイズ = ポジション評価額
                        position_value = price * size
                        
                        position_info = {
                            "positionId": position_id, 
                            "side": side,
                            "size": size,
                            "price": price,
                            "value": position_value,
                            "timestamp": timestamp,
                            "symbol": gmo_symbol  # symbolキーを明示的に追加
                        }
                        
                        if side == "BUY":
                            position_details["long_size"] += size
                        else:  # SELL
                            position_details["short_size"] += size
                        
                        position_details["positions"].append(position_info)
                
                # ネットポジション（買い - 売り）
                position_details["net_size"] = position_details["long_size"] - position_details["short_size"]
                
                # デバッグログ
                self.logger.info(f"{gmo_symbol}の建玉詳細: "
                            f"ロング={position_details['long_size']}, "
                            f"ショート={position_details['short_size']}, "
                            f"ネット={position_details['net_size']}")
                
                return position_details
            else:
                error_msg = margin_response.get("messages", [{}])[0].get("message_string", "不明なエラー")
                self.logger.error(f"建玉情報取得エラー: {error_msg}")
                return None
            
        except Exception as e:
            self.logger.error(f"詳細建玉情報取得エラー: {str(e)}", exc_info=True)
            return None

    def adjust_order_size(self, symbol, base_size):
        """取引所の最小注文量を考慮して注文サイズを調整する
        
        Parameters:
        symbol (str): 通貨ペア
        base_size (float): 基本注文サイズ
        
        Returns:
        float: 調整後の注文サイズ
        """
        # 通貨ペアごとの最小注文量を取得
        min_size = self.min_order_sizes.get(symbol, 0.001)  # デフォルトは0.001
        
        # 基本サイズが最小注文量より小さい場合は調整
        if base_size < min_size:
            self.logger.warning(f"{symbol}の注文サイズ({base_size})が最小注文量({min_size})より小さいため、調整します")
            return min_size
        
        # 通貨ごとの処理
        if symbol == "xrp_jpy":
            base_size = round(base_size / 10) * 10
            if base_size < 10:
                base_size = 10
        elif symbol == "sol_jpy":
            base_size = round(base_size, 1)
        elif symbol == "eth_jpy":
            base_size = round(base_size, 2)
        elif symbol == "ltc_jpy":
            base_size = int(base_size)
        elif symbol == "doge_jpy":
            base_size = round(base_size / 10) * 10
            if base_size < 10:
                base_size = 10
        elif symbol == "bcc_jpy":  # 追加
            base_size = round(base_size, 1)  # BCHは小数点以下2桁
        elif symbol == "ada_jpy":  # 追加
            base_size = round(base_size / 10) * 10  # 100単位に丸める
            if base_size < 10:
                base_size = 10
        
        return base_size

    def check_sufficient_funds(self, symbol, order_type, size, price=None, balance=None):
        """注文を出す前に十分な資金があるか確認する
        
        Parameters:
        symbol (str): 通貨ペア
        order_type (str): 注文タイプ ('buy' or 'sell')
        size (float): 注文サイズ
        price (float, optional): 注文価格 (指値の場合)。Noneの場合は現在価格を使用
        
        Returns:
        bool: 十分な資金がある場合はTrue
        """
        try:
            # 通貨ペアの分解
            base_coin, quote_coin = symbol.split('_')
            
            # 現在価格の取得（価格が指定されていない場合）
            if price is None:
                current_price = self.get_current_price(symbol)
                if current_price <= 0:
                    self.logger.error(f"{symbol}の現在価格を取得できませんでした")
                    return False
                price = current_price
            
            # 注文に必要な資金を計算
            if order_type == 'buy':
                # 買い注文の場合、quote_coin（JPYなど）が必要
                required_amount = price * size * 1.005  # 手数料を考慮して5%増し
                
                # バックテスト時は指定された残高を使用
                if balance is not None:
                    available_balance = balance
                else:
                    available_balance = self.get_total_balance()*2
                
                if available_balance < required_amount:
                    self.logger.warning(f"買い注文に必要な資金が不足しています: "
                                    f"必要量: {required_amount:.2f} {quote_coin}, 利用可能: {available_balance:.2f} {quote_coin}")
                    return False
                    
            elif order_type == 'sell':
                # 売り注文の場合、base_coin（BTC,ETHなど）が必要
                required_amount = size * 1.005  # 手数料を考慮
                
                # バックテスト時は指定された残高を使用
                if balance is not None:
                    available_balance = balance
                else:
                    available_balance = self.get_total_balance()*2
                
                if available_balance < required_amount:
                    self.logger.warning(f"売り注文に必要な資金が不足しています: "
                                    f"必要量: {required_amount:.6f} {base_coin}, 利用可能: {available_balance:.6f} {base_coin}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"資金確認エラー: {e}")
            return False
        
    def place_order(self, symbol, order_type, size, margin=True):
        """GMOコインでの信用取引注文を実行する（paper時はAPIを呼ばず即約定＆DB upsert）。"""
        import uuid, time

        # ---- 共通：サイズ正規化（GMOの最小単位に揃える） -----------------------
        def _normalize_size(sym: str, sz: float) -> float:
            # ※必要に応じて調整。未指定は四捨五入なしで返す
            if sym == "xrp_jpy":
                # 10通貨刻み、最小10
                v = round(sz / 10) * 10
                return max(v, 10)
            elif sym == "eth_jpy":
                return round(sz, 2)
            elif sym == "ltc_jpy":
                return int(sz)
            elif sym == "doge_jpy":
                v = round(sz / 10) * 10
                return max(v, 10)
            elif sym == "sol_jpy":
                return round(sz, 1)
            elif sym == "bcc_jpy":
                return round(sz, 1)  # BCH(BCC)は小数1桁
            else:
                return float(sz)

        is_live = bool(self.exchange_settings_gmo.get("live_trade"))
        norm_size = _normalize_size(symbol, float(size))
        side = "BUY" if order_type == "buy" else "SELL"
        gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace("_", ""))

        # ---- PAPER（live_trade=False）：APIを呼ばず即約定 ----------------------
        if not is_live:
            try:
                sim_order_id = f"SIM-{uuid.uuid4().hex[:12]}"
                sim_position_id = f"SIMPOS-{uuid.uuid4().hex[:12]}"

                # 平均約定価格は現在価格で近似（必要ならslippage/feeロジックを後付け）
                try:
                    avg_entry = float(self.get_current_price(symbol))
                except Exception:
                    avg_entry = float(self.entry_prices.get(symbol) or 0.0)

                executed_size_pos = float(norm_size)
                self.logger.info(f"[SIM ORDER] {symbol} {side} {executed_size_pos} -> oid={sim_order_id}")

                # 位置情報を内部状態に反映
                self.position_ids[symbol] = sim_position_id
                if avg_entry > 0:
                    self.entry_prices[symbol] = avg_entry
                self.entry_sizes[symbol] = executed_size_pos

                # DB upsert（source は sim を強制）
                default_source = "sim"

                def _to_uuid_or_none(v):
                    if not v:
                        return None
                    try:
                        import uuid as _u
                        return str(_u.UUID(str(v)))
                    except Exception:
                        return None

                strategy_uuid = _to_uuid_or_none(getattr(self, "strategy_id", None))

                self.logger.info(
                    "[DBHOOK][SIM] upsert_position: pos_id=%s size=%s avg=%s side=%s src=%s",
                    sim_position_id, executed_size_pos, avg_entry, ("long" if side == "BUY" else "short"), default_source
                )
                try:
                    upsert_position(
                        position_id=str(sim_position_id),
                        symbol=symbol,
                        side=("long" if side == "BUY" else "short"),
                        size=float(executed_size_pos),
                        avg_entry_price=float(avg_entry) if avg_entry else 0.0,
                        opened_at=utcnow(),
                        updated_at=utcnow(),
                        raw={"order_response": {"sim": True, "order_id": sim_order_id}},
                        strategy_id=strategy_uuid,
                        source=default_source,
                        user_id=getattr(self, "user_id", None),
                    )
                except Exception as e:
                    # SIMは止めたくないのでログだけ
                    self.logger.error(f"[DBHOOK][SIM] upsert_position失敗: {e}", exc_info=True)

                return {
                    "success": True,
                    "order_id": sim_order_id,
                    "executed_size": executed_size_pos,
                    "position_id": sim_position_id,
                }
            except Exception as e:
                self.logger.error(f"[SIM ORDER] 例外: {e}", exc_info=True)
                return {"success": False, "error": str(e), "executed_size": 0}  
        try:
            # GMO API が利用可能か確認
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません")
                return {'success': False, 'error': "GMO API not initialized", 'executed_size': 0}
            
            # 通貨ペアをGMO形式に変換
            gmo_symbol = gmo_symbol
            
            # 最小注文量の調整（正規化値を使用）
            size = norm_size            
            
            # 注文前の既存ポジションを記録（後で新規ポジションを判別するため）
            existing_positions = set()
            try:
                positions_response = self._get_margin_positions_safe(gmo_symbol)
                if positions_response.get("status") == 0:
                    positions_data = positions_response.get("data", {})
                    positions = positions_data.get("list", []) if isinstance(positions_data, dict) else positions_data
                    
                    for pos in positions:
                        if pos.get("symbol") == gmo_symbol:
                            pid = pos.get("positionId")
                            if pid is not None:
                                existing_positions.add(pid)

            except Exception as e:
                self.logger.warning(f"既存ポジション取得エラー: {e}")

            # --- 銘柄別サイズフォーマッタ（小数桁の制約に合わせる） ---
            from decimal import Decimal, ROUND_DOWN
            def _format_size(sym: str, sz: float) -> str:
                sym_l = sym.lower()
                # 受け付け小数桁のマップ（必要に応じて拡張）
                precision = {
                    "xrp_jpy": 0, "doge_jpy": 0, "ltc_jpy": 0, "ada_jpy": 0,
                    "eth_jpy": 2, "sol_jpy": 1, "bcc_jpy": 1, "bch_jpy": 1,
                }.get(sym_l, 3)
                q = Decimal(1).scaleb(-precision)
                val = Decimal(str(sz)).quantize(q, rounding=ROUND_DOWN)
                s = format(val, "f")
                if precision == 0 and "." in s:
                    s = s.split(".")[0]
                return s

            size_str = _format_size(symbol, size)

            # 新規注文のみを扱う（ログも送信値もフォーマット後を表示）
            self.logger.info(f"GMOコイン信用取引新規注文: {gmo_symbol} {side} {size_str}")

            # 新規注文データ（サイズは必ず文字列）
            order_data = {
                "symbol": gmo_symbol,
                "side": side,
                "executionType": "MARKET",
                "size": size_str,
                "settlePosition": "OPEN"  # 常に新規
            }
            
            # 新規注文の場合はmarginTradeTypeを指定
            order_data["marginTradeType"] = "SHORT" if side == "SELL" else "LONG"
            
            response = self.gmo_api._request("POST", "/v1/order", data=order_data)
            
            if response.get("status") == 0:
                # GMOの戻りが {"data": "...orderId..."} or {"data": {"orderId": "..."}}
                data = response.get("data")
                if isinstance(data, dict):
                    order_id = str(data.get("orderId") or data.get("order_id") or data.get("id") or "")
                else:
                    order_id = str(data)
                self.logger.info(f"注文成功: {gmo_symbol} {side} {size}, 注文ID: {order_id}")

                # --- 直後のポジション同期で「建玉の建値」と「実約定サイズ」を取得する ---
                position_id = None
                avg_entry = 0.0              # 取得できたら更新
                executed_size_pos = 0.0      # 建玉のサイズ（実約定合計）
                max_retries = 5
                retry_interval = 2  # 秒

                for retry in range(max_retries):
                    time.sleep(retry_interval)
                    try:
                        positions_response = self._get_margin_positions_safe(gmo_symbol)
                        if positions_response.get("status") == 0:
                            positions_data = positions_response.get("data", {})
                            positions = positions_data.get("list", []) if isinstance(positions_data, dict) else positions_data

                            # 既存ポジション集合にない「今回新規のポジション」を探索
                            for pos in positions:
                                if (
                                    pos.get("symbol") == gmo_symbol and
                                    pos.get("side") == side and
                                    pos.get("positionId") not in existing_positions
                                ):
                                    position_id = pos.get("positionId")
                                    # 建値（平均建玉価格）
                                    try:
                                        avg_entry = float(pos.get("price") or 0)
                                    except Exception:
                                        avg_entry = 0.0
                                    # 建玉サイズ（実約定合計サイズ）
                                    try:
                                        executed_size_pos = float(pos.get("size") or 0)
                                    except Exception:
                                        executed_size_pos = 0.0

                                    self.logger.info(
                                        f"新規ポジション取得: id={position_id}, price={avg_entry}, size={executed_size_pos} "
                                        f"(試行 {retry + 1}/{max_retries})"
                                    )
                                    break

                            if position_id:
                                # 見つかったら即ループ終了
                                break

                    except Exception as e:
                        self.logger.warning(f"ポジション同期試行{retry + 1}失敗: {e}")

                    if retry < max_retries - 1:
                        self.logger.info(f"ポジション未同期。再試行... ({retry + 2}/{max_retries})")

                # --- 取得できた情報を保存（建値・サイズ・ポジションID） ---
                if position_id:
                    self.position_ids[symbol] = position_id
                    self.logger.info(f"ポジションID保存成功: {symbol} = {position_id}")
                else:
                    self.logger.warning(f"ポジションIDを取得できませんでした: {symbol}（後で verify_positions で同期）")

                # --- DB: ポジションを upsert ---
                if position_id and executed_size_pos > 0:
                    # source は NOT NULL 想定：live は "real"
                    default_source = "real"
                    # --- UUID検証（不正ならNoneに落とす or 必要ならuuid4()を発行） ---
                    def _to_uuid_or_none(v):
                        if not v:
                            return None
                        try:
                            import uuid
                            return str(uuid.UUID(str(v)))
                        except Exception:
                            return None

                    strategy_uuid = _to_uuid_or_none(getattr(self, "strategy_id", None))
                    # 必須なら: strategy_uuid = strategy_uuid or str(uuid.uuid4())

                    # --- DBUPSERT：握りつぶさずFail-Fastに変更（原因追跡のため） ---
                    self.logger.info(
                        "[DBHOOK] upsert_position: pos_id=%s size=%s avg=%s strat=%s src=%s user=%s",
                        position_id, executed_size_pos, avg_entry, strategy_uuid,
                        default_source, getattr(self, "user_id", None)
                    )
                    upsert_position(
                        position_id=str(position_id),               # DB: varchar(80)
                        symbol=symbol,                               # DB: varchar(32)
                        side=("long" if side == "BUY" else "short"), # DB: varchar(8) 想定
                        size=float(executed_size_pos),               # DB: numeric(24,8)
                        avg_entry_price=float(avg_entry) if avg_entry else 0.0,
                        opened_at=utcnow(),                          # DB: timestamptz
                        updated_at=utcnow(),                          # DB: timestamptz
                        raw={"order_response": response},
                        strategy_id=strategy_uuid,
                        source=default_source,              # ← real を明示
                        user_id=getattr(self, "user_id", None)        # DB: int8
                    )

                # 建玉から建値が取れたら entry_prices を更新
                if avg_entry > 0:
                    self.entry_prices[symbol] = avg_entry
                    self.logger.info(f"[ENTRY] 建玉の建値で entry_price 更新: {symbol} = {avg_entry}")

                # 建玉から実約定サイズが取れたら entry_sizes を上書き（部分約定対応）
                if executed_size_pos > 0:
                    self.entry_sizes[symbol] = executed_size_pos
                    self.logger.info(f"[ENTRY] 建玉サイズで entry_size 更新: {symbol} = {executed_size_pos}")

                # --- フォールバック: 建玉から price が取れない場合は注文照会→約定履歴から平均約定価格を試みる ---
                if avg_entry <= 0:
                    try:
                        # 1) 注文照会で averagePrice 相当が取れればそれを使う
                        od = self.gmo_api._request("GET", "/v1/orders", params={"orderId": str(order_id)})
                        if od.get("status") == 0:
                            lst = od.get("data", {}).get("list", [])
                            if lst:
                                od0 = lst[0]
                                ap = float(od0.get("averagePrice") or 0)
                                ex_sz = float(od0.get("executedSize") or 0)
                                if ap > 0:
                                    avg_entry = ap
                                    self.entry_prices[symbol] = avg_entry
                                    self.logger.info(f"[ENTRY] 注文照会の平均約定価格で更新: {symbol} = {avg_entry}")
                                if ex_sz > 0 and (self.entry_sizes.get(symbol, 0) <= 0):
                                    self.entry_sizes[symbol] = ex_sz
                                    self.logger.info(f"[ENTRY] 注文照会の実行サイズで更新: {symbol} = {ex_sz}")

                        # 2) まだ price が無ければ約定履歴から VWAP を算出
                        if avg_entry <= 0:
                            hist = self.gmo_api._request(
                                "GET", "/v1/closedOrders",
                                params={"symbol": gmo_symbol, "date": now_jst().strftime("%Y%m%d")}
                            )
                            if hist.get("status") == 0:
                                for odr in hist.get("data", {}).get("list", []):
                                    if str(odr.get("orderId")) == str(order_id):
                                        # fills があれば VWAP（Σ p*s / Σ s）
                                        fills = odr.get("fills") or []
                                        num = 0.0
                                        den = 0.0
                                        for f in fills:
                                            try:
                                                p = float(f.get("price") or 0)
                                                s = float(f.get("size") or 0)
                                                num += p * s
                                                den += s
                                            except Exception:
                                                continue
                                        vwap = (num / den) if den > 0 else float(odr.get("averagePrice") or 0)
                                        if vwap > 0:
                                            avg_entry = vwap
                                            self.entry_prices[symbol] = avg_entry
                                            self.logger.info(f"[ENTRY] 約定履歴のVWAPで entry_price 更新: {symbol} = {avg_entry}")

                                        # 実行サイズのフォールバック更新
                                        try:
                                            ex_sz = float(odr.get("executedSize") or 0)
                                            if ex_sz > 0 and (self.entry_sizes.get(symbol, 0) <= 0):
                                                self.entry_sizes[symbol] = ex_sz
                                                self.logger.info(f"[ENTRY] 約定履歴の実行サイズで更新: {symbol} = {ex_sz}")
                                        except Exception:
                                            pass
                                        break
                    except Exception as e:
                        self.logger.warning(f"[ENTRY] 平均約定価格のフォールバック取得に失敗: {e}")

                # ここまでで avg_entry が取れなかった場合、旧ロジックの last を無理に入れない（ズレの原因）
                if avg_entry <= 0:
                    self.logger.warning("[ENTRY] 平均約定価格を取得できませんでした。entry_prices は未更新のままです。")

                # 戻り値の executed_size は「建玉サイズ」 > 「注文サイズ」の優先で返す
                executed_size_return = executed_size_pos if executed_size_pos > 0 else size

                return {
                    'success': True,
                    'order_id': order_id,
                    'executed_size': executed_size_return,
                    'position_id': position_id
                }

            else:
                # エラーハンドリング
                error_messages = response.get("messages", [])
                error_msg = error_messages[0].get("message_string", "Unknown error") if error_messages else "Unknown error"
                error_code = error_messages[0].get("message_code", "Unknown") if error_messages else "Unknown"
                
                self.logger.error(f"GMO注文エラー: {error_msg} (コード: {error_code})")
                self.logger.error(f"GMO注文レスポンス全体: {response}")
                
                return {'success': False, 'error': f"{error_msg} (Code: {error_code})", 'executed_size': 0}
                    
        except Exception as e:
            self.logger.error(f"注文処理中の例外エラー: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e), 'executed_size': 0}

    def verify_position_id(self, symbol):
        """特定の通貨ペアのポジションIDを検証・再取得する"""
        try:
            # paper(sim) モードでは取引所に問い合わせない
            if not self.exchange_settings_gmo.get("live_trade", False):
                self.logger.info(f"Paper mode: verify_position_id({symbol}) はスキップします")
                return True
            if not self.gmo_api:
                self.logger.warning("GMOコインAPIが初期化されていません（live想定）。verify_position_id をスキップ")
                return False
            
            # 通貨ペアをGMO形式に変換
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
            
            # ローカルに保存されているポジション情報
            local_position = self.positions.get(symbol)
            local_position_id = self.position_ids.get(symbol)
            
            if not local_position:
                self.logger.info(f"{symbol}のローカルポジションが存在しません")
                return True  # ポジションがなければ正常
            
            # GMOから最新のポジション情報を取得
            positions_response = self._get_margin_positions_safe(gmo_symbol)
            
            if positions_response.get("status") == 0:
                positions_data = positions_response.get("data", {})
                positions = positions_data.get("list", []) if isinstance(positions_data, dict) else positions_data
                
                # 該当通貨のポジションを検索
                found_positions = []
                for pos in positions:
                    if pos.get("symbol") == gmo_symbol:
                        found_positions.append(pos)
                
                if not found_positions:
                    self.logger.warning(f"{symbol}の実際のポジションが見つかりません。ローカル情報をクリアします。")
                    self.positions[symbol] = None
                    self.position_ids[symbol] = None
                    self.entry_prices[symbol] = 0
                    self.entry_times[symbol] = None
                    self.entry_sizes[symbol] = 0
                    self.save_positions()
                    return True
                
                # ポジションIDがない場合、適切なポジションを見つける
                if not local_position_id:
                    # エントリー価格とサイズが一致するポジションを探す
                    for pos in found_positions:
                        pos_size = float(pos.get("size", 0))
                        pos_side = "long" if pos.get("side") == "BUY" else "short"
                        
                        # サイズとポジションタイプが一致
                        if (abs(pos_size - self.entry_sizes[symbol]) < 0.001 and 
                            pos_side == local_position):
                            
                            position_id = pos.get("positionId")
                            self.position_ids[symbol] = position_id
                            self.logger.info(f"{symbol}のポジションIDを再取得しました: {position_id}")
                            self.save_positions()
                            return True
                    
                    # 一致するポジションが見つからない場合、最新のポジションを使用
                    if len(found_positions) == 1:
                        pos = found_positions[0]
                        position_id = pos.get("positionId")
                        self.position_ids[symbol] = position_id
                        self.logger.warning(f"{symbol}の単一ポジションIDを設定: {position_id}")
                        self.save_positions()
                        return True
                    else:
                        self.logger.error(f"{symbol}の適切なポジションを特定できません。手動での対応が必要です。")
                        return False
                
                # ポジションIDがある場合、存在を確認
                else:
                    for pos in found_positions:
                        if str(pos.get("positionId")) == str(local_position_id):
                            self.logger.info(f"{symbol}のポジションID {local_position_id} は有効です")
                            return True
                    
                    # ポジションIDが見つからない場合
                    self.logger.warning(f"{symbol}のポジションID {local_position_id} が見つかりません。再取得を試みます。")
                    self.position_ids[symbol] = None
                    return self.verify_position_id(symbol)  # 再帰的に再取得
                
            else:
                error_msg = positions_response.get("messages", [{}])[0].get("message_string", "不明なエラー")
                self.logger.error(f"ポジション情報取得エラー: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"{symbol}のポジションID検証中にエラー: {e}", exc_info=True)
            return False

    def check_order_execution(self, order_id, symbol, margin=False):
        """GMOコインでの注文の約定状況を確認する。
        - SIM注文（order_idが 'SIM-' 始まり）は即時に内部状態から約定量を返す
        - REAL注文は /v1/orders → （必要なら）/v1/closedOrders → （最後に）建玉照会の順で確認
        """
        try:
            # ---- 1) SIM（paper）なら即返す ----------------------------------
            if isinstance(order_id, str) and order_id.startswith("SIM-"):
                executed = float(self.entry_sizes.get(symbol) or 0.0)
                if executed <= 0:
                    # 念のため positions 由来サイズを優先
                    try:
                        pos_id = self.position_ids.get(symbol)
                        if pos_id:
                            executed = float(self.entry_sizes.get(symbol) or 0.0)
                    except Exception:
                        pass
                self.logger.info(f"[SIM EXEC-CHECK] {symbol} -> executed_size={executed}")
                return executed

            # ---- 2) REAL: GMO API 必須 ---------------------------------------
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません")
                return 0.0

            # 通貨ペアをGMO形式に変換（place_orderと同一仕様）
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))

            # 2-1) 注文情報（現行日の生注文）を確認
            path = "/v1/orders"
            params = {"orderId": str(order_id)}
            response = self.gmo_api._request("GET", path, params=params)
            self.logger.info(f"GMOコイン注文確認API応答: {response}")

            if response.get("status") == 0:
                data = response.get("data", {})
                order_list = data.get("list", [])
                if not order_list:
                    self.logger.warning("注文リストが空です（/v1/orders）")
                else:
                    od0 = order_list[0]
                    status = str(od0.get("status") or "").upper()
                    # executedSize は文字列/数値両対応
                    try:
                        executed_size = float(od0.get("executedSize") or 0)
                    except Exception:
                        executed_size = 0.0

                    self.logger.info(f"注文詳細: 状態={status}, 約定サイズ={executed_size}")

                    if status == "EXECUTED":
                        return executed_size
                    if status == "ORDERED" and executed_size > 0:
                        # 部分約定
                        return executed_size
                    if status in ("WAITING", "ORDERED"):
                        # 未約定 or 進行中
                        return 0.0
                    # CANCELED/EXPIREDなどは fallthrough（0で返す）

            # 2-2) /v1/orders で見つからない・または10002等のとき → 当日クローズドを確認
            messages = response.get("messages", []) if isinstance(response, dict) else []
            is_10002 = any(m.get("message_code") == "10002" for m in messages) if messages else False
            if (response.get("status") != 0 and is_10002) or (response.get("status") == 0 and not response.get("data", {}).get("list")):
                self.logger.info("注文が履歴に移動した可能性。/v1/closedOrders を確認します。")
                history_path = "/v1/closedOrders"
                current_date = now_jst().strftime("%Y%m%d")
                history_params = {"symbol": gmo_symbol, "date": current_date}
                history_response = self.gmo_api._request("GET", history_path, params=history_params)

                if history_response.get("status") == 0:
                    orders = history_response.get("data", {}).get("list", [])
                    for od in orders:
                        if str(od.get("orderId")) == str(order_id):
                            try:
                                ex_sz = float(od.get("executedSize") or 0)
                            except Exception:
                                ex_sz = 0.0
                            if ex_sz > 0:
                                self.logger.info(f"取引履歴から注文を確認: 約定量 {ex_sz}")
                                return ex_sz

            # 2-3) それでも不明なとき → 建玉照会で推定（新規OPENのはずなのでside一致の最新を拾う）
            try:
                positions_response = self._get_margin_positions_safe(gmo_symbol)
                if positions_response.get("status") == 0:
                    # ここでは単純に合計サイズで近似
                    pos_data = positions_response.get("data", {})
                    pos_list = pos_data.get("list", []) if isinstance(pos_data, dict) else pos_data
                    total_sz = 0.0
                    for p in pos_list:
                        if p.get("symbol") == gmo_symbol:
                            try:
                                total_sz += float(p.get("size") or 0)
                            except Exception:
                                pass
                    if total_sz > 0:
                        self.logger.info(f"建玉照会ベースの約定量推定: {total_sz}")
                        return total_sz
            except Exception as e:
                self.logger.warning(f"建玉照会のフォールバックで例外: {e}")

            # ここまでで確定できなければ 0.0
            return 0.0

        except Exception as e:
            self.logger.error(f"check_order_execution 例外: {e}", exc_info=True)
            return 0.0
            
    def execute_order_with_confirmation(
        self, symbol, order_type, size, max_retries=1,
        *,  # ここからキーワード専用引数
        timeframe: str | None = None,
        strength_score: float | None = None,
        rsi: float | None = None,
        adx: float | None = None,
        atr: float | None = None,
        di_plus: float | None = None,
        di_minus: float | None = None,
        ema_fast: float | None = None,
        ema_slow: float | None = None,
        strategy_id: str | None = "v1_weighted_signals",
        version: str | None = "2025-09-21",
        signal_raw: dict | None = None,
    ):
        """確実に注文を実行し、ポジションが実際に保有されていることを確認する"""
        # --- (追加) エントリー時のみシグナルを先に記録する --------------------
        # current_position が None = 新規エントリー、Noneでない = 決済フェーズの可能性
        # 決済は signals には記録しない方針（必要なら別途 exit_signals を用意）
        signal_id = ""
        try:
            current_position_preview = self.positions.get(symbol)
            # エントリー時のみ記録（決済では記録しない）
            if current_position_preview is not None:
                raise RuntimeError("skip_recording_signal_for_exit")

            side_for_signal = "long" if str(order_type).lower() == "buy" else "short"
            try:
                current_price_preview = float(self.get_current_price(symbol) or 0.0)
            except Exception:
                current_price_preview = 0.0

            indicators = {
                "RSI": rsi, "ADX": adx, "ATR": atr,
                "DI+": di_plus, "DI-": di_minus,
                "EMA_fast": ema_fast, "EMA_slow": ema_slow,
            } if any(v is not None for v in [rsi, adx, atr, di_plus, di_minus, ema_fast, ema_slow]) else None

            signal_id = self._record_signal(
                symbol=symbol,
                timeframe=timeframe or "15m",
                side=side_for_signal,
                price=current_price_preview,
                strength_score=strength_score,
                indicators=indicators,
                strategy_id=strategy_id,
                version=version,
                status="new",
                raw=(signal_raw or {}) | {"source": "execute_order_with_confirmation/pre_order"}
            )
        except Exception as e:
            self.logger.warning(f"シグナル事前記録スキップ: {e}")

        # ----------------------------------------------------------------------
        for attempt in range(max_retries):
            try:
                self.logger.info(f"注文試行 {attempt+1}/{max_retries}: {symbol} {order_type} {size}")

                # 現在のポジション状態（決済側かどうかの目安）
                current_position = self.positions.get(symbol)

                # 1) 注文実行
                order_result = self.place_order(symbol, order_type, size)
                if not order_result.get('success'):
                    self.logger.error(f"注文失敗: {order_result.get('error', '不明なエラー')}")
                    time.sleep(2)
                    continue

                order_id = order_result.get('order_id')
                if not order_id:
                    self.logger.error("注文成功したが注文IDがありません")
                    time.sleep(2)
                    continue

                # 2) 約定確認（数回試行）
                for check in range(5):
                    time.sleep(3)
                    executed_size = self.check_order_execution(order_id, symbol)

                    if executed_size > 0:
                        self.logger.info(f"注文約定確認完了: {symbol} {order_type} サイズ:{executed_size}")

                        # 決済注文ならローカルのポジション情報クリア
                        if current_position is not None:
                            _ot = str(order_type).lower()
                            if (current_position == 'long' and _ot == 'sell') or \
                               (current_position == 'short' and _ot == 'buy'):
                                self.logger.info(f"{symbol}のポジションを決済しました")
                                self.positions[symbol] = None
                                self.entry_prices[symbol] = 0
                                self.entry_times[symbol] = None
                                self.entry_sizes[symbol] = 0

                        # 約定確認直後：DB書き込みを一体トランザクションで確定
                        try:
                            exec_price = float(self.get_current_price(symbol) or 0.0)
                            # --- 新規エントリー時はローカル状態をここで確実に初期化 ---
                            if current_position is None:
                                _side = "long" if str(order_type).lower() == "buy" else "short"
                                self.positions[symbol] = _side
                                self.entry_prices[symbol] = float(exec_price or 0.0)
                                self.entry_times[symbol] = now_jst()
                                self.entry_sizes[symbol] = float(executed_size)

                            self.logger.info(
                                "[DBTX] committing order+fill(+signal) atomically: order_id=%s exec_size=%s price=%s symbol=%s",
                                order_id, executed_size, exec_price, symbol
                            )
                            from datetime import timezone
                            with begin() as conn:
                                # 1) 注文（冪等）
                                insert_order(
                                    order_id=str(order_id),
                                    symbol=symbol,
                                    side=("BUY" if str(order_type).lower()=="buy" else "SELL"),
                                    type_="MARKET",
                                    size=float(executed_size),
                                    status="ORDERED",
                                    requested_at=utcnow(),
                                    placed_at=utcnow(),
                                    raw={"place_order_response": order_result}
                                         | {"execution_check":"direct"}
                                         | {"signal_id": signal_id or None},
                                    conn=conn,
                                )
                                # 2) フィル（冪等, 同一Tx）
                                upsert_fill(
                                    fill_id=None,
                                    order_id=str(order_id),
                                    price=float(exec_price),
                                    size=float(executed_size),
                                    fee=None,
                                    executed_at=utcnow(),
                                    raw={
                                        "source": "entry_tx",
                                        "symbol": symbol,
                                        "signal_id": signal_id or None,
                                        "order_type": str(order_type).lower(),  # 参照用
                                    },
                                    conn=conn,
                                )
                                # 2.5) ポジション（約定が確定し position_id が既知なら同一Txで反映）
                                _pos_id = self.position_ids.get(symbol)
                                if _pos_id:
                                    sid = to_uuid_or_none(getattr(self, "strategy_id", None))
                                    upsert_position(
                                        position_id=str(_pos_id),
                                        symbol=symbol,
                                        side=("long" if str(order_type).lower()=="buy" else "short"),
                                        size=float(executed_size),
                                        avg_entry_price=float(exec_price),
                                        opened_at=utcnow(),
                                        updated_at=utcnow(),
                                        raw={"source": "entry_tx"},
                                        strategy_id=sid,                               # uuid or None
                                        user_id=getattr(self, "user_id", None),
                                        source=default_source_from_env(self),          # ← 非NULLを保証
                                        conn=conn,
                                    )                             
                                # 3) シグナルstatus（エントリー時だけ）
                                if signal_id:
                                    update_signal_status(signal_id, "sent", conn=conn)
                        except Exception as e:
                            # Fail-Fast：Tx障害時は成功扱いにしない
                            insert_error("entry/tx_commit", str(e),
                                         raw={"order_id": order_id, "symbol": symbol, "signal_id": signal_id or None})
                            raise

                        return {
                            'success': True,
                            'order_id': order_id,
                            'executed_size': executed_size
                        }

                    self.logger.info(f"約定待機中... 試行 {check+1}/5")

                # 3) 約定が取れない場合のフォールバック：建玉から確認
                self.logger.warning(f"注文は送信されましたが、約定確認に時間がかかっています: {symbol} {order_type}")
                time.sleep(2)
                position_details = self.get_position_details(symbol.split('_')[0])

                if position_details and position_details.get('positions'):
                    net_size = float(position_details.get('net_size', 0) or 0)
                    if abs(net_size) > 0:
                        self.logger.info(f"ポジション情報から建玉を確認: サイズ {net_size}")

                        # --- DB: フォールバックでも fills 登録 ---
                        try:
                            try:
                                exec_price = float(self.entry_prices.get(symbol) or 0.0)
                                if exec_price <= 0:
                                    exec_price = float(self.get_current_price(symbol) or 0.0)
                            except Exception:
                                exec_price = 0.0

                            mark_order_executed_with_fill(
                                order_id=str(order_id),
                                executed_size=float(abs(net_size)),
                                price=float(exec_price),
                                fee=None,
                                executed_at=utcnow(),
                                fill_raw={
                                    "source": order_type,
                                    "symbol": symbol,
                                    # （追加）signal_id を橋渡し
                                    "signal_id": signal_id or None
                                }
                            )
                            # --- メモリ側も同期（エントリー/エグジット双方の整合性向上） ---
                            if current_position is None:
                                # フォールバック成立 = エントリー成功とみなす
                                self.positions[symbol] = ("long" if str(order_type).lower()=="buy" else "short")
                                self.entry_prices[symbol] = float(exec_price or 0.0)
                                self.entry_times[symbol] = now_jst()
                                self.entry_sizes[symbol] = float(abs(net_size))
                            else:
                                # フォールバック成立 = 決済成功とみなす
                                _ot = str(order_type).lower()
                                if (current_position == 'long' and _ot == 'sell') or (current_position == 'short' and _ot == 'buy'):
                                    self.positions[symbol] = None
                                    self.entry_prices[symbol] = 0
                                    self.entry_times[symbol] = None
                                    self.entry_sizes[symbol] = 0
                        except Exception as e:
                            insert_error(
                                "execute_order_with_confirmation/fallback_fill",
                                str(e),
                                raw={"order_id": order_id, "symbol": symbol, "net_size": net_size, "signal_id": signal_id or None}
                            )

                        return {
                            'success': True,
                            'order_id': order_id,
                            'executed_size': abs(net_size)
                        }

                # 4) どちらでも確認できなかった
                self.logger.error(f"約定も建玉も確認できませんでした: {symbol} {order_type}")

            except Exception as e:
                self.logger.error(f"注文処理中のエラー: {e}")

            # 次の試行まで少し待機
            wait_time = 3 + random.uniform(0, 2)
            self.logger.info(f"{wait_time:.1f}秒待機して再試行します")
            time.sleep(wait_time)

        # 全て失敗
        self.logger.error(f"すべての試行が失敗しました: {symbol} {order_type} {size}")
        return {'success': False, 'error': "最大試行回数を超えました", 'executed_size': 0}

    # === replace the whole get_cached_data(...) ===
    def get_cached_data(self, symbol, timeframe, date_str=None, fallback_days=3):
        """キャッシュからデータを取得するか、必要に応じてAPIから取得（改良版）
        
        Parameters:
        symbol (str): 通貨ペア
        timeframe (str): 時間枠 (5min, 1hour, 1day など)
        date_str (str): 日付文字列 (YYYYMMDD)、Noneの場合は有効な日付を自動検索
        fallback_days (int): データが取得できない場合、何日前までさかのぼるか
        
        Returns:
        pandas.DataFrame: 価格データ
        """
        # date_strがNoneの場合、有効な日付を検索
        if date_str is None:
            # 直接今日の日付を使用してみる
            date_str = now_jst().strftime('%Y%m%d')
            self.logger.info(f"15分足データの日付として本日の日付 {date_str} を試行")
                
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{timeframe}_{date_str}.json")
        
        # キャッシュの有効期限チェック（追加）
        cache_valid = False
        if os.path.exists(cache_file):
            file_mtime = os.path.getmtime(cache_file)
            cache_age_hours = (time.time() - file_mtime) / 3600
            
            # 時間枠に応じた有効期限設定
            if timeframe == '5min' and cache_age_hours < 1:  # 5分足は1時間有効
                cache_valid = True
            elif timeframe == '1hour' and cache_age_hours < 3:  # 1時間足は3時間有効
                cache_valid = True
            elif timeframe == '1day' and cache_age_hours < 24:  # 日足は24時間有効
                cache_valid = True
        
        # キャッシュから読み込み試行
        if cache_valid:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    
                # キャッシュデータの検証を追加
                if data.get('success') != 1 or 'candlestick' not in data.get('data', {}):
                    self.logger.info(f"キャッシュデータが無効です: {cache_file}")
                    if os.path.exists(cache_file):
                        # 無効なキャッシュファイルをリネーム（後で分析用に保持）
                        invalid_file = cache_file + '.invalid'
                        try:
                            os.rename(cache_file, invalid_file)
                            self.logger.info(f"無効なキャッシュファイルをリネーム: {invalid_file}")
                        except:
                            # リネームに失敗したら削除
                            os.remove(cache_file)
                            self.logger.info(f"無効なキャッシュファイルを削除: {cache_file}")
                    data = {'success': 0}
            except json.JSONDecodeError:
                self.logger.info(f"キャッシュファイル破損: {cache_file}、APIから再取得します")
                if os.path.exists(cache_file):
                    # 破損ファイルをリネーム
                    corrupt_file = cache_file + '.corrupt'
                    try:
                        os.rename(cache_file, corrupt_file)
                    except:
                        os.remove(cache_file)
                data = {'success': 0}
            except Exception as e:
                self.logger.error(f"キャッシュ読み込みエラー: {e}")
                data = {'success': 0}
        else:
            # キャッシュが無効または存在しない場合
            if os.path.exists(cache_file):
                self.logger.debug(f"キャッシュの有効期限切れ: {cache_file}")
            data = {'success': 0}
        
        # APIからデータを取得（キャッシュが無効または失敗した場合）
        if data.get('success') != 1:
            # エクスポネンシャルバックオフによるリトライ実装
            retry_count = 0
            max_retries = 1
            retry_delay = 1  # 初期遅延2秒
            
            while retry_count < max_retries:
                try:
                    # APIからデータを取得
                    url = f'https://public.bitbank.cc/{symbol}/candlestick/{timeframe}/{date_str}'
                    
                    # ユーザーエージェントを追加（サーバー側でブロックされないように）
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    response = requests.get(url, headers=headers, timeout=15)  # タイムアウト増加
                    
                    # レート制限を遵守するための遅延
                    time.sleep(self.rate_limit_delay + random.uniform(0, 0.5))  # ランダム要素追加
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # 成功したらキャッシュに保存
                        if data.get('success') == 1 and 'candlestick' in data.get('data', {}):
                            try:
                                with open(cache_file, 'w') as f:
                                    json.dump(data, f)
                                break  # 成功したらループ終了
                            except Exception as e:
                                self.logger.error(f"キャッシュ保存エラー: {e}")
                        else:
                            error_code = data.get('data', {}).get('code', 'unknown')
                            self.logger.warning(f"API応答エラー: {url}, コード: {error_code}")
                    else:
                        self.logger.warning(f"HTTP応答エラー: {url}, コード: {response.status_code}")
                
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"APIリクエストエラー({retry_count+1}/{max_retries}): {e}")
                except ValueError as e:
                    self.logger.error(f"JSONパースエラー: {e}")
                except Exception as e:
                    self.logger.error(f"API呼び出し未知のエラー: {e}")
                
                # エクスポネンシャルバックオフでリトライ
                retry_count += 1
                if retry_count < max_retries:
                    base_delay = 0.3
                    max_delay = 2.0
                    sleep_time = min(max_delay, base_delay * (2 ** (retry_count - 1))) + random.uniform(0, 0.3)
                    self.logger.info(f"{sleep_time:.1f}秒待機してリトライします")
                    time.sleep(sleep_time)
        
        # データの変換
        if data.get('success') == 1 and 'candlestick' in data.get('data', {}):
            try:
                candles = data['data']['candlestick'][0]['ohlcv']
                
                # 空のデータをチェック
                if not candles:
                    self.logger.warning(f"APIからの空データ: {symbol} {timeframe} {date_str}")
                    return pd.DataFrame()
                
                # データフレームに変換
                df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
                
                # データ型変換をより堅牢に
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                # 9時間を追加して日本時間に変換
                df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=9)
                
                # データの整合性チェック（改良）
                # 高値 < 安値のような矛盾したデータがないか確認
                inconsistent = ((df['high'] < df['low']) | 
                            (df['high'] < df['open']) | 
                            (df['high'] < df['close']) |
                            (df['low'] > df['open']) | 
                            (df['low'] > df['close']))
                
                inconsistent_count = inconsistent.sum()
                if inconsistent_count > 0:
                    self.logger.warning(f"整合性のない価格データ: {inconsistent_count}件")
                    
                    # 整合性エラーの修正
                    for i in df.index[inconsistent]:
                        row = df.loc[i]
                        # 高値を最大値に、安値を最小値に修正
                        values = [row['open'], row['close'], row['high'], row['low']]
                        df.loc[i, 'high'] = max(values)
                        df.loc[i, 'low'] = min(values)
                
                # NaN値のチェックと修正
                nan_counts = df.isna().sum()
                if nan_counts.sum() > 0:
                    self.logger.warning(f"NaN値検出: {nan_counts.to_dict()}")
                    
                    # 前方値補完（修正版）
                    df = df.ffill()
                    
                    # それでも残るNaN値は後方値補完（修正版）
                    df = df.bfill()
                    
                    # それでも残るNaN値（両端など）は列の平均値で補完
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if df[col].isna().any():
                            df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 0)
                
                return df
            except Exception as e:
                self.logger.error(f"データ変換エラー: {e}", exc_info=True)
                return pd.DataFrame()
        
        # フォールバック処理（前日のデータを試す）
        if fallback_days > 0:
            try:
                self.logger.info(f"{symbol} {timeframe} {date_str}のデータ取得失敗。前日データを試みます")
                previous_date = (datetime.strptime(date_str, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
                return self.get_cached_data(symbol, timeframe, previous_date, fallback_days-1)
            except Exception as e:
                self.logger.error(f"フォールバック処理エラー: {e}")
                    
        return pd.DataFrame()  # 空のDataFrameを返す（失敗時）

    def build_features(self, df):
        """テクニカル指標の計算（改善版）"""
        # データが空または不足している場合は空のデータフレームを返す
        if df.empty:
            return df
        
        try:
            
            # 必要なカラムが存在するか確認
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"特徴量計算: 不足しているカラム={missing_columns}")
                # 不足しているカラムがあれば、初期値を追加
                for col in missing_columns:
                    if col in ['open', 'high', 'low', 'close']:
                        # 価格カラムがなければ警告して終了
                        self.logger.error(f"必須価格カラム {col} がありません。特徴量計算を中止します。")
                        return df
                    elif col == 'volume':
                        df['volume'] = 0
                    elif col == 'timestamp':
                        df['timestamp'] = pd.to_datetime('now')
            
            # データ型を確認し、必要に応じて変換
            for col in ['close', 'high', 'low', 'volume']:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    self.logger.warning(f"{col}カラムが数値型ではありません。変換を試みます。")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # NaN値をチェック
            nan_check = {col: df[col].isna().sum() for col in ['open', 'high', 'low', 'close', 'volume'] 
                        if col in df.columns}
            if any(nan_check.values()):
                self.logger.warning(f"特徴量計算: NaN値の数={nan_check}")
                # NaN値を前の値で埋める
                df = df.fillna(method='ffill')
                # それでも残るNaN値を0で埋める
                df = df.fillna(0)
            
            # ========== 改良版の指標計算 ==========
            
            if all(col in df.columns for col in ['high', 'low', 'close']):
                try:
                    # トゥルーレンジの計算
                    df['tr1'] = df['high'] - df['low']
                    df['tr2'] = abs(df['high'] - df['close'].shift(1))
                    df['tr3'] = abs(df['low'] - df['close'].shift(1))
                    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
                    
                    # +DMと-DMの計算
                    df['up_move'] = df['high'] - df['high'].shift(1)
                    df['down_move'] = df['low'].shift(1) - df['low']
                    
                    # +DIと-DIの計算前の条件設定
                    df['plus_dm'] = 0.0
                    mask = (df['up_move'] > df['down_move']) & (df['up_move'] > 0)
                    df.loc[mask, 'plus_dm'] = df.loc[mask, 'up_move']
                    
                    df['minus_dm'] = 0.0
                    mask = (df['down_move'] > df['up_move']) & (df['down_move'] > 0)
                    df.loc[mask, 'minus_dm'] = df.loc[mask, 'down_move']
                    
                    # 移動平均を使用してスムーズ化（14期間）
                    adx_period = 14
                    df['tr14'] = df['tr'].rolling(window=adx_period, min_periods=adx_period//2).mean()
                    df['plus_di14'] = 100 * (df['plus_dm'].rolling(window=adx_period, min_periods=adx_period//2).mean() / df['tr14'])
                    df['minus_di14'] = 100 * (df['minus_dm'].rolling(window=adx_period, min_periods=adx_period//2).mean() / df['tr14'])
                    
                    # DIの差の絶対値
                    df['di_diff'] = abs(df['plus_di14'] - df['minus_di14'])
                    df['di_sum'] = df['plus_di14'] + df['minus_di14']

                    # DXの計算
                    df['dx'] = 100 * (df['di_diff'] / df['di_sum'].replace(0, 1e-10))
                    
                    # ADXの計算（DXの14期間移動平均）
                    df['ADX'] = df['dx'].rolling(window=adx_period, min_periods=adx_period//2).mean()
                    
                    # ADXが極端に高い値になる場合があるので100に制限
                    df['ADX'] = df['ADX'].clip(0, 100)
                    
                    # NaN値の処理
                    df['ADX'] = df['ADX'].fillna(20)  # 初期値としてトレンドなしの状態を想定
                    
                    # 不要な中間計算列を削除
                    for col in ['tr1', 'tr2', 'tr3', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'tr14', 'di_diff', 'di_sum', 'dx']:
                        if col in df.columns:
                            df = df.drop(col, axis=1)
                
                except Exception as e:
                    self.logger.error(f"ADX計算エラー: {e}")
                    df['ADX'] = 20  # エラー時はデフォルト値を設定
                    df['plus_di14'] = 25
                    df['minus_di14'] = 25

            # RSI計算
            if 'close' in df.columns:
                delta = df['close'].diff()

                gain = delta.copy()
                loss = delta.copy()

                gain[gain < 0] = 0
                loss[loss > 0] = 0
                loss = abs(loss)

                # ここでEMA（指数移動平均）を使用するよう変更
                rsi_period = 7
                avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
                avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()

                # ゼロ除算防止（変わらず必要）
                rs = avg_gain / avg_loss.replace(0, 1e-10)
                df['RSI'] = 100 - (100 / (1 + rs))

                # 異常な値のチェックと修正（この処理はそのままでOK）
                df['RSI'] = df['RSI'].fillna(50)
                df['RSI'] = df['RSI'].clip(0, 100)

            # MFI計算（RSI計算の後）
            if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                try:
                    # 典型的な価格を計算
                    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
                    
                    # 資金フローを計算
                    df['money_flow'] = df['typical_price'] * df['volume']
                    
                    # 価格変化の計算
                    df['price_change'] = df['typical_price'].diff()
                    
                    # 正と負の資金フローを分離
                    df['positive_flow'] = np.where(df['price_change'] > 0, df['money_flow'], 0)
                    df['negative_flow'] = np.where(df['price_change'] < 0, df['money_flow'], 0)
                    
                    # MFI期間設定
                    mfi_period = 14
                    
                    # 期間内の合計を計算
                    positive_flow_sum = df['positive_flow'].rolling(window=mfi_period, min_periods=mfi_period//2).sum()
                    negative_flow_sum = df['negative_flow'].rolling(window=mfi_period, min_periods=mfi_period//2).sum()
                    
                    # ゼロ除算防止
                    negative_flow_sum = negative_flow_sum.replace(0, 1e-10)
                    
                    # 資金フロー比率
                    money_ratio = positive_flow_sum / negative_flow_sum
                    
                    # MFI計算
                    df['MFI'] = 100 - (100 / (1 + money_ratio))
                    
                    # NaN値やエラー値の処理
                    df['MFI'] = df['MFI'].fillna(50)
                    df['MFI'] = df['MFI'].clip(0, 100)
                    
                    # 中間計算列の削除
                    for col in ['typical_price', 'money_flow', 'price_change', 'positive_flow', 'negative_flow']:
                        if col in df.columns:
                            df = df.drop(col, axis=1)
                except Exception as e:
                    self.logger.error(f"MFI計算エラー: {e}")
                    df['MFI'] = 50  # エラー時はデフォルト値を設定
            
            # EMA計算（より堅牢な方法）
            if 'close' in df.columns:
                # 短期EMA
                df['EMA_short'] = df['close'].ewm(span=5, min_periods=3, adjust=False).mean()
                # 長期EMA
                df['EMA_long'] = df['close'].ewm(span=25, min_periods=12, adjust=False).mean()
            
            # 移動平均線（最小期間設定を追加）
            if 'close' in df.columns:
                df['MA25'] = df['close'].rolling(window=25, min_periods=5).mean()
            
            # ボリューム平均
            if 'volume' in df.columns:
                df['vol_avg'] = df['volume'].rolling(window=10, min_periods=3).mean()
            
            # CCIの堅牢な計算
            if all(col in df.columns for col in ['high', 'low', 'close']):
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                tp_ma = typical_price.rolling(window=10, min_periods=5).mean()
                
                # 平均偏差の計算改善
                def mad(x):
                    # 中央値からの絶対偏差の平均（より堅牢）
                    return np.abs(x - np.median(x)).mean()
                                
                tp_md = typical_price.rolling(window=10, min_periods=5).apply(mad, raw=True)


                # ゼロ除算防止とエラーハンドリング
                tp_md = tp_md.replace(0, 1e-10)
                df['CCI'] = (typical_price - tp_ma) / (0.015 * tp_md)
                
                # 極端な値の制限
                df['CCI'] = df['CCI'].clip(-200, 200)
            
            # ATR計算（改良版）
            if all(col in df.columns for col in ['high', 'low', 'close']):
                tr1 = df['high'] - df['low']
                tr2 = abs(df['high'] - df['close'].shift(1))
                tr3 = abs(df['low'] - df['close'].shift(1))
                
                # NaNを0に置換して計算エラーを防止
                tr2 = tr2.fillna(tr1)
                tr3 = tr3.fillna(tr1)
                
                # 各行ごとに最大値を計算
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df['ATR'] = tr.rolling(window=7, min_periods=3).mean()
            
            # ボリンジャーバンド計算（改良版）
            if 'close' in df.columns:
                ma_period = 20
                std_dev = 2
                # 最小期間を設定して安定性向上
                df['BB_mid'] = df['close'].rolling(window=ma_period, min_periods=max(5, ma_period//4)).mean()
                df['BB_std'] = df['close'].rolling(window=ma_period, min_periods=max(5, ma_period//4)).std()
                
                # NaN対策
                df['BB_mid'] = df['BB_mid'].fillna(df['close'])
                df['BB_std'] = df['BB_std'].fillna(df['close'] * 0.02)  # 標準的な変動として2%設定
                
                df['BB_upper'] = df['BB_mid'] + (df['BB_std'] * std_dev)
                df['BB_lower'] = df['BB_mid'] - (df['BB_std'] * std_dev)
                
                # ゼロ除算防止
                safe_mid = df['BB_mid'].replace(0, 1e-10)
                df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / safe_mid

            # MACD計算（改良版）
            if 'close' in df.columns:
                try:
                    # MACD設定
                    fast_period = 12
                    slow_period = 26
                    signal_period = 9
                    
                    # 短期EMAと長期EMAを計算
                    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
                    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
                    
                    # MACDライン（短期EMA - 長期EMA）
                    df['MACD'] = ema_fast - ema_slow
                    
                    # シグナルライン（MACDの9期間EMA）
                    df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
                    
                    # MACDヒストグラム（MACD - シグナル）
                    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
                    
                    # NaN値の処理
                    df['MACD'] = df['MACD'].fillna(0)
                    df['MACD_signal'] = df['MACD_signal'].fillna(0)
                    df['MACD_histogram'] = df['MACD_histogram'].fillna(0)
                    
                except Exception as e:
                    self.logger.error(f"MACD計算エラー: {e}")
                    df['MACD'] = 0
                    df['MACD_signal'] = 0
                    df['MACD_histogram'] = 0
            
            # NaN値の処理 (すべての計算が終わった後)
            for col in df.columns:
                if col not in ['timestamp', 'time', 'date'] and df[col].isna().any():
                    # カラムに応じて適切なデフォルト値を設定
                    if col == 'RSI':
                        df[col] = df[col].fillna(50)
                    elif col in ['EMA_short', 'EMA_long', 'MA25', 'BB_mid']:
                        df[col] = df[col].fillna(df['close'] if 'close' in df.columns else 0)
                    elif col == 'vol_avg':
                        df[col] = df[col].fillna(df['volume'].mean() if 'volume' in df.columns else 0)
                    elif col == 'CCI':
                        df[col] = df[col].fillna(0)
                    elif col == 'ATR':
                        df[col] = df[col].fillna(df['high'].mean() * 0.01 if 'high' in df.columns else 0)
                    elif col == 'BB_upper':
                        df[col] = df[col].fillna(df['close'] * 1.02 if 'close' in df.columns else 0)
                    elif col == 'BB_lower':
                        df[col] = df[col].fillna(df['close'] * 0.98 if 'close' in df.columns else 0)
                    elif col == 'BB_width':
                        df[col] = df[col].fillna(0.04)
                    else:
                        df[col] = df[col].fillna(0)

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].round(4)
            
            # 結果の検証
            debug_info = {
                'RSI': {'min': df['RSI'].min(), 'max': df['RSI'].max(), 'mean': df['RSI'].mean()} if 'RSI' in df.columns else {},
                'CCI': {'min': df['CCI'].min(), 'max': df['CCI'].max(), 'mean': df['CCI'].mean()} if 'CCI' in df.columns else {},
                'データ行数': len(df)
            }
            
            return df

        except Exception as e:
            self.logger.error(f"特徴量計算エラー: {e}", exc_info=True)
            # エラー時のフォールバック処理（基本機能確保）
            if not df.empty and 'close' in df.columns:
                # 最低限の指標を設定
                df['RSI'] = 50 
                df['EMA_short'] = df['close']
                df['EMA_long'] = df['close']
                df['MA25'] = df['close']
                if 'volume' in df.columns:
                    df['vol_avg'] = df['volume'].mean()
                else:
                    df['vol_avg'] = 0
                df['CCI'] = 0
                df['ATR'] = 0
                df['BB_mid'] = df['close']
                df['BB_upper'] = df['close'] * 1.02
                df['BB_lower'] = df['close'] * 0.98
                df['BB_width'] = 0.04
                
                self.logger.warning("エラー発生のためフォールバック指標を使用します")
            return df

    def generate_signals(self, symbol, df_5min, df_hourly):
        """ロングとショート両方のシグナル生成関数（ADX指標を追加）

        Parameters:
        symbol (str): 通貨ペア
        df_5min (pandas.DataFrame): 5分足データ
        df_hourly (pandas.DataFrame): 1時間足データ

        Returns:
        pandas.DataFrame: シグナルを追加したデータフレーム
        """

        # 通貨ペアごとの指標重み付け辞書 - MACDを追加
        indicator_weights_long = {
            'doge_jpy': {'bb': 0.05, 'cci': 0.25, 'rsi': 0.15, 'mfi': 0.05, 'adx': 0.45, 'volume': 0.00, 'ma': 0.05, 'atr': 0.0, 'macd': 0.0},
            'sol_jpy':  {'bb': 0.05, 'cci': 0.25, 'rsi': 0.15, 'mfi': 0.05, 'adx': 0.45, 'volume': 0.00, 'ma': 0.05, 'atr': 0.0, 'macd': 0.0},
            'xrp_jpy':  {'bb': 0.05, 'cci': 0.25, 'rsi': 0.15, 'mfi': 0.05, 'adx': 0.45, 'volume': 0.00, 'ma': 0.05, 'atr': 0.0, 'macd': 0.0},

            'ltc_jpy':  {'bb': 0.05, 'cci': 0.25, 'rsi': 0.15, 'mfi': 0.05, 'adx': 0.40, 'volume': 0.00, 'ma': 0.10, 'atr': 0.0, 'macd': 0.0},
            'ada_jpy':  {'bb': 0.05, 'cci': 0.25, 'rsi': 0.15, 'mfi': 0.05, 'adx': 0.40, 'volume': 0.00, 'ma': 0.10, 'atr': 0.0, 'macd': 0.0},

            'eth_jpy':  {'bb': 0.00, 'cci': 0.25, 'rsi': 0.15, 'mfi': 0.05, 'adx': 0.45, 'volume': 0.00, 'ma': 0.10, 'atr': 0.0, 'macd': 0.0},
            'bcc_jpy':  {'bb': 0.00, 'cci': 0.25, 'rsi': 0.15, 'mfi': 0.05, 'adx': 0.45, 'volume': 0.00, 'ma': 0.10, 'atr': 0.0, 'macd': 0.0},
        }

        indicator_weights_short = {
            'doge_jpy': {'adx': 0.25, 'cci': 0.30, 'rsi': 0.15, 'bb': 0.20, 'mfi': 0.05, 'volume': 0.05, 'ma': 0.00, 'atr': 0.0, 'macd': 0.0},
            'sol_jpy':  {'adx': 0.25, 'cci': 0.30, 'rsi': 0.15, 'bb': 0.20, 'mfi': 0.05, 'volume': 0.05, 'ma': 0.00, 'atr': 0.0, 'macd': 0.0},
            'xrp_jpy':  {'adx': 0.25, 'cci': 0.30, 'rsi': 0.15, 'bb': 0.20, 'mfi': 0.05, 'volume': 0.05, 'ma': 0.00, 'atr': 0.0, 'macd': 0.0},

            'ltc_jpy':  {'adx': 0.25, 'cci': 0.30, 'rsi': 0.15, 'bb': 0.20, 'mfi': 0.05, 'volume': 0.05, 'ma': 0.00, 'atr': 0.0, 'macd': 0.0},
            'ada_jpy':  {'adx': 0.25, 'cci': 0.30, 'rsi': 0.15, 'bb': 0.20, 'mfi': 0.05, 'volume': 0.05, 'ma': 0.00, 'atr': 0.0, 'macd': 0.0},

            'eth_jpy':  {'adx': 0.25, 'cci': 0.30, 'rsi': 0.15, 'bb': 0.20, 'mfi': 0.05, 'volume': 0.05, 'ma': 0.00, 'atr': 0.0, 'macd': 0.0},
            'bcc_jpy':  {'adx': 0.25, 'cci': 0.30, 'rsi': 0.15, 'bb': 0.20, 'mfi': 0.05, 'volume': 0.05, 'ma': 0.00, 'atr': 0.0, 'macd': 0.0},
        }

        signal_thresholds = {
            'doge_jpy': {'buy': 0.0, 'sell': 0.0},
            'sol_jpy':  {'buy': 0.0, 'sell': 0.0},
            'xrp_jpy':  {'buy': 0.0, 'sell': 0.0},
            'ltc_jpy':  {'buy': 0.0, 'sell': 0.0},
            'ada_jpy':  {'buy': 0.0, 'sell': 0.0},
            'eth_jpy':  {'buy': 0.0, 'sell': 0.0},
            'bcc_jpy':  {'buy': 0.0, 'sell': 0.0},
        }

        # デフォルト閾値（上記に含まれていない通貨ペア用）
        default_thresholds = {'buy': 0.51, 'sell': 0.51}
        
        # 買いシグナルと売りシグナル列を初期化
        df_5min['buy_signal'] = False
        df_5min['sell_signal'] = False

        # 最初に必要なカラムを確認し、空のデータフレームの場合にも対応
        if df_5min.empty:
            self.logger.warning(f"{symbol}の5分足データが空です")
            return pd.DataFrame({'buy_signal': [], 'sell_signal': []})

        # 必要なカラムがあるか確認 - MACDを追加
        required_columns = ['close', 'RSI', 'EMA_short', 'EMA_long', 'MA25', 'volume', 'vol_avg', 'CCI', 'ATR', 
                            'BB_upper', 'BB_lower', 'BB_mid', 'BB_width', 'ADX', 'plus_di14', 'minus_di14', 'MFI',
                            'MACD', 'MACD_signal', 'MACD_histogram']
        missing_columns = [col for col in required_columns if col not in df_5min.columns]

        if missing_columns:
            self.logger.warning(f"{symbol}のデータに不足している列があります: {missing_columns}")
            # 不足している列を追加（デフォルト値で）
            for col in missing_columns:
                if col == 'RSI':
                    df_5min[col] = 50  # 中間値
                elif col in ['EMA_short', 'EMA_long', 'MA25', 'BB_mid']:
                    df_5min[col] = df_5min['close'] if 'close' in df_5min.columns else 0
                elif col == 'vol_avg':
                    df_5min[col] = df_5min['volume'].mean() if 'volume' in df_5min.columns else 0
                elif col == 'CCI':
                    df_5min[col] = 0  # 中間値
                elif col == 'ATR':
                    df_5min[col] = 0  # デフォルト値
                elif col == 'BB_upper':
                    df_5min[col] = df_5min['close'] * 1.02 if 'close' in df_5min.columns else 0
                elif col == 'BB_lower':
                    df_5min[col] = df_5min['close'] * 0.98 if 'close' in df_5min.columns else 0
                elif col == 'BB_width':
                    df_5min[col] = 0.04  # デフォルト値
                elif col == 'ADX':
                    df_5min[col] = 20  # トレンドなしの状態をデフォルト値として設定
                elif col == 'plus_di14':
                    df_5min[col] = 25  # 中間値
                elif col == 'minus_di14':
                    df_5min[col] = 25  # 中間値
                elif col == 'MFI':
                    df_5min[col] = 50  # 中間値
                elif col in ['MACD', 'MACD_signal', 'MACD_histogram']:
                    df_5min[col] = 0  # デフォルト値
                else:
                    df_5min[col] = 0
        
        def rolling_minmax_scaler(series, window=50):
            """ローリング最小最大スケーラー（修正版）
            
            Parameters:
            series (pd.Series): 正規化対象のシリーズ
            window (int): ローリングウィンドウサイズ
            
            Returns:
            pd.Series: 正規化されたシリーズ
            """
            try:
                # 入力検証
                if series is None or series.empty or len(series) == 0:
                    return pd.Series([], dtype=float, index=series.index if hasattr(series, 'index') else [])
                
                # seriesをpandas Seriesに変換（numpy arrayの場合）
                if not isinstance(series, pd.Series):
                    series = pd.Series(series)
                
                # NaN値を事前に処理（新しいpandas形式）
                series_clean = series.ffill().bfill().fillna(0.5)
                
                scaled = []
                
                for i in range(len(series_clean)):
                    try:
                        if i < window:
                            # 初期値として中央値を設定
                            scaled.append(0.5)
                        else:
                            # iloc を使用して位置ベースでアクセス
                            window_data = series_clean.iloc[max(0, i - window):i]
                            
                            # ウィンドウデータが空でないことを確認
                            if len(window_data) == 0:
                                scaled.append(0.5)
                                continue
                            
                            # 最小値・最大値を計算
                            min_val = window_data.min()
                            max_val = window_data.max()
                            current_val = series_clean.iloc[i]
                            
                            # 値の有効性をチェック
                            if pd.isna(min_val) or pd.isna(max_val) or pd.isna(current_val):
                                scaled.append(0.5)
                                continue
                            
                            # 最大値と最小値が同じ場合の処理
                            if max_val == min_val or abs(max_val - min_val) < 1e-10:
                                scaled.append(0.5)  # 差がない場合は中央値
                            else:
                                # 正規化の計算
                                normalized = (current_val - min_val) / (max_val - min_val)
                                # 0-1の範囲にクリップ
                                normalized = max(0.0, min(1.0, normalized))
                                
                                # 結果が有効な数値であることを確認
                                if pd.isna(normalized) or not np.isfinite(normalized):
                                    normalized = 0.5
                                    
                                scaled.append(normalized)
                                
                    except Exception as inner_error:
                        # 個別の計算でエラーが発生した場合は中央値を使用
                        scaled.append(0.5)
                        continue
                
                # 元のseriesと同じindexを持つPandas Seriesとして返す
                result = pd.Series(scaled, index=series.index, dtype=float)
                
                # 結果の検証
                if result.isna().any():
                    result = result.fillna(0.5)
                    
                return result
                
            except Exception as e:
                # 全体的なエラーが発生した場合は、全て中央値のSeriesを返す
                if hasattr(series, 'index'):
                    return pd.Series([0.5] * len(series), index=series.index, dtype=float)
                else:
                    return pd.Series([0.5], dtype=float)

        # 各スコアをベクトル化して計算
        def calc_score(series, min_val, max_val, reverse=False):
            denom = max_val - min_val
            scores = (series - min_val) / denom
            if reverse:
                scores = 1 - scores
            return scores.clip(0, 1)

        # RSIスコア
        rsi_score_long = calc_score(df_5min['RSI'], 0, 100, reverse=True)
        rsi_score_short = calc_score(df_5min['RSI'], 0, 100)

        # CCIスコア
        cci_score_long = calc_score(df_5min['CCI'], -200, 200, reverse=True)
        cci_score_short = calc_score(df_5min['CCI'], -200, 200)

        # ボリュームスコア
        relative_volume = df_5min['volume'] / (df_5min['vol_avg'] + 1e-9)  # 0除算防止
        volume_score = calc_score(relative_volume, 0.5, 2.0)  # 0.5〜2.0をスコア範囲とする

        # ボリンジャーバンドスコア
        bb_score_long = calc_score(df_5min['close'], df_5min['BB_lower'], df_5min['BB_upper'], reverse=True)
        bb_score_short = calc_score(df_5min['close'], df_5min['BB_lower'], df_5min['BB_upper'])

        # 移動平均スコア
        ma_score_long = calc_score(df_5min['close'], df_5min['MA25'], df_5min['MA25']*1.05)  # 移動平均の5%上まで
        ma_score_short = calc_score(df_5min['close'], df_5min['MA25']*0.95, df_5min['MA25'], reverse=True)  # 移動平均の5%下まで

        # ADXスコアを追加（新規）
        # ADX値が高いほどトレンドが強く、シグナルの信頼性が高まる
        adx_trend_strength = calc_score(df_5min['ADX'], 15, 50)  # 15未満は弱いトレンド、50以上は非常に強いトレンド
        
        # MFIスコア（過売り/過買いの判断に使用）
        mfi_score_long = calc_score(df_5min['MFI'], 0, 100, reverse=True)
        mfi_score_short = calc_score(df_5min['MFI'], 0, 100)

        # ATRスコア（ボラティリティ指標として使用）
        # ATRが高い = ボラティリティが高い = エントリーリスクが高い
        # 一般的に、ATRが低い時の方がエントリーに適している
        atr_last50 = df_5min['ATR'].astype('float64').shift(1).rolling(50, min_periods=50)

        # pandas 1.5未満では interpolation='linear' を使用
        q80_last50 = atr_last50.quantile(0.8, interpolation='linear')
        q20_last50 = atr_last50.quantile(0.2, interpolation='linear')

        # 元の書き方に倣ってcalc_scoreでスコア化（reverse=True）
        atr_score_long  = calc_score(df_5min['ATR'], q20_last50, q80_last50, reverse=True)
        atr_score_short = calc_score(df_5min['ATR'], q20_last50, q80_last50, reverse=True)
        
        # +DIと-DIの差から方向性の強さを判断
        # +DIが-DIよりも大きい場合、上昇トレンドの可能性が高い
        df_5min['di_direction'] = df_5min['plus_di14'] - df_5min['minus_di14']
        
        # 上昇トレンド強度（+DIが-DIより高いほど強い）
        uptrend_strength = calc_score(df_5min['di_direction'], -30, 30)
        
        # 下降トレンド強度（-DIが+DIより高いほど強い）
        downtrend_strength = 1 - uptrend_strength
        
        # ADXとトレンド方向を組み合わせたスコア
        adx_score_long = adx_trend_strength * uptrend_strength
        adx_score_short = adx_trend_strength * downtrend_strength

        # MACDスコア（トレンド転換と勢いを判断）
        # MACDヒストグラムを主に使用（勢いの変化を捉える）
        # ヒストグラムが正の値 = 上昇の勢い、負の値 = 下降の勢い
        macd_score_long = calc_score(df_5min['MACD_histogram'], df_5min['MACD_histogram'].quantile(0.2), df_5min['MACD_histogram'].quantile(0.8))
        macd_score_short = calc_score(df_5min['MACD_histogram'], df_5min['MACD_histogram'].quantile(0.8), df_5min['MACD_histogram'].quantile(0.2))

        # MACDラインとシグナルラインのクロスオーバーも考慮
        # MACD > シグナル = 買いシグナル、MACD < シグナル = 売りシグナル
        macd_crossover = df_5min['MACD'] - df_5min['MACD_signal']
        macd_crossover_score_long = calc_score(macd_crossover, macd_crossover.quantile(0.2), macd_crossover.quantile(0.8))
        macd_crossover_score_short = calc_score(macd_crossover, macd_crossover.quantile(0.8), macd_crossover.quantile(0.2))

        # ヒストグラムとクロスオーバーを組み合わせたMACDスコア
        macd_score_long = (macd_score_long * 0.7 + macd_crossover_score_long * 0.3)
        macd_score_short = (macd_score_short * 0.7 + macd_crossover_score_short * 0.3)

        # 該当通貨ペアのロング/ショートの重みを取得（デフォルトあり）
        weights_long = indicator_weights_long.get(symbol, {
            'rsi': 0.15, 'cci': 0.15, 'volume': 0.15, 'bb': 0.15, 'ma': 0.15, 'adx': 0.15, 'mfi': 0.10, 'atr': 0.0, 'macd': 0.0
        })
        weights_short = indicator_weights_short.get(symbol, {
            'rsi': 0.15, 'cci': 0.15, 'volume': 0.15, 'bb': 0.15, 'ma': 0.15, 'adx': 0.15, 'mfi': 0.10, 'atr': 0.0, 'macd': 0.0
        })
        
        # 各スコアをdf_5minに追加
        df_5min['rsi_score_long'] = rsi_score_long
        df_5min['rsi_score_short'] = rsi_score_short
        df_5min['cci_score_long'] = cci_score_long
        df_5min['cci_score_short'] = cci_score_short
        df_5min['volume_score'] = volume_score
        df_5min['bb_score_long'] = bb_score_long
        df_5min['bb_score_short'] = bb_score_short
        df_5min['ma_score_long'] = ma_score_long
        df_5min['ma_score_short'] = ma_score_short
        df_5min['adx_score_long'] = adx_score_long
        df_5min['adx_score_short'] = adx_score_short
        df_5min['mfi_score_long'] = mfi_score_long
        df_5min['mfi_score_short'] = mfi_score_short
        df_5min['atr_score_long'] = atr_score_long
        df_5min['atr_score_short'] = atr_score_short
        df_5min['macd_score_long'] = macd_score_long
        df_5min['macd_score_short'] = macd_score_short

        #volume_score = calculate_volume_score(df_5min)

        # 重み付き買いシグナルスコアにMACDを追加
        df_5min['buy_score'] = (
            rsi_score_long * weights_long['rsi'] +
            cci_score_long * weights_long['cci'] +
            volume_score * weights_long['volume'] +
            bb_score_long * weights_long['bb'] +
            ma_score_long * weights_long['ma'] +
            adx_score_long * weights_long['adx'] +
            mfi_score_long * weights_long['mfi'] +
            atr_score_long * weights_long['atr'] +
            macd_score_long * weights_long['macd']  # 追加
        )

        # ショートスコア（売り）
        df_5min['sell_score'] = (
            rsi_score_short * weights_short['rsi'] +
            cci_score_short * weights_short['cci'] +
            volume_score * weights_short['volume'] +
            bb_score_short * weights_short['bb'] +
            ma_score_short * weights_short['ma'] +
            adx_score_short * weights_short['adx'] +
            mfi_score_short * weights_short['mfi'] +
            atr_score_short * weights_short['atr'] +
            macd_score_short * weights_short['macd']  # 追加
        )

        df_5min['buy_score_scaled'] = rolling_minmax_scaler(df_5min['buy_score'], window=50)
        df_5min['sell_score_scaled'] = rolling_minmax_scaler(df_5min['sell_score'], window=50)

        # 現在の通貨ペアの閾値を取得
        thresholds = signal_thresholds.get(symbol, default_thresholds)
        buy_signal_threshold = thresholds['buy']
        sell_signal_threshold = thresholds['sell']

        # 閾値判定（既存のコードを置き換え）
        df_5min['buy_signal'] = df_5min['buy_score_scaled'] >= buy_signal_threshold
        df_5min['sell_signal'] = df_5min['sell_score_scaled'] >= sell_signal_threshold

        if self.exchange_settings_gmo.get("live_trade", False) or getattr(self, "is_backtest", False):
            if symbol == 'bcc_jpy':
                df_5min.loc[df_5min['adx_score_long'] < 0.00435, 'buy_signal'] = False
                df_5min.loc[df_5min['atr_score_long'].between(0.675, 0.95),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_short'] < 0.0594, 'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_long'] > 0.788, 'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_long'].between(0.0367, 0.0546),'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_long'].between(0.155, 0.173),'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_long'].between(0.186, 0.222),'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_short'].between(0.129, 0.152),'sell_signal'] = False
                df_5min.loc[df_5min['bb_score_long'] > 0.75, 'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.4, 0.48),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_short'] > 0.74, 'sell_signal'] = False
                df_5min.loc[df_5min['mfi_score_long'].between(0.1, 0.154),'buy_signal'] = False
                df_5min.loc[df_5min['mfi_score_long'].between(0.456, 0.536),'buy_signal'] = False
                df_5min.loc[df_5min['mfi_score_long'].between(0.565, 0.64),'buy_signal'] = False
                df_5min.loc[df_5min['mfi_score_long'] > 0.79, 'buy_signal'] = False
                df_5min.loc[df_5min['mfi_score_short'].between(0.657, 0.751),'sell_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.408, 0.454),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.475, 0.57),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'].between(0.205, 0.237),'sell_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'].between(0.33, 0.42),'sell_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'].between(0.556, 0.645),'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.14, 0.175),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.14, 0.175),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_short'].between(0.079, 0.136),'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'].between(0.226, 0.254),'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'].between(0.278, 0.337),'sell_signal'] = False

            if symbol == 'doge_jpy':
                df_5min.loc[df_5min['atr_score_short'].between(0.0385, 0.15),'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.625, 0.662),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_short'] < 0.03741, 'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_short'].between(0.1, 0.18),'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_short'].between(0.228, 0.297),'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_short'].between(0.33, 0.382),'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_short'].between(0.437, 0.582),'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_short'].between(0.822, 0.826),'sell_signal'] = False
                df_5min.loc[df_5min['ma_score_long'].between(0.151, 0.165),'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_short'].between(0.191, 0.225),'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'] > 0.933, 'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'].between(0.34, 0.35),'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_long'] > 0.997, 'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.143, 0.21),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.413, 0.444),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.468, 0.51),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.565, 0.605),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.13, 0.222),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.625, 0.66),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'] < 0.15, 'sell_signal'] = False
                df_5min.loc[df_5min['mfi_score_short'] > 0.6, 'sell_signal'] = False

            if symbol == 'sol_jpy':
                df_5min.loc[df_5min['atr_score_long'].between(0.0067, 0.0366),'buy_signal'] = False
                df_5min.loc[df_5min['atr_score_short'].between(0.41, 0.69),'sell_signal'] = False
                df_5min.loc[df_5min['atr_score_short'].between(0.045, 0.11),'sell_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.573, 0.61),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'] > 0.757, 'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'].between(0.12, 0.18),'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.114, 0.149),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.17, 0.264),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.444, 0.56),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.2, 0.25),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_short'] < 0.15, 'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'].between(0.3, 0.355),'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'].between(0.428, 0.496),'sell_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.2, 0.235),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.425, 0.49),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_short'].between(0.457, 0.552),'sell_signal'] = False
                df_5min.loc[df_5min['bb_score_short'] > 0.571, 'sell_signal'] = False
                df_5min.loc[df_5min['ma_score_long'].between(0.2, 0.223),'buy_signal'] = False
                df_5min.loc[df_5min['mfi_score_long'].between(0.466, 0.529),'buy_signal'] = False
                df_5min.loc[df_5min['mfi_score_short'].between(0.247, 0.273),'sell_signal'] = False

            if symbol == 'ada_jpy':
                df_5min.loc[df_5min['atr_score_long'].between(0.475, 0.59),'buy_signal'] = False
                df_5min.loc[df_5min['atr_score_short'].between(0.235, 0.296),'sell_signal'] = False
                df_5min.loc[df_5min['atr_score_short'].between(0.38, 0.527),'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.35, 0.63),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.8, 0.95),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_short'] < 0.0011, 'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_short'].between(0.108, 0.188),'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_short'].between(0.51, 0.645),'sell_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.105, 0.138),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.233, 0.286),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.57, 0.61),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.691, 0.743),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'] > 0.768, 'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'].between(0.247, 0.372),'sell_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.315, 0.37),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_short'] > 0.534, 'sell_signal'] = False
                df_5min.loc[df_5min['mfi_score_short'] > 0.62, 'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_long'] > 0.786, 'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.25, 0.296),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.33, 0.36),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.493, 0.547),'buy_signal'] = False            
                df_5min.loc[df_5min['cci_score_long'].between(0.57, 0.616),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_short'].between(0.446, 0.482),'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'].between(0.6, 0.685),'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'] > 0.786, 'sell_signal'] = False
                df_5min.loc[df_5min['ma_score_long'].between(0.13, 0.145),'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_long'].between(0.173, 0.2),'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_short'].between(0.2, 0.25),'sell_signal'] = False

            if symbol == 'ltc_jpy':
                df_5min.loc[df_5min['atr_score_long'].between(0.61, 0.727),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.20, 0.30),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.494, 0.535),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_long'].between(0.58, 0.663),'buy_signal'] = False
                df_5min.loc[df_5min['cci_score_short'] < 0.12, 'sell_signal'] = False
                df_5min.loc[df_5min['atr_score_short'].between(0.11, 0.225),'sell_signal'] = False
                df_5min.loc[df_5min['atr_score_short'].between(0.239, 0.408),'sell_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.143, 0.305),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.4, 0.409),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_long'] > 0.686, 'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_short'].between(0.566, 0.72),'sell_signal'] = False
                df_5min.loc[df_5min['ma_score_long'].between(0.165, 0.184),'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_short'].between(0.0369, 0.0806),'sell_signal'] = False
                df_5min.loc[df_5min['ma_score_short'].between(0.13, 0.162),'sell_signal'] = False
                df_5min.loc[df_5min['ma_score_short'] > 0.25, 'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.321, 0.395),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_short'].between(0.217, 0.305),'sell_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.3, 0.42),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.446, 0.485),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'].between(0.12, 0.21),'sell_signal'] = False
                df_5min.loc[df_5min['mfi_score_long'].between(0.15, 0.23),'buy_signal'] = False
                df_5min.loc[df_5min['mfi_score_short'].between(0.424, 0.484),'sell_signal'] = False
                df_5min.loc[df_5min['mfi_score_short'] > 0.71, 'sell_signal'] = False

            if symbol == 'eth_jpy':
                df_5min.loc[df_5min['bb_score_short'] > 0.551, 'sell_signal'] = False
                df_5min.loc[df_5min['mfi_score_long'].between(0.27, 0.29),'buy_signal'] = False
                df_5min.loc[df_5min['mfi_score_short'].between(0.285, 0.31),'sell_signal'] = False
                df_5min.loc[df_5min['ma_score_long'].between(0.0758, 0.09),'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_short'].between(0.15, 0.17),'sell_signal'] = False
                df_5min.loc[df_5min['atr_score_long'].between(0.0515, 0.139),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.4, 0.5),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.55, 0.575),'buy_signal'] = False
                df_5min.loc[df_5min['mfi_score_short'] > 0.54, 'sell_signal'] = False

            if symbol == 'xrp_jpy':
                df_5min.loc[df_5min['atr_score_long'].between(0.15, 0.228),'buy_signal'] = False
                df_5min.loc[df_5min['atr_score_short'].between(0.013, 0.124),'sell_signal'] = False
                df_5min.loc[df_5min['atr_score_short'].between(0.93, 0.989),'sell_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.31, 0.42),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_long'].between(0.575, 0.612),'buy_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'].between(0.228, 0.5),'sell_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'].between(0.572, 0.593),'sell_signal'] = False
                df_5min.loc[df_5min['rsi_score_short'].between(0.702, 0.724),'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'] < 0.084, 'sell_signal'] = False
                df_5min.loc[df_5min['cci_score_short'].between(0.61, 0.69),'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.0348, 0.1),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.382, 0.481),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_long'].between(0.66, 0.78),'buy_signal'] = False
                df_5min.loc[df_5min['adx_score_short'] < 0.155, 'sell_signal'] = False
                df_5min.loc[df_5min['adx_score_short'].between(0.7, 0.763),'sell_signal'] = False
                df_5min.loc[df_5min['ma_score_long'] > 0.293, 'buy_signal'] = False
                df_5min.loc[df_5min['ma_score_short'].between(0.032, 0.077),'sell_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.478, 0.573),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_long'].between(0.67, 0.896),'buy_signal'] = False
                df_5min.loc[df_5min['bb_score_short'] > 0.78, 'sell_signal'] = False

        self._apply_rule_thresholds(
            df_5min,
            symbol,
            timeframe="15m",
            version=getattr(self, "rules_version", None)  # 未定義なら None でOK
        )

        if 'EMA_long' in df_5min.columns and len(df_5min) > 1:
            prev_price = df_5min['close'].shift(1)
            prev_ma = df_5min['EMA_long'].shift(1)
            
            window = 2  # 過去3本で判定（調整可）

            # 過去window本すべてで価格 <= EMAなら買いシグナル無効化
            buy_mask = (prev_price <= prev_ma).rolling(window).apply(lambda x: x.all(), raw=True).astype(bool)
            df_5min.loc[buy_mask, 'buy_signal'] = False

            # 過去window本すべてで価格 >= EMAなら売りシグナル無効化
            sell_mask = (prev_price >= prev_ma).rolling(window).apply(lambda x: x.all(), raw=True).astype(bool)
            df_5min.loc[sell_mask, 'sell_signal'] = False

        return df_5min

    def _apply_rule_thresholds(self, df: pd.DataFrame, symbol: str, timeframe: str = "5m", version: str | None = None):
        """SIM限定：DBのしきい値ルールを適用し、ヒット件数をログ出力"""
        # SIMのみ適用（LIVEでは何もしない）
        try:
            if self.exchange_settings_gmo.get("live_trade", False):
                return
        except Exception:
            pass

        if df is None or df.empty:
            return

        from db import fetch_signal_rules
        rules = fetch_signal_rules(
            symbol,
            timeframe,
            version=version,
            user_id=1,  # 固定
            strategy_id='ST0001',  # 固定
            only_open_ended=True,
        )
        if not rules:
            #self.logger.info(f"[rules][{symbol}] no active rules (timeframe={timeframe}, version={version})")
            return

        total = len(df)
        union_mask = pd.Series(False, index=df.index)   # すべてのルールのOR（論理和）

        def _build_mask(s: pd.Series, op: str, v1, v2):
            """各ルールの条件をmaskに変換（NaNは不一致扱い）"""
            if op == "between" and v1 is not None and v2 is not None:
                return s.between(float(v1), float(v2), inclusive="both")
            elif op == "<":
                return s < float(v1)
            elif op == "<=":
                return s <= float(v1)
            elif op == ">":
                return s > float(v1)
            elif op == ">=":
                return s >= float(v1)
            elif op == "==":
                return s == float(v1)
            elif op == "!=":
                return s != float(v1)
            elif op == "is_null":
                return s.isna()
            elif op == "is_not_null":
                return s.notna()
            else:
                return pd.Series(False, index=s.index)

        # 各ルールのヒット数をinfoログ出力しながら適用
        for r in rules:
            col = r["score_col"]; op = r["op"]; trg = r["target_side"]; act = r.get("action", "disable")
            if act != "disable":
                continue
            if col not in df.columns:
                #self.logger.info(f"[rules][{symbol}] skip: missing column '{col}' for rule {r}")
                continue

            s = df[col]
            mask = _build_mask(s, op, r.get("v1"), r.get("v2"))
            hits = int(mask.sum())

            # ルールごとのヒット数を出力
            # 例: [rules][xrp_jpy] sell cci_score_short between 0.10~0.20 hits=12/480
            rng = f"{r.get('v1')}~{r.get('v2')}" if op == "between" else f"{r.get('v1')}"
            #self.logger.info(f"[rules][{symbol}] {trg} {col} {op} {rng} hits={hits}/{total}")

            # 実際の適用（buy/sell_signal を Falseに）
            if hits > 0:
                union_mask |= mask
                if trg == "buy" and "buy_signal" in df.columns:
                    df.loc[mask, "buy_signal"] = False
                if trg == "sell" and "sell_signal" in df.columns:
                    df.loc[mask, "sell_signal"] = False

        # ORユニオンのユニーク件数を出力
        union_hits = int(union_mask.sum())
        #self.logger.info(f"[rules][{symbol}] OR-union unique hits={union_hits}/{total}")

    def generate_signals_with_sentiment(self, symbol, df_5min, df_hourly):
        """市場センチメントを考慮したシグナル生成関数（改良版）
        RSI=0かつCCI=0の場合は1時間エントリー禁止条件を追加
        
        Parameters:
        symbol (str): 通貨ペア
        df_5min (pandas.DataFrame): 5分足データ
        df_hourly (pandas.DataFrame): 1時間足データ
        
        Returns:
        pandas.DataFrame: シグナルを追加したデータフレーム
        """
        
        # 基本的なシグナル生成を実行する前に、データの有効性チェック
        if df_5min.empty:
            self.logger.warning(f"{symbol}の5分足データが空です")
            return pd.DataFrame({'buy_signal': [], 'sell_signal': []})
        
        # 必要なカラムが不足している場合は早期リターン
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in df_5min.columns]
        if missing_columns:
            self.logger.warning(f"{symbol}のデータに基本価格列がありません: {missing_columns}")
            return df_5min
        
        # 基本シグナル生成
        df_5min = self.generate_signals(symbol, df_5min, df_hourly)
        
        try:
            
            # EMA乖離フィルターの設定（通貨ペアごとにカスタマイズ可能）
            ema_deviation_settings = {
                'ltc_jpy': {'max_deviation': 1.1},  # 5%を超える乖離でシグナル無効化
                'xrp_jpy': {'max_deviation': 1.1},  # 4%を超える乖離でシグナル無効化
                'eth_jpy': {'max_deviation': 1.1},  # 4.5%を超える乖離でシグナル無効化
                'sol_jpy': {'max_deviation': 1.1},  # 6%を超える乖離でシグナル無効化
                'doge_jpy': {'max_deviation': 1.1}, # 5.5%を超える乖離でシグナル無効化
                'bcc_jpy': {'max_deviation': 0.9},  # 4.5%を超える乖離でシグナル無効化
                'ada_jpy': {'max_deviation': 1.1},  # 4%を超える乖離でシグナル無効化
            }
            
            # デフォルト設定
            default_max_deviation = 1.0  # 5%
            
            # 通貨ペアの設定を取得
            max_deviation = ema_deviation_settings.get(symbol, {}).get('max_deviation', default_max_deviation)
            
            # EMAからの乖離率をチェック（全レコードに対して）
            if 'close' in df_5min.columns and 'EMA_long' in df_5min.columns:
                # 各レコードに対して処理
                disabled_signals_count = 0
                
                # 乖離率を計算
                df_5min['ema_deviation'] = abs(((df_5min['close'] / df_5min['EMA_long']) - 1) * 100)
                
                # 有効なEMA値を持つレコードのみ処理（ゼロ除算防止）
                valid_records = df_5min[df_5min['EMA_long'] > 0].index
                
                # シグナルの元の状態を保存（ログ出力用）
                original_buy_signals = df_5min['buy_signal'].copy()
                original_sell_signals = df_5min['sell_signal'].copy()
                
                # 乖離が大きすぎる場合はシグナルを無効化
                mask = df_5min['ema_deviation'] > max_deviation
                df_5min.loc[mask & valid_records, 'buy_signal'] = False
                df_5min.loc[mask & valid_records, 'sell_signal'] = False
                
                # 変更されたシグナルをカウント（ログ出力用）
                disabled_buy = ((original_buy_signals == True) & (df_5min['buy_signal'] == False)).sum()
                disabled_sell = ((original_sell_signals == True) & (df_5min['sell_signal'] == False)).sum()
                total_disabled = disabled_buy + disabled_sell
                
                # ログ出力
                if total_disabled > 0:
                    self.logger.info(
                        f"{symbol}: EMAからの乖離が大きすぎるため {total_disabled} 件のシグナルを無効化しました "
                        f"(買い: {disabled_buy}件, 売り: {disabled_sell}件, 乖離上限: {float(max_deviation):.2f}%)"
                    )
            
            # 最終確認: シグナルがNaNの場合はFalseで埋める
            if 'buy_signal' in df_5min.columns and pd.isna(df_5min['buy_signal']).any():
                df_5min['buy_signal'] = df_5min['buy_signal'].fillna(False)
            if 'sell_signal' in df_5min.columns and pd.isna(df_5min['sell_signal']).any():
                df_5min['sell_signal'] = df_5min['sell_signal'].fillna(False)

            return df_5min
        
        except Exception as e:
            self.logger.error(f"{symbol}のシグナル生成中にエラーが発生: {e}", exc_info=True)
            
            # エラー発生時のフォールバック処理
            if 'buy_signal' not in df_5min.columns:
                df_5min['buy_signal'] = False
            if 'sell_signal' not in df_5min.columns:
                df_5min['sell_signal'] = False
                
            return df_5min
    
    def adjust_position_risk(self, symbol, position_type, entry_price, current_capital):
        """ポジションのリスクを調整してサイズを決定（改良版）
        
        Parameters:
        symbol (str): 通貨ペア
        position_type (str): ポジションタイプ ('long' or 'short')
        entry_price (float): エントリー価格
        current_capital (float): 現在の運用資金
        
        Returns:
        float: 調整後の注文サイズ
        float: リスク調整後の注文額
        """
        # 基本注文額（通常のTRADE_SIZE）
        base_order_amount = self.TRADE_SIZE
        
        # 注文サイズを計算（通貨単位）
        order_size = base_order_amount / entry_price
        
        # 最小注文量を考慮
        order_size = self.adjust_order_size(symbol, order_size)
        
        # 最終的な注文額（サイズ調整後）
        # final_order_amount = order_size * entry_price
        
        self.logger.info(f"{symbol} {position_type}ポジションのリスク調整: "
                    f"基本額:{base_order_amount}円 → 調整後:{base_order_amount:.0f}円 ({base_order_amount/base_order_amount:.2f}倍), "
                    f"サイズ:{order_size:.6f}")
        
        return order_size, base_order_amount


    def backtest(self, days_to_test, live_mode=False):
        """複数通貨ペアのバックテスト実行（方式A: 直近2本True→次バー始値エントリー／スレッドセーフ）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        import pandas as pd
        from datetime import timedelta

        self.logger.info(f"=== バックテスト開始 ({days_to_test}日間) ===")

        results = {}
        start_profit = self.total_profit  # バックテスト開始時の利益

        # スレッドセーフのためのロックオブジェクト
        balance_lock = threading.Lock()
        trade_logs_lock = threading.Lock()
        entry_sizes_lock = threading.Lock()

        # 共有変数
        total_balance = self.initial_capital + self.total_profit
        trade_logs = []  # 全取引ログを格納

        # 銘柄ごとのシグナル蓄積用（リスト→後で結合）
        signal_rows = {symbol: [] for symbol in self.symbols}

        def run_backtest(symbol):
            nonlocal total_balance  # ← 外側スコープの total_balance を更新するために必要

            # 各通貨ペア専用の変数（スレッドローカル）
            total_trades = 0
            symbol_profit = 0
            wins = 0
            long_trades = 0
            long_wins = 0
            long_profit = 0
            short_trades = 0
            short_wins = 0
            short_profit = 0

            # このスレッド専用の取引ログ
            thread_trade_logs = []

            self.logger.info(f"=== {symbol.upper()} のバックテスト ===")

            # 日付範囲の設定（live_mode でも同一のロジック）
            day_range = range(days_to_test, 0, -1)

            for day_offset in day_range:
                current_date = now_jst() - timedelta(days=day_offset)
                date_str = current_date.strftime('%Y%m%d')

                # 前日
                previous_date = current_date - timedelta(days=1)
                previous_date_str = previous_date.strftime('%Y%m%d')

                # 15分足データの取得（当日分）
                df_5min_current = self.get_cached_data(symbol, '15min', date_str)
                if df_5min_current.empty:
                    self.logger.warning(f"{date_str}の{symbol}の15分足データを取得できませんでした。")
                    continue

                # 前日の15分足
                df_5min_previous = self.get_cached_data(symbol, '15min', previous_date_str)

                # 前日＋当日の結合→ソート→重複除去→特徴量
                if not df_5min_previous.empty:
                    df_5min_combined = pd.concat([df_5min_previous, df_5min_current])
                    df_5min_combined = df_5min_combined.sort_values('timestamp')
                    df_5min_combined = df_5min_combined.drop_duplicates(subset=['timestamp'])
                    self.logger.info(f"{symbol}の結合データ: 前日={len(df_5min_previous)}本 + 当日={len(df_5min_current)}本 = 合計{len(df_5min_combined)}本")
                    df_5min_full = self.build_features(df_5min_combined.copy())
                else:
                    self.logger.warning(f"{previous_date_str}の{symbol}の前日データが欠損のため、当日データのみで処理します。")
                    df_5min_full = self.build_features(df_5min_current.copy())

                # 1時間足データの取得（3日分）
                hourly_candles = []
                for h_offset in range(2, -1, -1):
                    hourly_date = (current_date - timedelta(days=h_offset)).strftime('%Y%m%d')
                    df_hourly_day = self.get_cached_data(symbol, '1hour', hourly_date)
                    if not df_hourly_day.empty:
                        hourly_candles.append(df_hourly_day)

                if not hourly_candles:
                    self.logger.warning(f"{date_str}の{symbol}の時間足データを取得できませんでした。")
                    continue

                expected_records = 96  # 24時間分の15分足（4本/時 × 24）

                self.is_backtest = True

                try:
                    df_hourly = pd.concat(hourly_candles).sort_values('timestamp')
                    df_hourly = self.build_features(df_hourly)

                    # シグナル生成（※この時点で buy/sell_signal が確定している想定）
                    df_5min_full = self.generate_signals_with_sentiment(symbol, df_5min_full, df_hourly)

                    # 時間順ソート→最新24h（96本）を抽出
                    if 'timestamp' in df_5min_full.columns:
                        df_5min_full = df_5min_full.sort_values('timestamp').reset_index(drop=True)
                        if len(df_5min_full) > expected_records:
                            df_5min = df_5min_full.iloc[-expected_records:].copy().reset_index(drop=True)
                            self.logger.info(f"シグナル生成後の最新24時間分データ: {len(df_5min)}本（合計{len(df_5min_full)}本から抽出）")
                        else:
                            df_5min = df_5min_full.copy().reset_index(drop=True)
                            self.logger.info(f"データ総数が96レコード未満のため全データを使用: {len(df_5min)}本")
                    else:
                        df_5min_full = df_5min_full.reset_index(drop=True)
                        if len(df_5min_full) > expected_records:
                            df_5min = df_5min_full.iloc[-expected_records:].copy().reset_index(drop=True)
                        else:
                            df_5min = df_5min_full.copy().reset_index(drop=True)
                        self.logger.warning(f"タイムスタンプカラムがないため、インデックスで最新データを抽出: {len(df_5min)}本")

                    if df_5min.empty:
                        self.logger.error(f"データ抽出後にデータフレームが空になりました: {symbol}")
                        continue

                    required_columns = ['buy_signal', 'sell_signal']
                    missing_columns = [c for c in required_columns if c not in df_5min.columns]
                    if missing_columns:
                        self.logger.error(f"必要なカラムが不足しています ({symbol}): {missing_columns}")
                        continue

                    # NaN 安全化
                    df_5min['buy_signal'] = df_5min['buy_signal'].fillna(False).astype(bool)
                    df_5min['sell_signal'] = df_5min['sell_signal'].fillna(False).astype(bool)

                    self.logger.info(f"データ抽出完了: {symbol}, 最終データ数: {len(df_5min)}")

                    # その日のシグナルを蓄積（Excel側での集計用）
                    if 'timestamp' in df_5min.columns:
                        sig_day = df_5min[['timestamp', 'buy_signal', 'sell_signal']].copy()
                    else:
                        sig_day = df_5min.reset_index().rename(columns={'index': 'timestamp'})[['timestamp', 'buy_signal', 'sell_signal']]
                    sig_day['timestamp'] = pd.to_datetime(sig_day['timestamp'], errors='coerce')
                    signal_rows[symbol].append(sig_day)

                    # バックテストの主要ループ
                    position = None
                    entry_price = 0.0
                    entry_time = None
                    entry_rsi = None
                    entry_cci = None
                    entry_sentiment = {}
                    entry_scores = {}
                    order_size = 0.0  # スレッドローカルな注文サイズ
                    reentry_block_until = None  # type: ignore  # pd.Timestamp | None

                    for i in range(len(df_5min)):
                        row = df_5min.iloc[i]
                        price_close = row.get('close', None)
                        price_open  = row.get('open', price_close)  # open欠損時はclose代用
                        timestamp   = row['timestamp'] if 'timestamp' in df_5min.columns else df_5min.index[i]

                        current_rsi = row.get('RSI', None)
                        current_cci = row.get('CCI', None)

                        # === エントリー判定（方式A：直近2本がTrue → 次バーで入る） ===
                        if position is None:
                            # 実運用同等：決済直後は「次の15分足始値」まで再エントリー禁止
                            if reentry_block_until is not None:
                                now_ts = pd.to_datetime(timestamp)
                                if now_ts < reentry_block_until:
                                    # クールダウン中はこのバーのエントリー判定をスキップ
                                    continue
                            # i-2, i-1（確定済バー）を見る。i は「約定バー」
                            if i >= 2:
                                prev1 = df_5min.iloc[i-1]
                                prev2 = df_5min.iloc[i-2]

                                prev_buy_1  = bool(prev1.get('buy_signal', False))
                                prev_buy_2  = bool(prev2.get('buy_signal', False))
                                prev_sell_1 = bool(prev1.get('sell_signal', False))
                                prev_sell_2 = bool(prev2.get('sell_signal', False))

                                # 直近2本で同方向 buy が点灯、かつ反対シグナルは消灯
                                cond_long = (prev_buy_1 and prev_buy_2 and not prev_sell_1 and not prev_sell_2)
                                cond_short = (prev_sell_1 and prev_sell_2 and not prev_buy_1 and not prev_buy_2)

                                # ロングエントリー
                                if cond_long:
                                    entry_price = price_open  # 次バーの始値で約定
                                    if entry_price is None or entry_price <= 0:
                                        # openもcloseも欠損の場合はスキップ
                                        continue

                                    order_size = self.TRADE_SIZE / entry_price
                                    order_size = self.adjust_order_size(symbol, order_size)
                                    entry_amount = order_size * entry_price

                                    position = 'long'
                                    entry_time = timestamp
                                    entry_rsi = prev1.get('RSI', None)  # 判定に使った直近確定バーの値を記録
                                    entry_cci = prev1.get('CCI', None)
                                    entry_sentiment = self.sentiment.copy() if hasattr(self, 'sentiment') else {}

                                    # スコア類も prev1 から記録
                                    entry_scores = {
                                        'buy_score_scaled': prev1.get('buy_score_scaled', 0),
                                        'sell_score_scaled': prev1.get('sell_score_scaled', 0),
                                        'rsi_score_long': prev1.get('rsi_score_long', 0),
                                        'rsi_score_short': prev1.get('rsi_score_short', 0),
                                        'cci_score_long': prev1.get('cci_score_long', 0),
                                        'cci_score_short': prev1.get('cci_score_short', 0),
                                        'volume_score': prev1.get('volume_score', 0),
                                        'bb_score_long': prev1.get('bb_score_long', 0),
                                        'bb_score_short': prev1.get('bb_score_short', 0),
                                        'ma_score_long': prev1.get('ma_score_long', 0),
                                        'ma_score_short': prev1.get('ma_score_short', 0),
                                        'adx_score_long': prev1.get('adx_score_long', 0),
                                        'adx_score_short': prev1.get('adx_score_short', 0),
                                        'mfi_score_long': prev1.get('mfi_score_long', 0),
                                        'mfi_score_short': prev1.get('mfi_score_short', 0),
                                        'atr_score_long': prev1.get('atr_score_long', 0),
                                        'atr_score_short': prev1.get('atr_score_short', 0),
                                        'macd_score_long': prev1.get('macd_score_long', 0),
                                        'macd_score_short': prev1.get('macd_score_short', 0),
                                        'ema_deviation': prev1.get('ema_deviation', 0),
                                        'entry_atr': prev1.get('ATR', 0),
                                        'entry_adx': prev1.get('ADX', 0)
                                    }

                                    with entry_sizes_lock:
                                        self.entry_sizes[symbol] = order_size

                                    with balance_lock:
                                        balance_before_entry = total_balance
                                        total_balance -= entry_amount
                                        balance_after_entry = total_balance

                                    self.log_entry(symbol, 'long', entry_price, entry_time, entry_rsi, entry_cci,
                                                prev1.get('ATR', 0), prev1.get('ADX', 0), entry_sentiment)

                                # ショートエントリー
                                elif cond_short:
                                    entry_price = price_open
                                    if entry_price is None or entry_price <= 0:
                                        continue

                                    order_size = self.TRADE_SIZE / entry_price
                                    order_size = self.adjust_order_size(symbol, order_size)
                                    entry_amount = order_size * entry_price

                                    position = 'short'
                                    entry_time = timestamp
                                    entry_rsi = prev1.get('RSI', None)
                                    entry_cci = prev1.get('CCI', None)
                                    entry_sentiment = self.sentiment.copy() if hasattr(self, 'sentiment') else {}

                                    entry_scores = {
                                        'buy_score_scaled': prev1.get('buy_score_scaled', 0),
                                        'sell_score_scaled': prev1.get('sell_score_scaled', 0),
                                        'rsi_score_long': prev1.get('rsi_score_long', 0),
                                        'rsi_score_short': prev1.get('rsi_score_short', 0),
                                        'cci_score_long': prev1.get('cci_score_long', 0),
                                        'cci_score_short': prev1.get('cci_score_short', 0),
                                        'volume_score': prev1.get('volume_score', 0),
                                        'bb_score_long': prev1.get('bb_score_long', 0),
                                        'bb_score_short': prev1.get('bb_score_short', 0),
                                        'ma_score_long': prev1.get('ma_score_long', 0),
                                        'ma_score_short': prev1.get('ma_score_short', 0),
                                        'adx_score_long': prev1.get('adx_score_long', 0),
                                        'adx_score_short': prev1.get('adx_score_short', 0),
                                        'mfi_score_long': prev1.get('mfi_score_long', 0),
                                        'mfi_score_short': prev1.get('mfi_score_short', 0),
                                        'atr_score_long': prev1.get('atr_score_long', 0),
                                        'atr_score_short': prev1.get('atr_score_short', 0),
                                        'macd_score_long': prev1.get('macd_score_long', 0),
                                        'macd_score_short': prev1.get('macd_score_short', 0),
                                        'ema_deviation': prev1.get('ema_deviation', 0),
                                        'entry_atr': prev1.get('ATR', 0),
                                        'entry_adx': prev1.get('ADX', 0)
                                    }

                                    with entry_sizes_lock:
                                        self.entry_sizes[symbol] = order_size

                                    with balance_lock:
                                        balance_before_entry = total_balance
                                        total_balance -= entry_amount
                                        balance_after_entry = total_balance

                                    self.log_entry(symbol, 'short', entry_price, entry_time, entry_rsi, entry_cci,
                                                prev1.get('ATR', 0), prev1.get('ADX', 0), entry_sentiment)

                        # === イグジット判定 ===
                        elif position == 'long':
                            # 先読みを避けるため、利用可能なデータは i までに限定
                            exit_levels = self.calculate_dynamic_exit_levels(symbol, df_5min.iloc[:i+1], 'long', entry_price)

                            tp = exit_levels['take_profit_price']
                            sl = exit_levels['stop_loss_price']
                            price = price_close if price_close is not None else price_open

                            # 旧仕様に合わせて「close 判定」のまま（OHLC貫通は別パッチで用意可能）
                            do_exit = (price >= tp) or (price <= sl)
                            if do_exit:
                                exit_price = tp if price >= tp else sl
                                profit = (exit_price - entry_price) / entry_price * self.TRADE_SIZE
                                profit_pct = (exit_price - entry_price) / entry_price * 100.0

                                entry_amount = order_size * entry_price
                                exit_amount = order_size * exit_price

                                with balance_lock:
                                    balance_before_exit = total_balance
                                    total_balance += exit_amount
                                    balance_after_exit = total_balance

                                # 保有時間
                                if isinstance(timestamp, pd.Timestamp) and isinstance(entry_time, pd.Timestamp):
                                    holding_time = timestamp - entry_time
                                    hours = holding_time.total_seconds() / 3600.0
                                else:
                                    hours = 0.0

                                exit_reason = "利益確定" if profit > 0 else "損切り"
                                self.log_exit(symbol, 'long', exit_price, entry_price, timestamp, profit, profit_pct,
                                            exit_reason, hours, entry_sentiment)

                                buy5, sell5 = self._last5_flags_str(df_5min, i)

                                trade_data = {
                                    'symbol': symbol,
                                    'type': 'long',
                                    'entry_price': entry_price,
                                    'entry_time': entry_time,
                                    'entry_rsi': entry_rsi,
                                    'entry_cci': entry_cci,
                                    'entry_atr': entry_scores.get('entry_atr', 0),
                                    'entry_adx': entry_scores.get('entry_adx', 0),
                                    'buy_score': entry_scores.get('buy_score_scaled', 0),
                                    'sell_score': entry_scores.get('sell_score_scaled', 0),
                                    'exit_price': exit_price,
                                    'exit_time': timestamp,
                                    'size': order_size,
                                    'entry_amount': entry_amount,
                                    'balance_after_entry': balance_after_entry,
                                    'balance_after_exit': balance_after_exit,
                                    'profit': profit,
                                    'profit_pct': profit_pct,
                                    'exit_reason': exit_reason,
                                    'holding_hours': hours,
                                    'exit_buy_last5': buy5,
                                    'exit_sell_last5': sell5,
                                    'sentiment_bullish': entry_sentiment.get('bullish', 0),
                                    'sentiment_bearish': entry_sentiment.get('bearish', 0),
                                    'sentiment_volatility': entry_sentiment.get('volatility', 0),
                                    'rsi_score_long': entry_scores.get('rsi_score_long', 0),
                                    'rsi_score_short': entry_scores.get('rsi_score_short', 0),
                                    'cci_score_long': entry_scores.get('cci_score_long', 0),
                                    'cci_score_short': entry_scores.get('cci_score_short', 0),
                                    'volume_score': entry_scores.get('volume_score', 0),
                                    'bb_score_long': entry_scores.get('bb_score_long', 0),
                                    'bb_score_short': entry_scores.get('bb_score_short', 0),
                                    'ma_score_long': entry_scores.get('ma_score_long', 0),
                                    'ma_score_short': entry_scores.get('ma_score_short', 0),
                                    'adx_score_long': entry_scores.get('adx_score_long', 0),
                                    'adx_score_short': entry_scores.get('adx_score_short', 0),
                                    'mfi_score_long': entry_scores.get('mfi_score_long', 0),
                                    'mfi_score_short': entry_scores.get('mfi_score_short', 0),
                                    'atr_score_long': entry_scores.get('atr_score_long', 0),
                                    'atr_score_short': entry_scores.get('atr_score_short', 0),
                                    'macd_score_long': entry_scores.get('macd_score_long', 0),
                                    'macd_score_short': entry_scores.get('macd_score_short', 0),
                                    'ema_deviation': entry_scores.get('ema_deviation', 0)
                                }
                                thread_trade_logs.append(trade_data)

                                # 統計更新
                                symbol_profit += profit
                                long_profit += profit
                                total_trades += 1
                                long_trades += 1
                                wins += (profit > 0)
                                long_wins += (profit > 0)

                                # リセット
                                position = None
                                entry_scores = {}

                                # 実運用同等：次の15分足始値まで再エントリー禁止
                                ts = pd.to_datetime(timestamp)
                                reentry_block_until = ts.floor('15min') + pd.Timedelta(minutes=15)

                        elif position == 'short':
                            exit_levels = self.calculate_dynamic_exit_levels(symbol, df_5min.iloc[:i+1], 'short', entry_price)

                            tp = exit_levels['take_profit_price']
                            sl = exit_levels['stop_loss_price']
                            price = price_close if price_close is not None else price_open

                            do_exit = (price <= tp) or (price >= sl)
                            if do_exit:
                                exit_price = tp if price <= tp else sl
                                profit = (entry_price - exit_price) / entry_price * self.TRADE_SIZE
                                profit_pct = (entry_price - exit_price) / entry_price * 100.0

                                entry_amount = order_size * entry_price

                                with balance_lock:
                                    balance_before_exit = total_balance
                                    # 既存仕様：ショートは entry_amount + profit を現金側に反映
                                    total_balance += (entry_amount + profit)
                                    balance_after_exit = total_balance

                                if isinstance(timestamp, pd.Timestamp) and isinstance(entry_time, pd.Timestamp):
                                    holding_time = timestamp - entry_time
                                    hours = holding_time.total_seconds() / 3600.0
                                else:
                                    hours = 0.0

                                exit_reason = "利益確定" if profit > 0 else "損切り"
                                self.log_exit(symbol, 'short', exit_price, entry_price, timestamp, profit, profit_pct,
                                            exit_reason, hours, entry_sentiment)

                                buy5, sell5 = self._last5_flags_str(df_5min, i)

                                trade_data = {
                                    'symbol': symbol,
                                    'type': 'short',
                                    'entry_price': entry_price,
                                    'entry_time': entry_time,
                                    'entry_rsi': entry_rsi,
                                    'entry_cci': entry_cci,
                                    'entry_atr': entry_scores.get('entry_atr', 0),
                                    'entry_adx': entry_scores.get('entry_adx', 0),
                                    'buy_score': entry_scores.get('buy_score_scaled', 0),
                                    'sell_score': entry_scores.get('sell_score_scaled', 0),
                                    'exit_price': exit_price,
                                    'exit_time': timestamp,
                                    'size': order_size,
                                    'entry_amount': entry_amount,
                                    'balance_after_entry': balance_after_entry,
                                    'balance_after_exit': balance_after_exit,
                                    'profit': profit,
                                    'profit_pct': profit_pct,
                                    'exit_reason': exit_reason,
                                    'holding_hours': hours,
                                    'exit_buy_last5': buy5,
                                    'exit_sell_last5': sell5,
                                    'sentiment_bullish': entry_sentiment.get('bullish', 0),
                                    'sentiment_bearish': entry_sentiment.get('bearish', 0),
                                    'sentiment_volatility': entry_sentiment.get('volatility', 0),
                                    'rsi_score_long': entry_scores.get('rsi_score_long', 0),
                                    'rsi_score_short': entry_scores.get('rsi_score_short', 0),
                                    'cci_score_long': entry_scores.get('cci_score_long', 0),
                                    'cci_score_short': entry_scores.get('cci_score_short', 0),
                                    'volume_score': entry_scores.get('volume_score', 0),
                                    'bb_score_long': entry_scores.get('bb_score_long', 0),
                                    'bb_score_short': entry_scores.get('bb_score_short', 0),
                                    'ma_score_long': entry_scores.get('ma_score_long', 0),
                                    'ma_score_short': entry_scores.get('ma_score_short', 0),
                                    'adx_score_long': entry_scores.get('adx_score_long', 0),
                                    'adx_score_short': entry_scores.get('adx_score_short', 0),
                                    'mfi_score_long': entry_scores.get('mfi_score_long', 0),
                                    'mfi_score_short': entry_scores.get('mfi_score_short', 0),
                                    'atr_score_long': entry_scores.get('atr_score_long', 0),
                                    'atr_score_short': entry_scores.get('atr_score_short', 0),
                                    'macd_score_long': entry_scores.get('macd_score_long', 0),
                                    'macd_score_short': entry_scores.get('macd_score_short', 0),
                                    'ema_deviation': entry_scores.get('ema_deviation', 0)
                                }
                                thread_trade_logs.append(trade_data)

                                # 統計更新
                                symbol_profit += profit
                                short_profit += profit
                                total_trades += 1
                                short_trades += 1
                                wins += (profit > 0)
                                short_wins += (profit > 0)

                                position = None
                                entry_scores = {}

                                # 実運用同等：次の15分足始値まで再エントリー禁止
                                ts = pd.to_datetime(timestamp)
                                reentry_block_until = ts.floor('15min') + pd.Timedelta(minutes=15)

                except Exception as e:
                    self.logger.error(f"{date_str}の{symbol}バックテスト中にエラー: {e}")
                    continue

            # スレッドローカルの取引ログをグローバルに統合
            with trade_logs_lock:
                trade_logs.extend(thread_trade_logs)

            # グローバル利益に加算（スレッドセーフ）
            with balance_lock:
                self.total_profit += symbol_profit

            # 結果の集計
            if total_trades > 0:
                win_rate = wins / total_trades * 100.0
                avg_profit = symbol_profit / total_trades

                self.logger.info(f"・トレード回数: {total_trades} 回（ロング: {long_trades}回、ショート: {short_trades}回）")
                self.logger.info(f"・勝率: {win_rate:.2f}%")
                if long_trades > 0:
                    self.logger.info(f"・ロング勝率: {(long_wins / long_trades * 100.0):.2f}%")
                if short_trades > 0:
                    self.logger.info(f"・ショート勝率: {(short_wins / short_trades * 100.0):.2f}%")
                self.logger.info(f"・平均利益: {avg_profit:.2f} 円")
                self.logger.info(f"・トータル利益: {symbol_profit:.2f} 円（ロング: {long_profit:.2f}円、ショート: {short_profit:.2f}円）")

                return {
                    'trades': total_trades,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'symbol_profit': symbol_profit,
                    'long_trades': long_trades,
                    'long_win_rate': (long_wins / long_trades * 100.0) if long_trades > 0 else 0.0,
                    'long_profit': long_profit,
                    'short_trades': short_trades,
                    'short_win_rate': (short_wins / short_trades * 100.0) if short_trades > 0 else 0.0,
                    'short_profit': short_profit
                }
            else:
                self.logger.info("😅 トレードが発生しませんでした")
                return {'trades': 0}

        # ===== マルチスレッディングでバックテストを並列実行 =====
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_backtest, symbol): symbol for symbol in self.symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    self.logger.error(f"{symbol}のバックテスト中にエラー: {e}")

        # 取引詳細の最終サマリー出力
        self.output_trade_summary(trade_logs)

        # シグナル履歴の作成（Excel用）
        signal_history = {}
        for sym, parts in signal_rows.items():
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)

            # timestamp 整理
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
            df = df.set_index('timestamp')

            # 必須2列のみ
            signal_history[sym] = df[['buy_signal', 'sell_signal']]

        # Excelファイルに取引ログを保存
        if trade_logs:
            try:
                self.save_trade_logs_to_excel(trade_logs, signal_history)  # ← 変更点：signal_history を渡す
            except Exception as e:
                self.logger.error(f"取引ログExcelの保存に失敗: {e}")

        # バックテスト全体結果
        backtest_profit = self.total_profit - start_profit
        self.logger.info(f"\n=== バックテスト全体結果 ===")
        self.logger.info(f"・バックテスト収益: {backtest_profit:,.2f} 円")

        # バックテスト結果をファイルに保存
        self.save_backtest_result(results, days_to_test, start_profit)

        # Excel評価レポートの自動生成
        if trade_logs:
            self.logger.info("Excel評価レポートを生成中...")
            try:
                excel_report_path = self._generate_excel_report_from_trade_logs(trade_logs, days_to_test)
                if excel_report_path:
                    self.logger.info(f"Excel評価レポートが生成されました: {excel_report_path}")
                    # OSごとの自動オープン（失敗しても致命ではない）
                    try:
                        import subprocess
                        import platform
                        if platform.system() == "Windows":
                            subprocess.run(['start', 'excel', excel_report_path], shell=True, check=False)
                        elif platform.system() == "Darwin":
                            subprocess.run(['open', excel_report_path], check=False)
                        else:
                            subprocess.run(['xdg-open', excel_report_path], check=False)
                        self.logger.info("Excelでレポートを開きました")
                    except:
                        self.logger.info("手動でExcelファイルを開いてください")
                else:
                    self.logger.warning("Excel評価レポートの生成に失敗しました")
            except Exception as e:
                self.logger.error(f"Excel評価レポート生成エラー: {str(e)}", exc_info=True)
        else:
            self.logger.info("取引データがないためExcel評価レポートは生成されませんでした")

        return results


    def log_entry(self, symbol, position_type, entry_price, entry_time, entry_rsi, entry_cci, entry_atr, entry_adx, entry_sentiment):
        """エントリー情報のログ出力"""
        # None値のチェックを追加
        rsi_str = f"{entry_rsi:.1f}" if entry_rsi is not None else "N/A"
        cci_str = f"{entry_cci:.1f}" if entry_cci is not None else "N/A"
        atr_str = f"{entry_atr:.2f}" if entry_atr is not None else "N/A"
        adx_str = f"{entry_adx:.1f}" if entry_adx is not None else "N/A"  # ADXを追加
        
        self.logger.info(f"[エントリー] {symbol} {'ロング' if position_type == 'long' else 'ショート'} @ {entry_price:.2f}円 (時刻: {entry_time})")
        self.logger.info(f"  → RSI: {rsi_str}, CCI: {cci_str}, ATR: {atr_str}, ADX: {adx_str}")  # ADXを追加
        self.logger.info(f"  → センチメント: 強気 {entry_sentiment.get('bullish', 0):.1f}%, 弱気 {entry_sentiment.get('bearish', 0):.1f}%, ボラティリティ {entry_sentiment.get('volatility', 0):.1f}%")

    def log_exit(self, symbol, position_type, exit_price, entry_price, exit_time, profit, profit_pct, exit_reason, hours, entry_sentiment):
        """イグジット情報のログ出力"""
        self.logger.info(f"[決済] {symbol} {'ロング' if position_type == 'long' else 'ショート'} @ {exit_price:.2f}円 (時刻: {exit_time})")
        self.logger.info(f"  → エントリー: {entry_price:.2f}円, 決済: {exit_price:.2f}円")
        self.logger.info(f"  → 損益: {profit:.2f}円 ({profit_pct:.2f}%), 理由: {exit_reason}")
        self.logger.info(f"  → 保有時間: {hours:.1f}時間")
        self.logger.info(f"  → センチメント(エントリー時): 強気 {entry_sentiment['bullish']:.1f}%, 弱気 {entry_sentiment['bearish']:.1f}%")

    def output_trade_summary(self, trade_logs):
        """取引詳細サマリーの出力"""
        self.logger.info("\n=== 取引詳細サマリー ===")
        for trade in trade_logs:
            self.logger.info(f"{trade['symbol']} {trade['type'].upper()}:")
            self.logger.info(f"  エントリー: {trade['entry_price']:.2f}円 ({trade['entry_time']})")
            self.logger.info(f"  決済: {trade['exit_price']:.2f}円 ({trade['exit_time']})")
            self.logger.info(f"  損益: {trade['profit']:.2f}円 ({trade['profit_pct']:.2f}%)")
            self.logger.info(f"  保有時間: {trade['holding_hours']:.1f}時間")
            self.logger.info("")

    def _last5_flags_str(self, df_5min, end_idx):
        def seq(col):
            if col in df_5min.columns:
                s = df_5min[col].iloc[max(0, end_idx-5):end_idx].fillna(False).astype(bool)
                return ''.join('1' if v else '0' for v in s.tolist()).rjust(5, '0')
            return None
        return seq('buy_signal'), seq('sell_signal')


    def save_trade_logs_to_excel(self, trade_logs, signal_history=None):
        """
        取引ログを通貨ペアごとにシート分割して Excel (.xlsx) で保存する
        追加シートとして各通貨ペアのロング/ショート統計情報も保存
        新機能：ポジション保有状況を横棒グラフで可視化
        """
        # 出力ファイル名を作成
        trade_log_file = os.path.join(
            self.log_dir,
            f'backtest_trades_{now_jst().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )

        # DataFrame化
        df_trades = pd.DataFrame(trade_logs)

        if 'symbol' not in df_trades.columns:
            raise ValueError("'symbol' 列が見つかりません。")

        desired_columns = [
            'symbol', 'type', 'entry_price', 'entry_time', 'entry_rsi', 'entry_cci', 
            'entry_atr', 'entry_adx',
            'ema_deviation',
            'exit_price', 'exit_time', 'size', 
            'profit', 'profit_pct', 'exit_reason', 'holding_hours', 
            'buy_score', 'sell_score',
            # 追加スコア情報
            'rsi_score_long', 'rsi_score_short',
            'cci_score_long', 'cci_score_short',
            'volume_score',
            'bb_score_long', 'bb_score_short',
            'ma_score_long', 'ma_score_short',
            'adx_score_long', 'adx_score_short',
            'mfi_score_long', 'mfi_score_short',
            'atr_score_long', 'atr_score_short',
            'macd_score_long', 'macd_score_short',
            'exit_buy_last5', 'exit_sell_last5'

        ]
        
        # 必要な列が存在しない場合は追加
        for col in desired_columns:
            if col not in df_trades.columns:
                df_trades[col] = None
        
        # 列の順序を整える
        available_columns = [col for col in desired_columns if col in df_trades.columns]
        df_trades = df_trades[available_columns]

        # ExcelWriterで'symbol'ごとにシート書き込み
        with pd.ExcelWriter(trade_log_file, engine="xlsxwriter") as writer:
            # 1. 各通貨ペアごとのシート
            for symbol, grp in df_trades.groupby('symbol'):
                sheet_name = str(symbol)[:31]  # シート名は31文字まで
                grp.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 2. サマリーシートの追加
            # 集計用のデータフレームを準備
            if not df_trades.empty:
                # 各通貨ペアとポジションタイプ（long/short）ごとの統計を計算
                summary_data = []
                
                # 各通貨ペアごとの統計
                for symbol in df_trades['symbol'].unique():
                    symbol_df = df_trades[df_trades['symbol'] == symbol]
                    
                    # ロングとショートに分割
                    long_df = symbol_df[symbol_df['type'] == 'long']
                    short_df = symbol_df[symbol_df['type'] == 'short']
                    
                    # ロングの統計
                    long_trades = len(long_df)
                    long_profit = long_df['profit'].sum() if not long_df.empty else 0
                    long_wins = (long_df['profit'] > 0).sum() if not long_df.empty else 0
                    long_losses = (long_df['profit'] <= 0).sum() if not long_df.empty else 0
                    long_win_rate = (long_wins / long_trades * 100) if long_trades > 0 else 0
                    long_avg_profit = long_df['profit'].mean() if not long_df.empty else 0
                    long_avg_win = long_df.loc[long_df['profit'] > 0, 'profit'].mean() if not long_df.empty and (long_df['profit'] > 0).any() else 0
                    long_avg_loss = long_df.loc[long_df['profit'] <= 0, 'profit'].mean() if not long_df.empty and (long_df['profit'] <= 0).any() else 0
                    
                    # ショートの統計
                    short_trades = len(short_df)
                    short_profit = short_df['profit'].sum() if not short_df.empty else 0
                    short_wins = (short_df['profit'] > 0).sum() if not short_df.empty else 0
                    short_losses = (short_df['profit'] <= 0).sum() if not short_df.empty else 0
                    short_win_rate = (short_wins / short_trades * 100) if short_trades > 0 else 0
                    short_avg_profit = short_df['profit'].mean() if not short_df.empty else 0
                    short_avg_win = short_df.loc[short_df['profit'] > 0, 'profit'].mean() if not short_df.empty and (short_df['profit'] > 0).any() else 0
                    short_avg_loss = short_df.loc[short_df['profit'] <= 0, 'profit'].mean() if not short_df.empty and (short_df['profit'] <= 0).any() else 0
                    
                    # 合計の統計
                    total_trades = long_trades + short_trades
                    total_profit = long_profit + short_profit
                    total_wins = long_wins + short_wins
                    total_losses = long_losses + short_losses
                    total_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
                    
                    # データの追加
                    summary_data.append({
                        'Symbol': symbol,
                        'Position Type': 'Long',
                        'Trades': long_trades,
                        'Win Rate (%)': long_win_rate,
                        'Total Profit': long_profit,
                        'Avg Profit': long_avg_profit,
                        'Avg Win': long_avg_win,
                        'Avg Loss': long_avg_loss
                    })
                    
                    summary_data.append({
                        'Symbol': symbol,
                        'Position Type': 'Short',
                        'Trades': short_trades,
                        'Win Rate (%)': short_win_rate,
                        'Total Profit': short_profit,
                        'Avg Profit': short_avg_profit,
                        'Avg Win': short_avg_win,
                        'Avg Loss': short_avg_loss
                    })
                    
                    summary_data.append({
                        'Symbol': symbol,
                        'Position Type': 'Total',
                        'Trades': total_trades,
                        'Win Rate (%)': total_win_rate,
                        'Total Profit': total_profit,
                        'Avg Profit': total_profit / total_trades if total_trades > 0 else 0,
                        'Avg Win': None,  # 総合の場合は個別の勝ち負けの平均は計算しない
                        'Avg Loss': None
                    })
                
                # サマリーデータフレームの作成
                df_summary = pd.DataFrame(summary_data)
                
                # サマリーシートに書き込み
                df_summary.to_excel(writer, sheet_name='統計サマリー', index=False)
                
                # ワークブックとワークシートのオブジェクトを取得
                workbook = writer.book
                summary_sheet = writer.sheets['統計サマリー']
                
                # 数値フォーマットの設定
                money_format = workbook.add_format({'num_format': '¥#,##0'})
                percent_format = workbook.add_format({'num_format': '0.00%'})
                
                # カラムの幅を設定
                summary_sheet.set_column('A:A', 10)  # Symbol
                summary_sheet.set_column('B:B', 12)  # Position Type
                summary_sheet.set_column('C:C', 8)   # Trades
                summary_sheet.set_column('D:D', 12)  # Win Rate
                summary_sheet.set_column('E:E', 15, money_format)  # Total Profit
                summary_sheet.set_column('F:F', 12, money_format)  # Avg Profit
                summary_sheet.set_column('G:G', 12, money_format)  # Avg Win
                summary_sheet.set_column('H:H', 12, money_format)  # Avg Loss
                
                # 3. 各通貨ペアごとの総利益チャートシートの追加
                # 通貨ペアごとの総利益を集計
                symbol_profits = df_trades.groupby('symbol')['profit'].sum().reset_index()
                symbol_profits = symbol_profits.sort_values('profit', ascending=False)
                
                # チャートシートを作成
                chart_sheet = writer.book.add_worksheet('利益チャート')
                
                # データをシートに書き込み
                chart_sheet.write_row('A1', ['Symbol', 'Total Profit'])
                for i, (symbol, profit) in enumerate(zip(symbol_profits['symbol'], symbol_profits['profit'])):
                    chart_sheet.write(i+1, 0, symbol)
                    chart_sheet.write(i+1, 1, profit)
                
                # チャート作成
                chart = workbook.add_chart({'type': 'column'})
                chart.add_series({
                    'name': 'Total Profit',
                    'categories': ['利益チャート', 1, 0, len(symbol_profits), 0],
                    'values': ['利益チャート', 1, 1, len(symbol_profits), 1],
                    'data_labels': {'value': True},
                })
                
                chart.set_title({'name': '通貨ペアごとの総利益'})
                chart.set_y_axis({'name': '利益（円）'})
                chart.set_x_axis({'name': '通貨ペア'})
                chart.set_size({'width': 720, 'height': 400})
                chart_sheet.insert_chart('D2', chart)
                
                # 4. ロング/ショートの比較チャートの追加
                # ロング/ショートの利益を集計
                position_summary = []
                for symbol in df_trades['symbol'].unique():
                    symbol_df = df_trades[df_trades['symbol'] == symbol]
                    
                    # ロングとショートに分割して集計
                    long_profit = symbol_df[symbol_df['type'] == 'long']['profit'].sum()
                    short_profit = symbol_df[symbol_df['type'] == 'short']['profit'].sum()
                    
                    position_summary.append({
                        'Symbol': symbol,
                        'Long Profit': long_profit,
                        'Short Profit': short_profit
                    })
                
                # ポジションサマリーデータフレームの作成
                df_position = pd.DataFrame(position_summary)
                
                # ポジションサマリーシートの作成
                position_sheet = writer.book.add_worksheet('ロングショート比較')
                
                # データをシートに書き込み
                position_sheet.write_row('A1', ['Symbol', 'Long Profit', 'Short Profit'])
                for i, row in enumerate(df_position.itertuples(index=False)):
                    position_sheet.write(i+1, 0, row.Symbol)
                    # 修正: カラム名に '_' が入るとタプルでのアクセスが変わるため、位置でアクセスする
                    position_sheet.write(i+1, 1, row[1])  # Long Profit は2番目の列 (インデックス1)
                    position_sheet.write(i+1, 2, row[2])  # Short Profit は3番目の列 (インデックス2)
                
                # チャート作成
                position_chart = workbook.add_chart({'type': 'column'})
                position_chart.add_series({
                    'name': 'Long Profit',
                    'categories': ['ロングショート比較', 1, 0, len(df_position), 0],
                    'values': ['ロングショート比較', 1, 1, len(df_position), 1],
                    'data_labels': {'value': True},
                })
                
                position_chart.add_series({
                    'name': 'Short Profit',
                    'categories': ['ロングショート比較', 1, 0, len(df_position), 0],
                    'values': ['ロングショート比較', 1, 2, len(df_position), 2],
                    'data_labels': {'value': True},
                })
                
                position_chart.set_title({'name': '通貨ペアごとのロング/ショート利益比較'})
                position_chart.set_y_axis({'name': '利益（円）'})
                position_chart.set_x_axis({'name': '通貨ペア'})
                position_chart.set_size({'width': 720, 'height': 400})
                position_sheet.insert_chart('D2', position_chart)

                # 5. 修正版：総合タイムライン（ポジションタイムラインシートは削除）
                self._create_overall_timeline_chart_fixed(writer, df_trades, signal_history)

        self.logger.info(f"取引ログを Excel に保存しました: {trade_log_file}")

    def _create_overall_timeline_chart_fixed(self, writer, df_trades, signal_history=None):
        """
        全通貨ペアのタイムラインを表示
        1ブロック(3行) = [ポジション, buy_signal, sell_signal]
        列は15分単位
        """
        try:
            workbook = writer.book
            overall_sheet = workbook.add_worksheet('総合タイムライン')

            # === 時間範囲（15分刻み） ===
            all_times = []
            for _, tr in df_trades.iterrows():
                if pd.notna(tr.get('entry_time')) and pd.notna(tr.get('exit_time')):
                    all_times.extend([pd.to_datetime(tr['entry_time']),
                                    pd.to_datetime(tr['exit_time'])])
            if not all_times:
                self.logger.warning("総合タイムライン作成：有効な時間データがありません")
                return

            min_time = min(all_times).floor('15min')
            max_time = max(all_times).ceil('15min')
            total_quarters = int((max_time - min_time).total_seconds() // 900) + 1  # 900秒=15分
            max_display_quarters = min(total_quarters, 8800)  # 列数上限

            symbols = sorted(df_trades['symbol'].unique())

            # === ヘッダ ===
            headers = ['通貨/種別'] + [
                (min_time + pd.Timedelta(minutes=15*i)).strftime("%m-%d %H:%M")
                for i in range(max_display_quarters)
            ]
            for col, h in enumerate(headers):
                overall_sheet.write(0, col, h)

            # === フォーマット定義 ===
            fmt_long  = workbook.add_format({'bg_color': '#9AD9A1'})   # 濃い緑
            fmt_short = workbook.add_format({'bg_color': '#F28B82'})   # 濃い赤
            fmt_buy   = workbook.add_format({'bg_color': '#D7ECD9'})   # 薄緑
            fmt_sell  = workbook.add_format({'bg_color': '#FDE2E1'})   # 薄赤
            fmt_gray  = workbook.add_format({'font_color': '#999999'}) # 数字グレー

            # === 事前に行列を用意 ===
            def quarter_idx(ts):
                return int((ts - min_time).total_seconds() // 900)

            pos_matrix  = {s: [0]*max_display_quarters for s in symbols}
            buy_matrix  = {s: [0]*max_display_quarters for s in symbols}
            sell_matrix = {s: [0]*max_display_quarters for s in symbols}

            # === ポジション埋め ===
            for _, tr in df_trades.iterrows():
                if pd.isna(tr.get('entry_time')) or pd.isna(tr.get('exit_time')):
                    continue
                s = tr['symbol']
                start = max(0, min(max_display_quarters-1,
                                quarter_idx(pd.to_datetime(tr['entry_time']))))
                end   = max(0, min(max_display_quarters-1,
                                quarter_idx(pd.to_datetime(tr['exit_time']))))
                pv = 1 if tr.get('type') == 'long' else -1
                for q in range(start, end+1):
                    pos_matrix[s][q] = pv

            # === シグナル埋め ===
            if signal_history:
                for s in symbols:
                    df_sig = signal_history.get(s)
                    if df_sig is None or not {'buy_signal', 'sell_signal'} <= set(df_sig.columns):
                        continue
                    df_sig = df_sig.copy()
                    if not isinstance(df_sig.index, pd.DatetimeIndex):
                        if 'timestamp' in df_sig.columns:
                            df_sig.set_index(pd.to_datetime(df_sig['timestamp']), inplace=True)
                        else:
                            continue
                    # 15分単位に resample
                    quarterly = (
                        df_sig[['buy_signal', 'sell_signal']]
                        .astype(bool)
                        .resample('15min', label='left', closed='left')
                        .max()
                        .fillna(False)
                    )
                    for q in range(max_display_quarters):
                        bucket_start = min_time + pd.Timedelta(minutes=15*q)
                        if bucket_start in quarterly.index:
                            if bool(quarterly.at[bucket_start, 'buy_signal']):
                                buy_matrix[s][q] = 1
                            if bool(quarterly.at[bucket_start, 'sell_signal']):
                                sell_matrix[s][q] = 1

            # === シート出力 ===
            for i, s in enumerate(symbols):
                row_base = 1 + i*3
                overall_sheet.write(row_base + 0, 0, f'{s} (position)')
                overall_sheet.write(row_base + 1, 0, f'{s} buy_signal')
                overall_sheet.write(row_base + 2, 0, f'{s} sell_signal')

                # 値書き込み
                for q in range(max_display_quarters):
                    overall_sheet.write(row_base + 0, 1+q, pos_matrix[s][q])
                    overall_sheet.write(row_base + 1, 1+q, buy_matrix[s][q])
                    overall_sheet.write(row_base + 2, 1+q, sell_matrix[s][q])

                # 条件付き書式（色付け）
                overall_sheet.conditional_format(row_base + 0, 1, row_base + 0, max_display_quarters,
                                                {'type': 'cell', 'criteria': '==', 'value': 1,  'format': fmt_long})
                overall_sheet.conditional_format(row_base + 0, 1, row_base + 0, max_display_quarters,
                                                {'type': 'cell', 'criteria': '==', 'value': -1, 'format': fmt_short})
                overall_sheet.conditional_format(row_base + 1, 1, row_base + 1, max_display_quarters,
                                                {'type': 'cell', 'criteria': '==', 'value': 1,  'format': fmt_buy})
                overall_sheet.conditional_format(row_base + 2, 1, row_base + 2, max_display_quarters,
                                                {'type': 'cell', 'criteria': '==', 'value': 1,  'format': fmt_sell})

                # 数字をグレーに（任意）
                overall_sheet.conditional_format(row_base, 1, row_base+2, max_display_quarters,
                                                {'type': 'cell', 'criteria': '>=', 'value': -1, 'format': fmt_gray})

            # 先頭列の幅を広げる
            overall_sheet.set_column(0, 0, 22)

        except Exception as e:
            self.logger.error(f"総合タイムライン作成エラー: {e}", exc_info=True)

    def save_backtest_result(self, results, days_to_test, start_profit):
        """バックテスト結果をJSONファイルに保存"""
        backtest_profit = self.total_profit - start_profit
        backtest_result_file = os.path.join(self.log_dir, f'backtest_result_{now_jst().strftime("%Y%m%d_%H%M%S")}.json')
        try:
            with open(backtest_result_file, 'w') as f:
                json.dump({
                    'start_date': (now_jst() - timedelta(days=days_to_test)).strftime("%Y-%m-%d"),
                    'end_date': now_jst().strftime("%Y-%m-%d"),
                    'days_tested': days_to_test,
                    'total_profit': backtest_profit,
                    'results': {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv 
                                for kk, vv in v.items()} 
                            for k, v in results.items()}
                }, f, indent=4)
            self.logger.info(f"バックテスト結果を保存しました: {backtest_result_file}")
        except Exception as e:
            self.logger.error(f"バックテスト結果保存エラー: {e}")

    def run_live(self):
        """
        リアルタイムトレーディングモード（backtest関数との整合性が取れた本番環境最適化版）
        - バックテスト関数と同じ判断プロセスを使用
        - APIエラーハンドリングの強化
        - 資金と取引サイズの動的最適化
        - 詳細なレポート生成と監視
        """
        self.logger.info("=== 本番環境最適化版リアルタイムトレーディング開始 ===")
    
        # 初期化
        start_date = now_jst()
        start_balance = 0
        trade_logs = []
        stats = {
            'total_trades': 0,
            'total_wins': 0,
            'total_losses': 0,
            'total_profit': 0,
            'total_loss': 0,
            'daily_trades': 0,
            'daily_wins': 0,
            'daily_losses': 0,
            'daily_profit': 0,
            'daily_loss': 0,
            'last_report_time': dt.datetime.now(JST)
        }
        
        # 起動メッセージ送信
        startup_message = (
            "🚀 トレードボット起動\n"
            f"📊 対象通貨ペア: {', '.join(self.symbols)}\n"
            f"💰 運用資金: {self.initial_capital:,}円\n"
            f"🔧 テストモード: {'有効' if self.test_mode else '無効'}\n"
            f"📆 開始時刻: {start_date.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_notification("トレードボット起動", startup_message, "info")

        try:
            # API接続確認
            if not self._verify_api_connection():
                self.logger.critical("API接続テストに失敗しました。設定を確認してから再起動してください。")
                return
                
            # 取引ログ用のリスト（バックテストと同様）
            trade_logs = []

            # 起動時にポジション検証を実行
            self.verify_positions()

            # 起動時の資産状況を取得
            try:
                start_balance = self.get_total_balance()
                self.logger.info(f"起動時資産: {start_balance:,.0f}円")
            except Exception as e:
                self.logger.error(f"起動時資産取得エラー: {e}")
                start_balance = self.initial_capital
                self.logger.info(f"起動時資産取得エラーのため初期資金を使用: {start_balance:,.0f}円")

            # カウンター変数
            position_verify_counter = 0
            sentiment_update_counter = 0
            health_check_counter = 0
            
            # エラー管理
            error_states = {
                'consecutive_api_errors': 0,
                'consecutive_data_errors': 0,
                'last_successful_cycle': now_jst(),
                'error_log': []
            }

            # バックテストにあわせてlast_sentiment_timeを初期化
            self.last_sentiment_time = None

            # メインループ - より堅牢なエラーハンドリングを追加
            while True:
                loop_start_time = now_jst()
                
                try:
                    # 現在の日本時間を取得
                    jst_now = dt.datetime.now(JST)
                    
                    # システム健全性チェック
                    health_check_counter += 1
                    if health_check_counter >= 60:  # 5時間ごと(60 * 5分)
                        self._perform_health_check()
                        health_check_counter = 0
                    
                    # 定期的なポジション検証
                    self.logger.info("定期ポジション検証を実行します")
                    self.verify_positions()
                    
                    # バックアップと増資チェック（1日1回）
                    if jst_now.hour == 0 and jst_now.minute < 5:  # 深夜0時台の最初のサイクル
                        self.logger.info("日次バックアップと増資チェックを実行します")
                        self.check_backup_needed()
                        self.check_monthly_increase()
                    
                    last_rep = stats.get('last_report_time')
                    try:
                        last_rep_jst = last_rep.astimezone(JST)
                    except Exception:
                        # naive 対策：JST とみなす（必要に応じて UTC 扱いに変更）
                        last_rep_jst = last_rep.replace(tzinfo=JST)

                    # 日次レポート送信（日本時間の深夜0時）
                    if (jst_now.hour == 0 and jst_now.minute < 5 and last_rep_jst.date() != jst_now.date()):
                        self._send_daily_report(stats)
                        # レポート送信後に日次変数をリセット
                        stats['last_report_time'] = jst_now
                        stats['daily_trades'] = 0
                        stats['daily_wins'] = 0
                        stats['daily_losses'] = 0
                        stats['daily_profit'] = 0
                        stats['daily_loss'] = 0
                    
                    # 全体の資金状況を表示
                    total_balance = self.get_total_balance()
                    self.logger.info(f"\n=== 資金状況: {total_balance:,.0f} 円（日本時間: {jst_now.strftime('%Y-%m-%d %H:%M:%S')}）===")
                    self.logger.info(f"開始資金: {start_balance:,.0f} 円 / 運用利益: {self.total_profit:,.2f} 円 / 評価額: {total_balance:,.0f} 円")
                    
                    # 現在のポジション情報を表示
                    self._display_current_positions()
                        
                    # 各通貨ペアを順番に処理（バックテストと同様、同期処理にして確実性を高める）
                    for symbol in self.symbols:
                        try:
                            self.logger.info(f"\n{symbol}の分析開始...")
                            
                            # データ取得（バックテストと同じデータ取得方法を使用）
                            df_5min, df_hourly = self._get_market_data(symbol)
                            if df_5min is None or df_hourly is None:
                                continue
                            
                            # 特徴量計算
                            df_5min = self.build_features(df_5min)
                            df_hourly = self.build_features(df_hourly)
                            
                            # 処理中のDataFrameを一時的に保存 - バックテストと同様
                            self.df_5min = df_5min
                            
                            # シグナル生成（センチメント考慮版）
                            df_5min = self.generate_signals_with_sentiment(symbol, df_5min, df_hourly)

                            df_5min = df_5min.sort_values('timestamp').reset_index(drop=True)
                            if len(df_5min) > 96:
                                df_5min = df_5min.iloc[-96:].copy().reset_index(drop=True)
                            
                            # 最新のシグナル情報を取得
                            latest_signals = df_5min.iloc[-2]
                            previous_signals = df_5min.iloc[-3]
                            
                            # 現在のポジション状態を確認
                            position = self.positions.get(symbol)
                            
                            # バックテストと同様の手順でポジションを判断
                            
                            # ポジションがない場合のエントリー判断
                            if position is None:
                                # ★ ここから追加: 再エントリー解禁時刻のチェック
                                unlock_time = self.reentry_block_until.get(symbol)
                                if unlock_time is not None and now_jst() < unlock_time:
                                    remain = (unlock_time - now_jst()).total_seconds()
                                    self.logger.info(
                                        f"{symbol}: 決済直後のクールダウン中（次の15分足待ち）。残り {remain:.0f} 秒は新規エントリー判定をスキップ"
                                    )
                                    # このシンボルの残処理をスキップ
                                    continue

                                else:
                                    # 解禁済みなら通常のエントリー判定
                                    if latest_signals.get('buy_signal', False) and previous_signals.get('buy_signal', False):
                                        self._handle_entry(symbol, 'long', latest_signals, stats, trade_logs)
                                        # エントリーできたのでブロック情報は不要
                                        self.reentry_block_until[symbol] = None
                                    elif latest_signals.get('sell_signal', False) and previous_signals.get('sell_signal', False):
                                        self._handle_entry(symbol, 'short', latest_signals, stats, trade_logs)
                                        self.reentry_block_until[symbol] = None

                            
                            # ポジションがある場合のイグジット判断
                            elif position is not None:
                                self._check_exit_conditions(symbol, stats, trade_logs, df_5min)
                            
                            # DataFrameの参照を削除（バックテストと同様）
                            if hasattr(self, 'df_5min'):
                                del self.df_5min
                            
                        except Exception as e:
                            self.logger.error(f"{symbol}の処理中にエラーが発生: {str(e)}", exc_info=True)
                            error_states['error_log'].append(f"{now_jst()}: {symbol}処理エラー - {str(e)}")
                            continue
                    
                    # メインループサイクル間のスリープ（5分間隔）
                    elapsed_time = (now_jst() - loop_start_time).total_seconds()
                    sleep_time = max(28 - elapsed_time, 1)  # 少なくとも1秒は待機
                    self.logger.info(f"次のサイクルまで{sleep_time:.1f}秒待機します")
                    
                    # 成功したのでエラーカウンターをリセット
                    error_states['consecutive_api_errors'] = 0
                    error_states['consecutive_data_errors'] = 0
                    error_states['last_successful_cycle'] = now_jst()
                    
                    # スリープ中に1分ごとにキーボード割り込みをチェック
                    sleep_interval = min(60, sleep_time)
                    sleep_count = int(sleep_time / sleep_interval)
                    
                    for _ in range(sleep_count):
                        time.sleep(sleep_interval)
                    
                    # 残りの時間を待機
                    remain = sleep_time - (sleep_interval * sleep_count)
                    if remain > 0:
                        time.sleep(remain)
                    
                except requests.exceptions.RequestException as e:
                    # APIリクエスト関連のエラー
                    error_states['consecutive_api_errors'] += 1
                    error_message = f"APIリクエストエラー: {str(e)}"
                    self.logger.error(error_message)
                    error_states['error_log'].append(f"{now_jst()}: {error_message}")
                    
                    # APIエラーが連続して発生した場合の対応
                    if error_states['consecutive_api_errors'] >= 3:
                        self.logger.critical(f"連続{error_states['consecutive_api_errors']}回のAPIエラーが発生しました")
                        # 通知送信
                        self.send_notification(
                            "API接続エラー警告", 
                            f"連続{error_states['consecutive_api_errors']}回のAPIエラーが発生しました。ネットワーク接続と認証情報を確認してください。", 
                            "error"
                        )
                        # より長い待機時間を設定（徐々に増加）
                        wait_time = min(300 * error_states['consecutive_api_errors'], 1800)  # 最大30分
                        self.logger.info(f"APIエラー発生のため{wait_time}秒待機します")
                        time.sleep(wait_time)
                    else:
                        # 通常の待機
                        self.logger.info("APIエラー発生のため30秒待機します")
                        time.sleep(30)
                
                except Exception as e:
                    # その他の一般的なエラー
                    error_states['consecutive_data_errors'] += 1
                    error_message = f"実行中にエラーが発生: {str(e)}"
                    self.logger.error(error_message, exc_info=True)
                    error_states['error_log'].append(f"{now_jst()}: {error_message}")
                    
                    # エラーが連続して発生した場合の対応
                    if error_states['consecutive_data_errors'] >= 5:
                        self.logger.critical(f"連続{error_states['consecutive_data_errors']}回のエラーが発生しました")
                        
                        # 重大なエラーの場合はメール通知
                        self.send_notification(
                            "システムエラー警告", 
                            f"連続{error_states['consecutive_data_errors']}回のエラーが発生しました。\n最新のエラー: {str(e)}", 
                            "error"
                        )
                        
                        # より長い待機時間を設定
                        wait_time = min(120 * error_states['consecutive_data_errors'], 900)  # 最大15分
                        self.logger.info(f"エラー発生のため{wait_time}秒待機します")
                        time.sleep(wait_time)
                        
                        # 連続エラーが10回以上発生した場合は安全のため一時停止
                        if error_states['consecutive_data_errors'] >= 10:
                            self.logger.critical("連続エラーが10回以上発生したため、取引を一時停止します。管理者の確認が必要です。")
                            self.send_notification(
                                "取引一時停止", 
                                "連続エラーが10回以上発生したため、取引を一時停止しました。ログを確認し、問題を解決してからボットを再起動してください。", 
                                "error"
                            )
                            
                            # エラーログをファイルに保存
                            self._save_error_log(error_states['error_log'])
                            return  # プログラムを終了
                    else:
                        # 通常の待機
                        self.logger.info("エラー発生のため60秒待機します")
                        time.sleep(60)
                        
                    # 最後の成功から3時間以上経過している場合は通知
                    time_since_last_success = (now_jst() - error_states['last_successful_cycle']).total_seconds() / 3600
                    if time_since_last_success >= 3:
                        self.logger.warning(f"最後の成功から{time_since_last_success:.1f}時間経過しています")
                        self.send_notification(
                            "長時間エラー発生", 
                            f"最後の成功から{time_since_last_success:.1f}時間が経過しています。システムを確認してください。", 
                            "error"
                        )
                        # エラーカウンタをリセット（通知の連続発生を防止）
                        error_states['consecutive_data_errors'] = 1
        
        except KeyboardInterrupt:
            self.logger.info("キーボード割り込みを受け取りました。プログラムを安全に終了します。")
        
        finally:
            self.is_backtest = False
            # 稼働状況のレポートを出力
            self._generate_final_report(start_date, start_balance, stats, trade_logs)
            
            # 取引ログをファイルに保存
            # Excel出力は本番（live）のみ実行。SIM時はスキップして落ちないようにする
            if self.exchange_settings_gmo.get('live_trade'):
                try:
                    self.save_trade_logs_to_excel(trade_logs)
                except Exception as e:
                    self.logger.warning(f"Excel出力に失敗しました（処理を継続します）: {e}")
            
            # 進行中のポジション情報を保存
            self.save_positions()
            
            self.logger.info("プログラムを終了します。")

    def _verify_api_connection(self):
        """APIへの接続を検証する"""
        self.logger.info("API接続検証を実行中...")    # paper(=sim) モードでは API 接続テストをスキップ
        if not self.exchange_settings_gmo.get("live_trade", False):
            self.logger.warning("Paper mode: API接続テストをスキップします（APIキー未設定でも続行）")
            return True

        self.logger.info("API接続検証を実行中...")     

        try:
            # API認証情報の確認
            if not self.exchange_settings_gmo.get('api_key') or not self.exchange_settings_gmo.get('api_secret'):
                self.logger.error("API認証情報が設定されていません")
                return False
            
            # 資産残高の取得を試みる
            for symbol in self.symbols:
                base_coin = symbol.split('_')[0]
                balance = self.get_balance(base_coin)
                self.logger.info(f"{base_coin}の残高取得テスト: {balance}")
                # 通貨ごとに少なくとも1つは成功すればOK
                if balance is not None:
                    return True
                    
            self.logger.error("API接続テストに失敗しました")
            return False
        
        except Exception as e:
            self.logger.error(f"API接続検証中にエラーが発生: {str(e)}")
            return False

    def _display_current_positions(self):
        """現在のポジション情報を表示する"""
        self.logger.info("\n現在のポジション:")
        has_positions = False
        
        for symbol in self.symbols:
            position = self.positions.get(symbol)
            if position:
                has_positions = True
                entry_price = self.entry_prices[symbol]
                current_price = self.get_current_price(symbol)
                if current_price > 0:
                    # 損益計算
                    if position == 'long':
                        profit_pct = (current_price / entry_price - 1) * 100
                    else:  # short
                        profit_pct = (entry_price / current_price - 1) * 100
                    
                    # 保有時間の計算
                    if self.entry_times[symbol]:
                        holding_time = dt.datetime.now(JST) - self.entry_times[symbol]
                        hours = holding_time.total_seconds() / 3600
                        
                        # 利益状態に応じて表示を変える
                        if profit_pct > 0:
                            status = "利益中"
                        else:
                            status = "損失中"
                            
                        self.logger.info(f"{symbol}: {position} ({profit_pct:+.2f}%, 現在価格: {current_price:,.2f}), {status}, 保有時間: {hours:.1f}時間")
                else:
                    self.logger.info(f"{symbol}: {position} (現在価格取得エラー)")
            else:
                self.logger.info(f"{symbol}: ポジションなし")
        
        if not has_positions:
            self.logger.info("現在ポジションはありません")

    def _get_market_data(self, symbol):
        """
        市場データを取得する（15分足: 前日＋当日＋必要に応じて前々日を結合し、timestampで一意化）
        ※ここでは「96本へのトリムは行わない」。この後段（build_features → シグナル判定の後）で
        バックテストと同じタイミングでトリムする想定。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (15分足データ, 1時間足データ)
                - 15分足: timestampでソート・重複排除済み
                - 1時間足: 直近3日を結合、timestampでソート・重複排除済み
            取得不能時は (None, None)
        """
        try:
            current_date = now_jst()

            # ========= 15分足（直近最大3日分を結合 → ソート → 重複排除。ここではトリムしない） =========
            self.logger.info(f"{symbol}の15分足データ取得を開始します（前日＋当日＋必要に応じて前々日を結合）")

            fifteen_parts = []
            for day_offset in (1, 0, 2):  # 優先順: 前日 → 当日 → 前々日（不足時の補完）
                d = current_date - timedelta(days=day_offset)
                dstr = d.strftime('%Y%m%d')
                df_day = self.get_cached_data(symbol, '15min', dstr)
                if df_day is None or df_day.empty:
                    continue

                # timestamp 正規化（列が無い場合は index を昇格）
                if 'timestamp' not in df_day.columns:
                    df_day = df_day.reset_index()
                    # 汎用的に index→timestamp へ
                    if 'index' in df_day.columns:
                        df_day = df_day.rename(columns={'index': 'timestamp'})

                df_day['timestamp'] = pd.to_datetime(df_day['timestamp'], errors='coerce', utc=False)
                df_day = df_day.dropna(subset=['timestamp'])
                df_day = df_day.sort_values('timestamp')

                fifteen_parts.append(df_day)

            if not fifteen_parts:
                self.logger.warning(f"{symbol}: 15分足データを取得できませんでした")
                return None, None

            df_15m = pd.concat(fifteen_parts, ignore_index=True)
            # timestamp で整列 → 重複排除（後勝ち＝最新の再計算値を残す）
            df_15m = df_15m.sort_values('timestamp')
            before = len(df_15m)
            df_15m = df_15m.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
            after = len(df_15m)
            if before != after:
                self.logger.info(f"{symbol} 15分足: 重複 {before - after} 件を排除、最終 {after} 本")

            # ========= 1時間足（直近3日分を結合 → ソート → 重複排除） =========
            hourly_parts = []
            for h_offset in range(2, -1, -1):
                d = current_date - timedelta(days=h_offset)
                dstr = d.strftime('%Y%m%d')
                df_h = self.get_cached_data(symbol, '1hour', dstr)
                if df_h is None or df_h.empty:
                    continue

                if 'timestamp' not in df_h.columns:
                    df_h = df_h.reset_index()
                    if 'index' in df_h.columns:
                        df_h = df_h.rename(columns={'index': 'timestamp'})

                df_h['timestamp'] = pd.to_datetime(df_h['timestamp'], errors='coerce', utc=False)
                df_h = df_h.dropna(subset=['timestamp']).sort_values('timestamp')
                hourly_parts.append(df_h)

            if hourly_parts:
                df_hourly = pd.concat(hourly_parts, ignore_index=True)
                df_hourly = df_hourly.sort_values('timestamp')
                before_h = len(df_hourly)
                df_hourly = df_hourly.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
                after_h = len(df_hourly)
                if before_h != after_h:
                    self.logger.info(f"{symbol} 1時間足: 重複 {before_h - after_h} 件を排除、最終 {after_h} 本")
            else:
                self.logger.warning(f"{symbol}: 1時間足データを取得できませんでした")
                df_hourly = pd.DataFrame()

            self.logger.info(f"{symbol}の15分足データ: 合計 {len(df_15m)} 本 / 1時間足データ: 合計 {len(df_hourly)} 本")
            return df_15m, df_hourly

        except Exception as e:
            self.logger.error(f"{symbol}のデータ取得中にエラーが発生: {str(e)}", exc_info=True)
            return None, None

    def _build_entry_explanation(self, symbol, position_type, signal_data, price):
        def _f(x, nd=None):
            try:
                return float(x) if x is not None else nd
            except Exception:
                return nd

        th = getattr(self, "entry_thresholds", None) or {
            "adx_trend_min": 25.0,
            "rsi_long_min": 58.0,
            "rsi_short_max": 42.0,
            "score_long_min": 0.55,
            "score_short_min": 0.55,
            "ema_cross_required": True,
        }
        # 入力スナップショット（None安全化）
        rsi  = _f(signal_data.get('RSI'))
        adx  = _f(signal_data.get('ADX'))
        atr  = _f(signal_data.get('ATR'))
        di_p = _f(signal_data.get('plus_di14'))
        di_m = _f(signal_data.get('minus_di14'))
        ema_fast = _f(signal_data.get('EMA_short'))
        ema_slow = _f(signal_data.get('EMA_long'))
        buy_score  = _f(signal_data.get('buy_score'))
        sell_score = _f(signal_data.get('sell_score'))

        # 使ったしきい値（実際に方向で使うキーだけを詰めると良い）
        thresholds_dict = {
            "adx_trend_min": th["adx_trend_min"],
            "ema_cross_required": th["ema_cross_required"],
        }
        if position_type == "long":
            thresholds_dict.update({
                "rsi_min": th["rsi_long_min"],
                "score_min": th["score_long_min"],
            })
        else:
            thresholds_dict.update({
                "rsi_max": th["rsi_short_max"],
                "score_min": th["score_short_min"],
            })

        # 判定根拠（シンプルかつ再現可能に）
        checks = []
        if adx is not None:
            checks.append(f"ADX={adx:.1f}≥{th['adx_trend_min']}" if adx >= th["adx_trend_min"] else f"ADX={adx:.1f}<{th['adx_trend_min']}")
        if rsi is not None:
            if position_type == "long":
                checks.append(f"RSI={rsi:.1f}≥{th['rsi_long_min']}")
            else:
                checks.append(f"RSI={rsi:.1f}≤{th['rsi_short_max']}")
        if ema_fast is not None and ema_slow is not None and th["ema_cross_required"]:
            cross_ok = (ema_fast > ema_slow) if position_type == "long" else (ema_fast < ema_slow)
            checks.append(f"EMA{'>' if cross_ok else '×'} (fast={ema_fast:.1f}, slow={ema_slow:.1f})")

        score_used = buy_score if position_type == "long" else sell_score
        if score_used is not None:
            th_key = "score_long_min" if position_type == "long" else "score_short_min"
            checks.append(f"score={score_used:.3f}≥{th[th_key]}")

        # 一文説明（通知・DBで読みやすく）
        dir_ja = "ロング" if position_type == "long" else "ショート"
        explanation_text = (
            f"{symbol.upper()} {dir_ja}エントリー: "
            + "; ".join(c for c in checks if c) +
            (f"; price={price:.2f}" if price else "")
        )

        # signal_raw：生情報＋使った設定・特徴量
        signal_raw = {
            "features": {
                "rsi": rsi, "adx": adx, "atr": atr,
                "di_plus": di_p, "di_minus": di_m,
                "ema_fast": ema_fast, "ema_slow": ema_slow,
                "buy_score": buy_score, "sell_score": sell_score,
            },
            "thresholds": thresholds_dict,
            "notes": {
                "position_type": position_type,
                "ema_cross_required": th["ema_cross_required"],
            }
        }
        return thresholds_dict, explanation_text, signal_raw

    def _handle_entry(self, symbol, position_type, signal_data, stats, trade_logs):
        """エントリー処理を実行する（backtest関数と整合性あり）
        
        Parameters:
        symbol (str): 通貨ペア
        position_type (str): ポジションタイプ ('long' or 'short')
        current_price (float): 現在価格
        signal_data (pandas.Series): シグナルデータ
        stats (dict): 統計情報の辞書
        trade_logs (list): 取引ログのリスト
        """

        def _f(x, nd=None):
            try:
                return float(x) if x is not None else nd
            except Exception:
                return nd

        # テクニカル指標値の取得
        entry_rsi = signal_data.get('RSI', None)
        entry_cci = signal_data.get('CCI', None)
        entry_atr = signal_data.get('ATR', None)
        entry_adx = signal_data.get('ADX', None)
        entry_di_plus = signal_data.get('plus_di14', None)
        entry_di_minus = signal_data.get('minus_di14', None)
        ema_fast = signal_data.get('EMA_short', None)
        ema_slow = signal_data.get('EMA_long', None)

        # スコア情報の取得（新規追加）
        buy_score = signal_data.get('buy_score', 0)
        sell_score = signal_data.get('sell_score', 0)

        # EMA乖離率の取得（新規追加）
        ema_deviation = signal_data.get('ema_deviation', 0)

        current_price = self.get_current_price(symbol)
        
        # リスク調整済みの取引サイズを計算
        order_size, order_amount = self.adjust_position_risk(
            symbol, 
            position_type, 
            current_price, 
            self.initial_capital + self.total_profit
        )
        
        # 資金チェック
        if not self.check_sufficient_funds(
            symbol, 
            'buy' if position_type == 'long' else 'sell', 
            order_size, 
            current_price
        ):
            self.logger.warning(f"{symbol}の{position_type}エントリーに必要な資金が不足しています")
            return
        
        # 注文実行
        self.logger.info(f"{symbol}の{position_type}エントリー注文を実行します: 価格 {current_price:.2f}, サイズ {order_size:.3f}, 金額 {order_amount:.0f}円")
        
        current_price = self.get_current_price(symbol)

        # None安全化
        entry_rsi = _f(signal_data.get('RSI'))
        entry_adx = _f(signal_data.get('ADX'))
        entry_atr = _f(signal_data.get('ATR'))
        entry_di_plus  = _f(signal_data.get('plus_di14'))
        entry_di_minus = _f(signal_data.get('minus_di14'))
        ema_fast = _f(signal_data.get('EMA_short'))
        ema_slow = _f(signal_data.get('EMA_long'))
        buy_score  = _f(signal_data.get('buy_score'))
        sell_score = _f(signal_data.get('sell_score'))
        ema_deviation = _f(signal_data.get('ema_deviation'))

        # 説明・しきい値・生情報
        thresholds_dict, explanation_text, signal_raw = self._build_entry_explanation(
            symbol=symbol,
            position_type=position_type,
            signal_data=signal_data,
            price=current_price
        )

        # バージョン
        version = self._current_strategy_version()  # 上で例示

        thresholds_dict, explanation_text, signal_raw = self._build_entry_explanation(
            symbol, position_type, signal_data, price=current_price
        )

        order_result = self.execute_order_with_confirmation(
            symbol,
            'buy' if position_type == 'long' else 'sell',
            order_size,
            timeframe="15m",
            strength_score=float(signal_data.get('buy_score', 0) if position_type=='long' else signal_data.get('sell_score', 0)),
            rsi=float(signal_data.get('RSI'))           if signal_data.get('RSI') is not None else None,
            adx=float(signal_data.get('ADX'))           if signal_data.get('ADX') is not None else None,
            atr=float(signal_data.get('ATR'))           if signal_data.get('ATR') is not None else None,
            di_plus=float(signal_data.get('plus_di14')) if signal_data.get('plus_di14') is not None else None,
            di_minus=float(signal_data.get('minus_di14')) if signal_data.get('minus_di14') is not None else None,
            ema_fast=float(signal_data.get('EMA_short')) if signal_data.get('EMA_short') is not None else None,
            ema_slow=float(signal_data.get('EMA_long'))  if signal_data.get('EMA_long')  is not None else None,
            strategy_id=getattr(self, "strategy_id", "v1_weighted_signals"),
            version=self._current_strategy_version(),   # ← ハッシュ付きで日付+短縮hashを自動付与
            signal_raw=signal_raw,                      # ← 使った値と閾値のスナップショット
        )

        if order_result['success']:
            executed_size = order_result.get('executed_size', 0)
            # 実際に約定したサイズで金額を再計算
            final_order_amount = executed_size * current_price
            
            # ポジション情報を更新
            self.positions[symbol] = position_type
            self.entry_prices[symbol] = current_price
            self.entry_times[symbol] = now_jst()
            self.entry_sizes[symbol] = executed_size

            # GMOコインの場合、position_idを取得して保存
            time.sleep(3)  # 注文が約定するまで待機
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
            positions_response = self._get_margin_positions_safe(gmo_symbol)
            
            if positions_response.get("status") == 0:
                positions_data = positions_response.get("data", {})
                positions = positions_data.get("list", []) if isinstance(positions_data, dict) else positions_data
                
                # 最新のポジションを探す（同じsymbolで最新のもの）
                for pos in positions:
                    if pos.get("symbol") == gmo_symbol:
                        # 既にposition_idがある場合はスキップ
                        if symbol not in self.position_ids or self.position_ids[symbol] is None:
                            self.position_ids[symbol] = pos.get("positionId")
                            self.logger.info(f"新規ポジションID保存: {symbol} = {self.position_ids[symbol]}")
                            break
            
            # エントリー時の市場センチメントを保存（backtest関数と整合性を取る）
            entry_sentiment = self.sentiment.copy() if hasattr(self, 'sentiment') else {}
            
            # backtest関数と同様のログ出力
            self.log_entry(symbol, position_type, current_price, now_jst(), entry_rsi, entry_cci, entry_atr, entry_adx, entry_sentiment)
            
            # 通知送信（修正箇所）
            if self.notification_settings.get('send_on_entry', True):
                # --- df_5min を安全に取得（signal_dataに無ければキャッシュ等から） ---
                df_5min = None
                # 1) シグナル生成側で _df_5min を埋めているケース
                if isinstance(signal_data, dict):
                    df_5min = signal_data.get('_df_5min')
                # 2) ボット内のキャッシュに持っているケース（環境に合わせて調整）
                if df_5min is None and hasattr(self, 'df_5min_cache'):
                    df_5min = self.df_5min_cache.get(symbol)
                if df_5min is None and hasattr(self, 'latest_df_5min'):
                    df_5min = getattr(self, 'latest_df_5min', {}).get(symbol)

                # --- TP/SLを算出（df_5minが無ければNoneで送る） ---
                tp_level = sl_level = None
                try:
                    if df_5min is not None and len(df_5min) > 0:
                        exit_levels = self.calculate_dynamic_exit_levels(
                            symbol=symbol,
                            df_5min=df_5min,
                            position_type=position_type,          # long/short
                            entry_price=current_price              # いま入った価格
                        ) if 'position_type' in self.calculate_dynamic_exit_levels.__code__.co_varnames else \
                        self.calculate_dynamic_exit_levels(symbol, df_5min, position_type, current_price)
                        tp_level = float(exit_levels.get('take_profit_price')) if exit_levels else None
                        sl_level = float(exit_levels.get('stop_loss_price')) if exit_levels else None
                except Exception as e:
                    self.logger.warning(f"TP/SL算出に失敗（entry通知は継続）: {e}")

                # --- 根拠タグ（最低限） ---
                reason_tags = []
                # トレンド・ボラ・オシレーターの簡易タグ化
                try:
                    last = None
                    if df_5min is not None and len(df_5min) > 0:
                        last = df_5min.iloc[-1]
                    # 値源は signal_data を優先、無ければ df_5min の最新から
                    rsi = float(entry_rsi) if entry_rsi is not None else (
                        float(last["RSI"]) if (last is not None and "RSI" in df_5min.columns and not pd.isna(last["RSI"])) else None
                    )
                    adx = float(entry_adx) if entry_adx is not None else (
                        float(last["ADX"]) if (last is not None and "ADX" in df_5min.columns and not pd.isna(last["ADX"])) else None
                    )
                    atr = float(entry_atr) if entry_atr is not None else (
                        float(last["ATR"]) if (last is not None and "ATR" in df_5min.columns and not pd.isna(last["ATR"])) else None
                    )

                    # SMAは環境の列名に合わせて調整（SMA_FAST/SMA_SLOW or SMA20/SMA50 など）
                    sma_fast = None
                    sma_slow = None
                    for fast_key in ("SMA_FAST","SMA20","SMA_20"):
                        if fast_key in (signal_data.keys() if isinstance(signal_data, dict) else []):
                            sma_fast = float(signal_data.get(fast_key)); break
                    if sma_fast is None and last is not None:
                        for fast_key in ("SMA_FAST","SMA20","SMA_20"):
                            if fast_key in df_5min.columns and not pd.isna(last.get(fast_key)):
                                sma_fast = float(last.get(fast_key)); break

                    for slow_key in ("SMA_SLOW","SMA50","SMA_50"):
                        if slow_key in (signal_data.keys() if isinstance(signal_data, dict) else []):
                            sma_slow = float(signal_data.get(slow_key)); break
                    if sma_slow is None and last is not None:
                        for slow_key in ("SMA_SLOW","SMA50","SMA_50"):
                            if slow_key in df_5min.columns and not pd.isna(last.get(slow_key)):
                                sma_slow = float(last.get(slow_key)); break

                    # タグ付け
                    if adx is not None and adx >= 25:
                        reason_tags.append("adx_trend")
                    if atr is not None and current_price:
                        atr_pct = 100.0 * atr / float(current_price)
                        if atr_pct >= 1.6:
                            reason_tags.append("atr_wide")
                    if sma_fast is not None and sma_slow is not None:
                        reason_tags.append("sma_bull" if sma_fast > sma_slow else "sma_bear")
                    if rsi is not None:
                        if rsi <= 30: reason_tags.append("rsi_oversold")
                        elif rsi >= 70: reason_tags.append("rsi_overbought")
                    # 方向タグ
                    reason_tags.append("long_entry" if position_type == "long" else "short_entry")
                    # 複数根拠なら confluence
                    if len(reason_tags) >= 3:
                        reason_tags.append("confluence")
                except Exception as e:
                    self.logger.warning(f"reason_tags生成に失敗: {e}")
                    reason_tags = reason_tags or []

                # --- 指標スナップショットを作成 ---
                ind = IndicatorSnapshot(
                    rsi = float(entry_rsi) if entry_rsi is not None else None,
                    adx = float(entry_adx) if entry_adx is not None else None,
                    atr = float(entry_atr) if entry_atr is not None else None,
                    sma_fast = sma_fast,
                    sma_slow = sma_slow,
                    price = float(current_price) if current_price is not None else None,
                    timeframe = "5m"
                )

                # --- 通知サイド（BUY/SELL） ---
                side_for_notice = "BUY" if position_type == "long" else "SELL"

                # --- エントリー時の総合スコア（あれば使用） ---
                entry_total_score = None
                if isinstance(self.entry_scores.get(symbol, {}), dict):
                    entry_total_score = self.entry_scores[symbol].get("entry_total_score")
                if entry_total_score is None:
                    # 無ければ buy_score/sell_score のどちらかを自然に採用
                    entry_total_score = float(buy_score) if position_type == "long" else float(sell_score)

                # --- コンテキストを組んで送信 ---
                ctx = SignalContext(
                    symbol=symbol,
                    side=side_for_notice,
                    reason_tags=reason_tags,
                    tp=tp_level,
                    sl=sl_level,
                    score=entry_total_score
                )
                try:
                    body = compose_signal_message(ctx, ind)
                    self.send_notification(
                        subject=f"{symbol.upper()} {side_for_notice}",
                        message=body
                    )
                except Exception as e:
                    self.logger.exception(f"ENTRY通知失敗: {e}")

            # スコア値を保存
            self.entry_scores[symbol] = {
                'entry_atr': entry_atr,
                'entry_adx': entry_adx,
                'rsi_score_long': signal_data.get('rsi_score_long', 0),
                'rsi_score_short': signal_data.get('rsi_score_short', 0),
                'cci_score_long': signal_data.get('cci_score_long', 0),
                'cci_score_short': signal_data.get('cci_score_short', 0),
                'volume_score': signal_data.get('volume_score', 0),
                'bb_score_long': signal_data.get('bb_score_long', 0),
                'bb_score_short': signal_data.get('bb_score_short', 0),
                'ma_score_long': signal_data.get('ma_score_long', 0),
                'ma_score_short': signal_data.get('ma_score_short', 0),
                'adx_score_long': signal_data.get('adx_score_long', 0),
                'adx_score_short': signal_data.get('adx_score_short', 0),
                'mfi_score_long': signal_data.get('mfi_score_long', 0),
                'mfi_score_short': signal_data.get('mfi_score_short', 0),
                'atr_score_long': signal_data.get('atr_score_long', 0),
                'atr_score_short': signal_data.get('atr_score_short', 0),
                'macd_score_long': signal_data.get('macd_score_long', 0), 
                'macd_score_short': signal_data.get('macd_score_short', 0),
                'ema_deviation': ema_deviation
            }
            
            # 取引ログを記録する部分で、スコア情報を追加
            trade_log_entry = {
                'symbol': symbol,
                'type': position_type,
                'action': 'entry',
                'price': current_price,
                'size': executed_size,
                'amount': final_order_amount,
                'time': now_jst(),
                'rsi': entry_rsi,
                'cci': entry_cci,
                'entry_atr': entry_atr,
                'entry_adx': entry_adx,
                'ema_deviation': ema_deviation,
                'sentiment': entry_sentiment,
                'buy_score': buy_score,  # これで buy_score_scaled が記録される
                'sell_score': sell_score,  # これで sell_score_scaled が記録される
                # その他スコア情報...
                'rsi_score_long': signal_data.get('rsi_score_long', 0),
                'rsi_score_short': signal_data.get('rsi_score_short', 0),
                'cci_score_long': signal_data.get('cci_score_long', 0),
                'cci_score_short': signal_data.get('cci_score_short', 0),
                'volume_score': signal_data.get('volume_score', 0),
                'bb_score_long': signal_data.get('bb_score_long', 0),
                'bb_score_short': signal_data.get('bb_score_short', 0),
                'ma_score_long': signal_data.get('ma_score_long', 0),
                'ma_score_short': signal_data.get('ma_score_short', 0),
                'adx_score_long': signal_data.get('adx_score_long', 0),
                'adx_score_short': signal_data.get('adx_score_short', 0),
                'mfi_score_long': signal_data.get('mfi_score_long', 0),
                'mfi_score_short': signal_data.get('mfi_score_short', 0),
                'atr_score_long': signal_data.get('atr_score_long', 0),
                'atr_score_short': signal_data.get('atr_score_short', 0),
                'macd_score_long': signal_data.get('macd_score_long', 0), 
                'macd_score_short': signal_data.get('macd_score_short', 0) 
            }
            trade_logs.append(trade_log_entry)
            
            # ポジション保存
            self.save_positions()
        else:
            # 注文失敗
            error_message = order_result.get('error', '不明なエラー')
            self.logger.error(f" {symbol}の{position_type}エントリー注文が失敗しました: {error_message}")

    def _ceil_to_next_15min(self, ts):
        """ts の次の15分境界（:00/:15/:30/:45 の“次”）を返す"""
        ts = ts.replace(second=0, microsecond=0)
        add = 15 - (ts.minute % 15)
        if add == 0:
            add = 15
        return ts + timedelta(minutes=add)

    def _check_exit_conditions(self, symbol, stats, trade_logs, df_5min):
        """イグジット条件をチェックし、必要に応じて決済を実行する（backtest関数と整合性あり）
        
        Parameters:
        symbol (str): 通貨ペア
        current_price (float): 現在価格
        stats (dict): 統計情報の辞書
        trade_logs (list): 取引ログのリスト
        """
        position = self.positions.get(symbol)
        # ポジションが無ければ何もしない（SIMのループでNoneが混ざるケース対策）
        if not position:
            return

        entry_price = self.entry_prices.get(symbol) or 0.0
        entry_time = self.entry_times.get(symbol)
        # エントリー時刻が未設定の場合のセーフティ（SIMでの復元/初期化漏れ対策）
        if entry_time is None:
            self.logger.warning(f"{symbol}: entry_time が未設定のため現在時刻を代入します（SIM safety）")
            entry_time = now_jst()
            self.entry_times[symbol] = entry_time

        entry_size = self.entry_sizes[symbol]
        
        # 保有時間の計算
        holding_time = now_jst() - entry_time
        hours = holding_time.total_seconds() / 3600
        
        # イグジット条件の計算
        exit_condition = False
        exit_reason = ""

        # --- ここから判定ブロック置き換え（指値なし・バー内タッチ検出） ---
        exit_levels = self.calculate_dynamic_exit_levels(symbol, df_5min, position, entry_price)
        take_profit_price = float(exit_levels['take_profit_price'])
        stop_loss_price   = float(exit_levels['stop_loss_price'])

        # 現在値（通知や近接判定に使うだけ）
        current_price = self.get_current_price(symbol)

        # 最新5分バーの高値・安値で「一瞬タッチ」を検出
        last_bar = df_5min.iloc[-1] if len(df_5min) > 0 else None
        bar_high = float(last_bar['high']) if last_bar is not None else None
        bar_low  = float(last_bar['low'])  if last_bar is not None else None

        exit_condition = False
        exit_reason = ""

        if position == 'long':
            # 5分足の高値がTP以上、または安値がSL以下に触れていれば即EXIT
            tp_touch = (bar_high is not None and bar_high >= take_profit_price)
            sl_touch = (bar_low  is not None and bar_low  <= stop_loss_price)
        else:  # short
            tp_touch = (bar_low  is not None and bar_low  <= take_profit_price)   # 価格が下抜けたら利確
            sl_touch = (bar_high is not None and bar_high >= stop_loss_price)     # 価格が上抜けたら損切

        if tp_touch or sl_touch:
            exit_condition = True
            exit_reason = "利益確定" if tp_touch else "損切り"
        else:
            # バー更新タイミングの取りこぼし対策として、近接時だけ短時間の高頻度再確認（任意）
            # 例: 0.05% 以内に近づいたら最大3秒 100ms間隔で current_price を再チェック
            band = 0.0005  # 0.05%
            near = False
            if current_price:
                if position == 'long':
                    near = (abs(current_price - take_profit_price) <= take_profit_price * band) or \
                        (abs(current_price - stop_loss_price)   <= stop_loss_price   * band)
                else:
                    near = (abs(current_price - take_profit_price) <= take_profit_price * band) or \
                        (abs(current_price - stop_loss_price)   <= stop_loss_price   * band)
            if near:
                deadline = time.time() + 3.0
                while time.time() < deadline and not exit_condition:
                    cp = self.get_current_price(symbol) or current_price
                    if position == 'long':
                        if cp >= take_profit_price or cp <= stop_loss_price:
                            exit_condition = True
                            exit_reason = "利益確定" if cp >= take_profit_price else "損切り"
                    else:
                        if cp <= take_profit_price or cp >= stop_loss_price:
                            exit_condition = True
                            exit_reason = "利益確定" if cp <= take_profit_price else "損切り"
                    if not exit_condition:
                        time.sleep(0.1)
        
        # 長時間保有の処理（48時間以上で警告、72時間以上で強制決済）- これはlive固有の機能
        if hours >= 72:
            self.logger.warning(f"{symbol}のポジションが72時間以上保有されています。強制決済します。")
            exit_condition = True
            exit_reason = "長時間保有による強制決済"
        elif hours >= 48:
            self.logger.warning(f"{symbol}のポジションが48時間以上保有されています。監視を強化してください。")
        
        # イグジット条件を満たしている場合
        if exit_condition:
            # 決済注文の実行
            self.logger.info(f"{symbol}の{position}ポジションをイグジットします: {exit_reason}")
            
            # 通貨ペアの全ポジションを取得
            position_details = self.get_position_details(symbol.split('_')[0])

            if not position_details or not position_details.get("positions"):
                self.logger.warning(f"{symbol}の決済対象ポジションが見つかりません。ポジション同期を試みます。")
                # ポジション同期を試みる
                self.verify_positions()
                position_details = self.get_position_details(symbol.split('_')[0])
                
                if not position_details or not position_details.get("positions"):
                    self.logger.error(f"{symbol}の決済対象ポジションが再取得できませんでした。手動での対応が必要です。")
                    self.send_notification(
                        f"ポジション決済エラー: {symbol}",
                        f"決済対象ポジションが見つからないため、決済できませんでした。\n"
                        f"手動での対応が必要です。",
                        "error"
                    )
                    return

            # 複数ポジションを決済
            total_order_ids = []
            close_results = []
            success_count = 0
            failed_count = 0
            total_executed_size = 0

            # GMOコインの形式に変換
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))

            # ポジションの反対売買方向を決定
            side = "BUY" if position == 'short' else "SELL"
            # live / sim の分岐フラグ（以降で使う）
            is_live = bool(self.exchange_settings_gmo.get("live_trade", False))
            # SIM時の総約定サイズを自前集計
            sim_total_executed_size = 0.0

            # 各ポジションを順番に決済
            for pos in position_details["positions"]:
                # 同じシンボルのポジションのみ処理（安全対策）
                if pos.get("symbol") != gmo_symbol:
                    self.logger.info(f"シンボル不一致: 期待={gmo_symbol}, 実際={pos.get('symbol')}")
                    continue
                    
                # ポジションの方向が一致するか確認（買いポジションには売り決済、売りポジションには買い決済）
                pos_side = pos.get("side")
                if (position == 'long' and pos_side != "BUY") or (position == 'short' and pos_side != "SELL"):
                    self.logger.info(f"ポジション方向不一致: position={position}, pos_side={pos_side}")
                    continue
                    
                position_id = pos.get("positionId")  # ここがpositionIdになっている可能性あり
                if position_id is None:
                    position_id = pos.get("position_id")  # 別の形式も試す
                    
                if position_id is None:
                    self.logger.error(f"ポジションIDが見つかりません: {pos}")
                    continue

                pos_size = pos.get("size", 0)
                
                self.logger.info(f"個別ポジション決済: {gmo_symbol} {side} サイズ:{pos_size} (ポジションID: {position_id})")
                
                # ポジションサイズを適切にフォーマット
                if symbol == "doge_jpy" or symbol == "DOGE_JPY":
                    formatted_size = str(int(float(pos_size)))
                elif symbol == "xrp_jpy" or symbol == "XRP_JPY":
                    formatted_size = str(int(float(pos_size)))
                elif symbol == "eth_jpy" or symbol == "ETH_JPY":
                    formatted_size = str(round(float(pos_size), 2))
                elif symbol == "ltc_jpy" or symbol == "LTC_JPY":
                    formatted_size = str(int(float(pos_size)))
                elif symbol == "sol_jpy" or symbol == "SOL_JPY":
                    formatted_size = str(round(float(pos_size), 1))
                elif symbol == "bcc_jpy" or symbol == "BCH_JPY":  # 追加
                    formatted_size = str(round(float(pos_size), 1))
                elif symbol == "ada_jpy" or symbol == "ADA_JPY":  # 追加
                    formatted_size = str(int(float(pos_size)))
                else:
                    formatted_size = str(pos_size)


                # 決済注文実行
                # --- SIM(Paper) モード：APIは呼ばず、擬似注文を発行して成功扱いにする ---
                if not is_live:
                    order_id = f"SIM-CLOSE-{uuid.uuid4().hex[:12]}"
                    self.logger.info(f"[SIM CLOSE] {gmo_symbol} {side} size={pos_size} -> order_id={order_id}")
                    success_count += 1
                    total_order_ids.append(order_id)
                    sim_total_executed_size += float(pos_size or 0.0)
                    close_results.append({
                        'position_id': position_id,
                        'success': True,
                        'order_id': order_id,
                        'size': pos_size
                    })
                    continue  # API呼び出しは行わない

                try:
                    response = self.gmo_api.close_position(
                        symbol=gmo_symbol,
                        position_id=int(position_id),
                        size=formatted_size,
                        side=side,
                        position_type="MARKET"
                    )
                    
                    if response.get("status") == 0:
                        order_id = str(response.get("data"))
                        self.logger.info(f"決済注文成功: 注文ID={order_id}")
                        success_count += 1
                        total_order_ids.append(order_id)
                        
                        close_results.append({
                            'position_id': position_id,
                            'success': True,
                            'order_id': order_id,
                            'size': pos_size
                        })
                    else:
                        error_messages = response.get("messages", [])
                        error_msg = error_messages[0].get("message_string", "Unknown error") if error_messages else "Unknown error"
                        self.logger.error(f"決済注文エラー: {error_msg}")
                        failed_count += 1
                        
                        close_results.append({
                            'position_id': position_id,
                            'success': False,
                            'error': error_msg,
                            'size': pos_size
                        })
                except Exception as e:
                    self.logger.error(f"ポジション決済中のエラー: {str(e)}")
                    failed_count += 1
                    
                    close_results.append({
                        'position_id': position_id,
                        'success': False,
                        'error': str(e),
                        'size': pos_size
                    })
            
            # 決済結果の集計（SIM/LIVE 分岐）
            if success_count > 0:
                if is_live:
                    # LIVE：少し待ってから各注文の約定サイズを照会
                    time.sleep(3)
                    for result in close_results:
                        if result.get('success'):
                            order_id = result.get('order_id')
                            executed_size = self.check_order_execution(order_id, symbol)
                            if executed_size > 0:
                                total_executed_size += float(executed_size)
                else:
                    # SIM：API照会せず、自前で集計したサイズをそのまま採用
                    total_executed_size = float(sim_total_executed_size)
                            
                # Tx前で確定しておく（クリアされる前に掴む）
                entry_pos_id_for_trade = self.position_ids.get(symbol)

                # 成功した決済がある場合：ここで注文+フィルを原子的に確定
                self.logger.info(f"{symbol}のポジション決済結果: 成功={success_count}, 失敗={failed_count}, 総約定サイズ={total_executed_size}")

                # 代表注文ID（成功したものの先頭を採用：SIM/LIVE 共通）
                repr_order_id = total_order_ids[0] if total_order_ids else None

                # 事前ガード：IDなし or 約定サイズ0 なら、DB書込/トレード作成はしない
                if not repr_order_id or float(total_executed_size) <= 0.0:
                    insert_error(
                        "close/skip_db_no_exec",
                        f"repr_order_id={repr_order_id}, total_executed_size={total_executed_size}",
                        raw={"symbol": symbol, "results": close_results},
                    )
                    # ローカル状態や通知も「クローズ未確定」として扱うなら、ここで return して様子見
                    return

                try:
                    exec_price = float(self.get_current_price(symbol) or 0.0)
                    closed_size = float(total_executed_size)
                    saved_scores = self.entry_scores.get(symbol, {})
                    
                    trade_log_exit = {
                        "symbol": symbol,
                        "type": position,
                        "entry_price": float(entry_price),
                        "exit_price": float(exec_price),
                        "size": closed_size,
                        "profit": (exec_price - entry_price) * closed_size if position == "long" else (entry_price - exec_price) * closed_size,
                        "profit_pct": ((exec_price - entry_price) / entry_price * 100.0) if position == "long" else ((entry_price - exec_price) / entry_price * 100.0),
                        "holding_hours": hours,
                        "exit_reason": exit_reason,
                        # 参考情報（rawに残す用）
                        "entry_time": entry_time,
                        "scores": {
                            "entry_atr": saved_scores.get("entry_atr", 0),
                            "entry_adx": saved_scores.get("entry_adx", 0),
                            "rsi_score_long":  saved_scores.get("rsi_score_long", 0),
                            "rsi_score_short": saved_scores.get("rsi_score_short", 0),
                            "cci_score_long":  saved_scores.get("cci_score_long", 0),
                            "cci_score_short": saved_scores.get("cci_score_short", 0),
                            "volume_score":    saved_scores.get("volume_score", 0),
                            "bb_score_long":   saved_scores.get("bb_score_long", 0),
                            "bb_score_short":  saved_scores.get("bb_score_short", 0),
                            "ma_score_long":   saved_scores.get("ma_score_long", 0),
                            "ma_score_short":  saved_scores.get("ma_score_short", 0),
                            "adx_score_long":  saved_scores.get("adx_score_long", 0),
                            "adx_score_short": saved_scores.get("adx_score_short", 0),
                            "mfi_score_long":  saved_scores.get("mfi_score_long", 0),
                            "mfi_score_short": saved_scores.get("mfi_score_short", 0),
                            "atr_score_long":  saved_scores.get("atr_score_long", 0),
                            "atr_score_short": saved_scores.get("atr_score_short", 0),
                            "macd_score_long": saved_scores.get("macd_score_long", 0),
                            "macd_score_short":saved_scores.get("macd_score_short", 0),
                        },
                    }

                    trade_logs.append(trade_log_exit)

                    from sqlalchemy.exc import IntegrityError
                    def _dbg_dump(label, **kw):
                        try:
                            self.logger.info(f"[TXDBG] {label} | " + " | ".join(f"{k}={v!r}" for k,v in kw.items()))
                        except Exception:
                            pass

                    with begin() as conn:
                        side_order = "SELL" if position == "long" else "BUY"
                        side_trade = position  # long/short
                        exec_at = utcnow()

                        # 1) orders/fills
                        _dbg_dump("before mark_order_executed_with_fill",
                                order_id=str(repr_order_id), executed_size=float(closed_size), price=float(exec_price),
                                fee=None, executed_at=exec_at, symbol=symbol, side=side_order, type_="MARKET",
                                size_hint=float(closed_size))
                        try:
                            mark_order_executed_with_fill(
                                order_id=str(repr_order_id),
                                executed_size=float(closed_size),
                                price=float(exec_price),
                                fee=None,
                                executed_at=exec_at,
                                fill_raw={"source": "exit_tx", "symbol": symbol},
                                order_raw={"close_results": close_results},
                                symbol=symbol,
                                side=side_order,
                                type_="MARKET",
                                size_hint=float(closed_size),
                                requested_at=exec_at,
                                placed_at=exec_at,
                                conn=conn,
                            )
                        except Exception as e:
                            self.logger.error(f"[TXERR] mark_order_executed_with_fill failed: {type(e).__name__} orig={getattr(e,'orig',None)} "
                                            f"stmt={getattr(e,'statement',None)} params={getattr(e,'params',None)}")
                            raise

                        # 2) positions（position_id フォールバック付き）
                        # 優先順: キャッシュ -> Tx前に掴んだID -> close_resultsに含まれるID
                        pos_id_for_update = (
                            (self.position_ids.get(symbol) or None)
                            or (entry_pos_id_for_trade or None)
                            or (next((r.get("position_id") for r in close_results if r.get("success") and r.get("position_id")), None))
                        )
                        if pos_id_for_update:
                            sid = to_uuid_or_none(getattr(self, "strategy_id", None))
                            _dbg_dump(
                                "before upsert_position",
                                position_id=str(pos_id_for_update),
                                symbol=symbol,
                                side=side_trade,
                                size=0.0,
                                avg_entry_price=float(self.entry_prices.get(symbol) or 0.0),
                                updated_at=exec_at,
                                strategy_id=sid,
                                user_id=getattr(self, "user_id", None),
                                source=default_source_from_env(self),
                            )
                            try:
                                upsert_position(
                                    position_id=str(pos_id_for_update),
                                    symbol=symbol,
                                    side=side_trade,
                                    size=0.0,
                                    avg_entry_price=float(self.entry_prices.get(symbol) or 0.0),
                                    opened_at=None,
                                    updated_at=exec_at,
                                    raw={"closed_by": "auto", "close_results": close_results},
                                    strategy_id=sid,
                                    user_id=getattr(self, "user_id", None),
                                    source=default_source_from_env(self),
                                    conn=conn,
                                )
                            except Exception as e:
                                self.logger.error(
                                    f"[TXERR] upsert_position failed: {type(e).__name__} orig={getattr(e,'orig',None)} "
                                    f"stmt={getattr(e,'statement',None)} params={getattr(e,'params',None)}"
                                )
                                raise
                        else:
                            self.logger.warning(
                                "[TXWARN] skip upsert_position: position_id not found "
                                f"(symbol={symbol}, cache={self.position_ids.get(symbol)}, entry_pos_id_for_trade={entry_pos_id_for_trade})"
                            )

                        # 3) trades
                        _dbg_dump("before insert_trade",
                                symbol=trade_log_exit.get("symbol"), side=side_trade,
                                entry_position_id=entry_pos_id_for_trade, exit_order_id=str(repr_order_id),
                                entry_price=_num(trade_log_exit["entry_price"]),
                                exit_price=_num(trade_log_exit["exit_price"]),
                                size=_num(trade_log_exit["size"]),
                                pnl=_num(trade_log_exit["profit"]),
                                pnl_pct=_num(trade_log_exit["profit_pct"]),
                                holding_hours=_num(trade_log_exit.get("holding_hours")),
                                closed_at=exec_at,
                                strategy_id=to_uuid_or_none(getattr(self,"strategy_id",None)),
                                user_id=getattr(self,"user_id",None),
                                source=default_source_from_env(self))
                        try:
                            insert_trade(
                                trade_id=None,
                                symbol=trade_log_exit["symbol"],
                                side=side_trade,
                                entry_position_id=entry_pos_id_for_trade,
                                exit_order_id=str(repr_order_id),
                                entry_price=_num(trade_log_exit["entry_price"]),
                                exit_price=_num(trade_log_exit["exit_price"]),
                                size=_num(trade_log_exit["size"]),
                                pnl=_num(trade_log_exit["profit"]),
                                pnl_pct=_num(trade_log_exit["profit_pct"]),
                                holding_hours=_num(trade_log_exit.get("holding_hours")),
                                closed_at=exec_at,
                                raw=trade_log_exit,
                                strategy_id=to_uuid_or_none(getattr(self, "strategy_id", None)),
                                user_id=getattr(self, "user_id", None),
                                source=default_source_from_env(self),
                                conn=conn,
                            )
                        except Exception as e:
                            self.logger.error(f"[TXERR] insert_trade failed: {type(e).__name__} orig={getattr(e,'orig',None)} "
                                            f"stmt={getattr(e,'statement',None)} params={getattr(e,'params',None)}")
                            raise

                except IntegrityError as ie:
                    insert_error("close/tx_commit/integrity", str(ie), raw={"symbol": symbol, "results": close_results})
                    raise   # ← Fail-Fast（ここで止める）
                except Exception as e:
                    insert_error("close/tx_commit", str(e), raw={"symbol": symbol, "results": close_results})
                    raise   # ← Fail-Fast（ここで止める）
                    
                # 損益計算
                profit = trade_log_exit["profit"]
                profit_pct = trade_log_exit["profit_pct"]
                
                # 手数料を考慮
                #profit *= 0.9976  # 0.24%の手数料
                
                # 統計情報の更新
                self.total_profit += profit
                stats['total_trades'] += 1
                stats['daily_trades'] += 1
                
                if profit > 0:
                    stats['total_wins'] += 1
                    stats['daily_wins'] += 1
                    stats['total_profit'] += profit
                    stats['daily_profit'] += profit
                else:
                    stats['total_losses'] += 1
                    stats['daily_losses'] += 1
                    stats['total_loss'] += abs(profit)
                    stats['daily_loss'] += abs(profit)
                
                # ログ出力
                entry_sentiment = getattr(self, 'sentiment', {}).copy() if hasattr(self, 'sentiment') else {}
                self.log_exit(symbol, position, current_price, entry_price, now_jst(), profit, profit_pct, exit_reason, hours, entry_sentiment)
                
                # 通知送信
                if self.notification_settings.get('send_on_exit', True):
                    exit_side = "EXIT-LONG" if position == "long" else "EXIT-SHORT"
                    try:
                        # こちらに差し替え
                        self._send_exit_detail(
                            symbol,
                            exit_price=trade_log_exit["exit_price"],      # 約定平均が取れればそちらを渡す
                            timeframe="15m",                # 主に使う時間足を指定
                            reason=exit_reason
                        )
                    except Exception as e:
                        self.logger.exception(f"EXIT詳細通知失敗: {e}")
                        # フォールバック（従来のシンプル通知）
                        self._notify_signal(
                            side=exit_side,
                            symbol=symbol,
                            price=float(current_price) if current_price is not None else None,
                            tp=float(take_profit_price) if take_profit_price is not None else None,
                            sl=float(stop_loss_price) if stop_loss_price is not None else None,
                            score=None,
                            reason_tags=[exit_reason] if exit_reason else None
                        )
                
                # ポジション情報をリセット
                self.positions[symbol] = None
                self.entry_prices[symbol] = 0
                self.entry_times[symbol] = None
                self.entry_sizes[symbol] = 0
                self.position_ids[symbol] = None  # ポジションIDもクリア
                self.entry_scores[symbol] = {}

                now = now_jst()
                unlock_time = self._ceil_to_next_15min(now)
                self.reentry_block_until[symbol] = unlock_time
                self.logger.info(
                    f"{symbol}: 決済後の再エントリーを次の15分足開始までブロックします（解禁: {unlock_time}）"
                )

                # ポジション情報を保存
                self.save_positions()
            else:
                # 全ての決済が失敗した場合
                self.logger.error(f"{symbol}の全てのポジション決済に失敗しました")
                
                # 通知を送信
                self.send_notification(
                    f"決済エラー: {symbol} {position}",
                    f"全てのポジション決済が失敗しました。手動での対応が必要かもしれません。",
                    "error"
                )

    def _perform_health_check(self):
        """システムの健全性チェックを実行する"""
        self.logger.info("システム健全性チェックを実行中...")
        
        try:
            # メモリ使用量の確認
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.logger.info(f"メモリ使用量: {memory_mb:.2f}MB")
            
            # ディスク容量の確認
            du = psutil.disk_usage('/')
            free_gb = du.free / 1024 / 1024 / 1024
            self.logger.info(f"ディスク空き容量: {free_gb:.2f}GB (使用率: {du.percent:.1f}%)")
            
            # ディスクが少ない場合は警告
            if free_gb < 1.0:
                self.logger.warning("ディスク容量が不足しています。不要なファイルを削除してください。")
                self.send_notification(
                    "ディスク容量警告",
                    f"ディスク空き容量が {free_gb:.2f}GB しかありません。不要なファイルを削除してください。",
                    "warning"
                )
            
            # ログファイルサイズの確認
            log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.log')]
            if len(log_files) > 50:
                self.logger.warning(f"ログファイルが多すぎます: {len(log_files)}個")
                # 古いログファイルを削除
                log_files_with_time = [(f, os.path.getmtime(os.path.join(self.log_dir, f))) for f in log_files]
                log_files_with_time.sort(key=lambda x: x[1])  # 修正時間でソート
                
                # 最新の30ファイル以外は削除
                for f, _ in log_files_with_time[:-30]:
                    try:
                        os.remove(os.path.join(self.log_dir, f))
                        self.logger.info(f"古いログファイルを削除: {f}")
                    except Exception as e:
                        self.logger.error(f"ログファイル削除エラー: {str(e)}")
            
            # キャッシュサイズの確認
            cache_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in os.listdir(self.cache_dir) if os.path.isfile(os.path.join(self.cache_dir, f)))
            cache_size_mb = cache_size / 1024 / 1024
            self.logger.info(f"キャッシュサイズ: {cache_size_mb:.2f}MB")
            
            # キャッシュが大きすぎる場合は一部クリア
            if cache_size_mb > 500:  # 500MB以上
                self.logger.warning(f"キャッシュサイズが大きすぎます: {cache_size_mb:.2f}MB")
                
                # 5日以上前のキャッシュファイルを削除
                now = time.time()
                for f in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, f)
                    if os.path.isfile(file_path):
                        if now - os.path.getmtime(file_path) > 5 * 24 * 3600:  # 5日以上前
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                self.logger.error(f"キャッシュファイル削除エラー: {str(e)}")
            
            self.logger.info("システム健全性チェックが完了しました")
            return True
        except Exception as e:
            self.logger.error(f"システム健全性チェック中にエラーが発生: {str(e)}")
            return False

    def send_daily_report(self, day: Optional[datetime] = None):
        """
        手動/外部呼び出し用の公開メソッド。
        実処理は _send_daily_report に委譲する（正規化経路を強制）。
        """
        stats = {}
        if day is not None:
            start, _ = self._day_bounds_jst(day)
            # _send_daily_report 側でこのキーを見て開始日を上書き
            stats["force_day_start"] = start
        return self._send_daily_report(stats)


    def _day_bounds_jst(self, day=None):
        day = (day or dt.datetime.now(JST)).astimezone(JST)
        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end

    def _generate_final_report(self, start_date, start_balance, stats, trade_logs):
        """最終レポートを生成する
        
        Parameters:
        start_date (datetime): 開始日時
        start_balance (float): 開始時の資金
        stats (dict): 統計情報の辞書
        trade_logs (list): 取引ログのリスト
        """
        end_balance = self.get_total_balance()
        running_time = now_jst() - start_date
        days = running_time.days
        hours = running_time.seconds // 3600
        minutes = (running_time.seconds % 3600) // 60
        
        # 総利益計算
        total_return = 0
        if start_balance > 0:
            total_return = (end_balance - start_balance) / start_balance * 100
        
        # 勝率計算
        win_rate = 0
        if stats['total_trades'] > 0:
            win_rate = stats['total_wins'] / stats['total_trades'] * 100
        
        # プロフィットファクター
        profit_factor = 0
        if stats['total_loss'] > 0:
            profit_factor = stats['total_profit'] / stats['total_loss']
        
        # 通貨ペアごとの成績
        symbol_stats = {}
        for log in trade_logs:
            if log.get('action') == 'exit':  # 決済ログのみを使用
                symbol = log.get('symbol')
                profit = log.get('profit', 0)
                
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'profit': 0,
                        'loss': 0
                    }
                
                symbol_stats[symbol]['trades'] += 1
                if profit > 0:
                    symbol_stats[symbol]['wins'] += 1
                    symbol_stats[symbol]['profit'] += profit
                else:
                    symbol_stats[symbol]['losses'] += 1
                    symbol_stats[symbol]['loss'] += abs(profit)
        
        # レポート表示
        self.logger.info("\n=== 最終取引レポート ===")
        self.logger.info(f"期間: {start_date.strftime('%Y-%m-%d %H:%M')} ~ {now_jst().strftime('%Y-%m-%d %H:%M')}")
        self.logger.info(f"稼働時間: {days}日 {hours}時間 {minutes}分")
        self.logger.info(f"取引回数: {stats['total_trades']}回 (勝ち: {stats['total_wins']}回, 負け: {stats['total_losses']}回)")
        self.logger.info(f"勝率: {win_rate:.2f}%")
        self.logger.info(f"プロフィットファクター: {profit_factor:.2f}")
        self.logger.info(f"利益合計: {stats['total_profit']:,.0f}円")
        self.logger.info(f"損失合計: {stats['total_loss']:,.0f}円")
        self.logger.info(f"純利益: {self.total_profit:,.0f}円")
        self.logger.info(f"開始時資金: {start_balance:,.0f}円")
        self.logger.info(f"終了時資金: {end_balance:,.0f}円")
        self.logger.info(f"収益率: {total_return:+.2f}%")
        
        # 通貨ペアごとの成績
        self.logger.info("\n=== 通貨ペアごとの成績 ===")
        for symbol, stats in symbol_stats.items():
            win_rate = 0
            if stats['trades'] > 0:
                win_rate = (stats['wins'] / stats['trades']) * 100
            
            symbol_profit_factor = 0
            if stats['loss'] > 0:
                symbol_profit_factor = stats['profit'] / stats['loss']
            
            net_profit = stats['profit'] - stats['loss']
            
            self.logger.info(f"{symbol}:")
            self.logger.info(f"  取引回数: {stats['trades']}回 (勝ち: {stats['wins']}回, 負け: {stats['losses']}回)")
            self.logger.info(f"  勝率: {win_rate:.2f}%")
            self.logger.info(f"  プロフィットファクター: {symbol_profit_factor:.2f}")
            self.logger.info(f"  利益: {stats['profit']:,.0f}円, 損失: {stats['loss']:,.0f}円")
            self.logger.info(f"  純利益: {net_profit:+,.0f}円")
        
        # 最終レポートの通知送信
        report_body = (
            f"📊 最終取引レポート\n\n"
            f"⏱️ 期間: {days}日 {hours}時間 {minutes}分\n"
            f"🔄 取引回数: {stats['total_trades']}回\n"
            f"📈 勝率: {win_rate:.2f}%\n"
            f"💹 プロフィットファクター: {profit_factor:.2f}\n"
            f"💰 純利益: {self.total_profit:+,.0f}円\n"
            f"📊 収益率: {total_return:+.2f}%\n"
        )
        self.send_notification("ボット停止 - 最終レポート", report_body, "info")

    def _save_error_log(self, error_logs):
        """エラーログをファイルに保存する
        
        Parameters:
        error_logs (list): エラーログのリスト
        """
        try:
            error_log_file = os.path.join(self.log_dir, f'error_log_{now_jst().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(error_log_file, 'w') as f:
                f.write("===== トレーディングボットエラーログ =====\n")
                f.write(f"生成時刻: {now_jst().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for log in error_logs:
                    f.write(f"{log}\n")
            
            self.logger.info(f"エラーログを保存しました: {error_log_file}")
        except Exception as e:
            self.logger.error(f"エラーログの保存中にエラーが発生: {str(e)}")

    def _get_dynamic_profit_loss_settings(self, symbol, df_5min, is_active_hours=False):
        """通貨ペアとマーケット状況に応じた動的な利確・損切り設定を取得
        
        Parameters:
        symbol (str): 通貨ペア
        is_active_hours (bool): アクティブな取引時間帯かどうか
        
        Returns:
        dict: 利確・損切り設定
        """
        # 通貨ペアごとの基本設定
        base_settings = {
            "ltc_jpy": {
                "long_profit_take": 1.022,   # 2.5%の利益確定
                "long_stop_loss": 0.972,     # 2.0%の損切り
                "short_profit_take": 0.978,  # 2.5%の利益確定
                "short_stop_loss": 1.028     # 2.0%の損切り
            },
            "xrp_jpy": {    
                "long_profit_take": 1.022,   # 2.5%の利益確定
                "long_stop_loss": 0.972,     # 2.0%の損切り
                "short_profit_take": 0.978,  # 2.5%の利益確定
                "short_stop_loss": 1.028     # 2.0%の損切り
            },
            "eth_jpy": {
                "long_profit_take": 1.022,   # 2.5%の利益確定
                "long_stop_loss": 0.970,     # 2.0%の損切り
                "short_profit_take": 0.978,  # 2.5%の利益確定
                "short_stop_loss": 1.030     # 2.0%の損切り
            },
            "sol_jpy": {
                "long_profit_take": 1.025,   # 3.0%の利益確定
                "long_stop_loss": 0.970,     # 2.5%の損切り
                "short_profit_take": 0.975,  # 3.0%の利益確定
                "short_stop_loss": 1.030     # 2.5%の損切り
            },
            "doge_jpy": {
                "long_profit_take": 1.022,   # 2.5%の利益確定
                "long_stop_loss": 0.970,     # 2.0%の損切り
                "short_profit_take": 0.978,  # 2.5%の利益確定
                "short_stop_loss": 1.030     # 2.0%の損切り
            },
            "bcc_jpy": {  # 新規追加
                "long_profit_take": 1.022,   # 2.5%の利益確定
                "long_stop_loss": 0.972,     # 2.0%の損切り
                "short_profit_take": 0.978,  # 2.5%の利益確定
                "short_stop_loss": 1.028     # 2.0%の損切り
            },
            "ada_jpy": {  # 新規追加
                "long_profit_take": 1.022,   # 2.5%の利益確定
                "long_stop_loss": 0.972,     # 2.0%の損切り
                "short_profit_take": 0.978,  # 2.5%の利益確定
                "short_stop_loss": 1.028     # 2.0%の損切り
            }
        }

        # デフォルト設定（通貨ペアが見つからない場合）
        default_settings = {
            "long_profit_take": 1.020,
            "long_stop_loss": 0.985,
            "short_profit_take": 0.980,
            "short_stop_loss": 1.015
        }
        
        # 通貨ペアの設定を取得
        settings = base_settings.get(symbol, default_settings).copy()
        
        return settings

    def calculate_dynamic_exit_levels(self, symbol, df_5min, position_type, entry_price):
        """
        通貨ペア・ポジションタイプ別のATR & ADXに応じて、利確・損切レベルを調整
        """
        try:
            # 最新ATRの取得
            atr_series = df_5min['ATR'].dropna()
            if atr_series.empty:
                raise ValueError("ATRデータが存在しません。")
            atr = atr_series.iloc[-2]

            # 最新ADXの取得
            adx_series = df_5min['ADX'].dropna()
            adx = adx_series.iloc[-2] if not adx_series.empty else None

            plus_di = df_5min['DI+'].dropna().iloc[-2] if 'DI+' in df_5min.columns else None
            minus_di = df_5min['DI-'].dropna().iloc[-2] if 'DI-' in df_5min.columns else None

            # ベースのTP/SL設定取得
            base = self._get_dynamic_profit_loss_settings(symbol, df_5min)
            if position_type == 'long':
                tp_ratio = base['long_profit_take']
                sl_ratio = base['long_stop_loss']
            else:
                tp_ratio = base['short_profit_take']
                sl_ratio = base['short_stop_loss']

            base_tp_pct = abs(tp_ratio - 1.0)
            base_sl_pct = abs(sl_ratio - 1.0)

            # 通貨ペア別ATR倍率設定
            atr_thresholds = {
                'ltc_jpy': {'low': 60, 'high': 75, 'low_mult': 0.88, 'high_mult_long': 1.10, 'high_mult_short': 1.10},
                'ada_jpy': {'low': 0.50, 'high': 0.57, 'low_mult': 0.88, 'high_mult_long': 1.10, 'high_mult_short': 1.07},
                'xrp_jpy': {'low': 1.5, 'high': 2.1, 'low_mult': 0.95, 'high_mult_long': 1.00, 'high_mult_short': 0.90},
                'eth_jpy': {'low': 2000, 'high': 2500, 'low_mult': 0.92, 'high_mult_long': 1.00, 'high_mult_short': 1.10},
                'sol_jpy': {'low': 100, 'high': 140, 'low_mult': 0.88, 'high_mult_long': 1.10, 'high_mult_short': 1.03},
                'doge_jpy': {'low': 0.20, 'high': 0.24, 'low_mult': 0.93, 'high_mult_long': 0.95, 'high_mult_short': 1.00},
                'bcc_jpy': {'low': 310, 'high': 350, 'low_mult': 0.88, 'high_mult_long': 1.08, 'high_mult_short': 1.08}
            }
            default_setting = {'low': 0.5, 'high': 2.0, 'low_mult': 0.9, 'high_mult_long': 1.1, 'high_mult_short': 1.1}
            config = atr_thresholds.get(symbol, default_setting)
            adx_thresholds = {
                'ltc_jpy':   {'low': 20, 'high': 50, 'low_mult': 0.83, 'high_mult': 1.15},
                'ada_jpy':   {'low': 22, 'high': 48, 'low_mult': 0.83, 'high_mult': 1.17},
                'xrp_jpy':   {'low': 20, 'high': 50, 'low_mult': 0.80, 'high_mult': 1.20},
                'eth_jpy':   {'low': 20, 'high': 50, 'low_mult': 0.85, 'high_mult': 1.17},
                'sol_jpy':   {'low': 20, 'high': 50, 'low_mult': 0.85, 'high_mult': 1.15},
                'doge_jpy':  {'low': 20, 'high': 50, 'low_mult': 0.80, 'high_mult': 1.20},
                'bcc_jpy':   {'low': 22, 'high': 32, 'low_mult': 0.83, 'high_mult': 1.15}
            }
            default_adx_setting = {'low': 20, 'high': 50, 'low_mult': 0.90, 'high_mult': 1.10}
            adx_config = adx_thresholds.get(symbol, default_adx_setting)

            # ATRベース調整
            tp_pct = base_tp_pct
            sl_pct = base_sl_pct
            if atr < config['low']:
                tp_pct *= config['low_mult']
                sl_pct *= config['low_mult']
            elif atr > config['high']:
                if position_type == 'long':
                    tp_pct *= config['high_mult_long']
                    sl_pct *= config['high_mult_long']
                else:
                    tp_pct *= config['high_mult_short']
                    sl_pct *= config['high_mult_short']

            # ADXベース調整（弱→狭め、強→広げ）
            if adx is not None:
                if adx < adx_config['low']:
                    tp_pct *= adx_config['low_mult']
                    sl_pct *= adx_config['low_mult']
                elif adx > adx_config['high']:
                    tp_pct *= adx_config['high_mult']
                    sl_pct *= adx_config['high_mult']

            # --- ADX減衰チェック ---
            adx_prev = df_5min['ADX'].dropna().iloc[-3] if len(df_5min['ADX'].dropna()) >= 3 else adx
            adx_decreasing = (adx is not None and adx_prev is not None and adx < adx_prev)

            # --- DIクロスチェック ---
            di_cross_signal = False
            if plus_di is not None and minus_di is not None:
                if position_type == 'long' and plus_di < minus_di:
                    di_cross_signal = True
                elif position_type == 'short' and minus_di < plus_di:
                    di_cross_signal = True

            # --- 早期警戒シグナルでの補正 ---
            if adx_decreasing or di_cross_signal:
                tp_pct *= 0.8    # 利確幅を縮小（早期確定）
                sl_pct *= 0.9    # 損切りを少し手前に


            # トレンド方向と一致しているかを+DI / -DIで判定
            if plus_di is not None and minus_di is not None:
                if position_type == 'long' and plus_di < minus_di:
                    tp_pct *= 0.90  # トレンド逆行時は利確を保守的に
                    sl_pct *= 0.95
                elif position_type == 'short' and minus_di < plus_di:
                    tp_pct *= 0.90
                    sl_pct *= 0.95

            # ========== 逆方向 4本中3本で開始、同方向 2本連続で解除（ロック継続版） ==========
            # 対象バー：確定足のみ（-1 は含めない）
            score_true_thresh = 0.5
            has_buy_sig  = 'buy_signal'  in df_5min.columns
            has_sell_sig = 'sell_signal' in df_5min.columns
            has_buy_scr  = 'buy_score'   in df_5min.columns
            has_sell_scr = 'sell_score'  in df_5min.columns

            def lastN_bool(series_or_boollike, N):
                """確定N本（終値確定済み）を True/False 配列で取得（不足・NaNは False 扱い）"""
                seq = series_or_boollike.iloc[-(N+1):-1]  # 例: N=4 → -5:-1（-5,-4,-3,-2）
                return seq.fillna(False).astype(bool)

            # ---- True/False 配列の用意（フォールバック：score >= 0.5） ----
            if has_buy_sig:
                buy_seq4 = lastN_bool(df_5min['buy_signal'], 4)
                buy_seq2 = lastN_bool(df_5min['buy_signal'], 2)
            elif has_buy_scr:
                buy_seq4 = lastN_bool(df_5min['buy_score'] >= score_true_thresh, 4)
                buy_seq2 = lastN_bool(df_5min['buy_score'] >= score_true_thresh, 2)
            else:
                buy_seq4 = buy_seq2 = None

            if has_sell_sig:
                sell_seq4 = lastN_bool(df_5min['sell_signal'], 4)
                sell_seq2 = lastN_bool(df_5min['sell_signal'], 2)
            elif has_sell_scr:
                sell_seq4 = lastN_bool(df_5min['sell_score'] >= score_true_thresh, 4)
                sell_seq2 = lastN_bool(df_5min['sell_score'] >= score_true_thresh, 2)
            else:
                sell_seq4 = sell_seq2 = None

            def is_three_of_four(seq):
                return (seq is not None) and (len(seq) == 4) and (int(seq.sum()) >= 3)

            def is_two_streak_true(seq):
                return (seq is not None) and (len(seq) == 2) and bool(seq.iloc[0]) and bool(seq.iloc[1])

            # ---- 開始（逆方向3/4）／解除（同方向2連続）の判定 ----
            if position_type == 'long':
                opp_start   = is_three_of_four(sell_seq4)   # 逆方向（ロング時は sell）が4本中3本
                same_unlock = is_two_streak_true(buy_seq2)  # 同方向（ロング時は buy）が2本連続
            else:  # short
                opp_start   = is_three_of_four(buy_seq4)    # 逆方向（ショート時は buy）が4本中3本
                same_unlock = is_two_streak_true(sell_seq2) # 同方向（ショート時は sell）が2本連続

            # ---- 状態辞書の安全初期化 ----
            if not hasattr(self, 'opposite_narrow_state'):
                self.opposite_narrow_state = {}
            if symbol not in self.opposite_narrow_state:
                self.opposite_narrow_state[symbol] = {'long': False, 'short': False}
            if position_type not in self.opposite_narrow_state[symbol]:
                self.opposite_narrow_state[symbol][position_type] = False

            lock_on = self.opposite_narrow_state[symbol][position_type]

            # ---- 状態更新ロジック ----
            # 解除：同方向が2本連続で出たらロックOFF
            if same_unlock:
                lock_on = False
            # 開始：逆方向が4本中3本 かつ 解除条件は出ていない → ロックON
            elif opp_start and not same_unlock:
                lock_on = True
            # それ以外：状態維持

            # 保存
            self.opposite_narrow_state[symbol][position_type] = lock_on

            # ---- 適用 ----
            opposite_streak_factor = 0.9  # 狭め係数（必要に応じて調整）
            if lock_on:
                sl_pct *= opposite_streak_factor
            # ======================================================================


            # エグジット価格計算
            if position_type == 'long':
                take_profit_price = entry_price * (1 + tp_pct)
                stop_loss_price = entry_price * (1 - sl_pct)
                take_profit_ratio = 1 + tp_pct
                stop_loss_ratio = 1 - sl_pct
            else:
                take_profit_price = entry_price * (1 - tp_pct)
                stop_loss_price = entry_price * (1 + sl_pct)
                take_profit_ratio = 1 - tp_pct
                stop_loss_ratio = 1 + sl_pct

            risk_reward_ratio = tp_pct / sl_pct if sl_pct != 0 else float('inf')

            return {
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_ratio': take_profit_ratio,
                'stop_loss_ratio': stop_loss_ratio,
                'atr_value': atr,
                'risk_reward_ratio': risk_reward_ratio
            }

        except Exception as e:
            self.logger.error(f"{symbol}のATR利確損切計算中にエラー: {str(e)}", exc_info=True)
            default_settings = self._get_dynamic_profit_loss_settings(symbol, df_5min)
            if position_type == 'long':
                return {
                    'take_profit_price': entry_price * default_settings['long_profit_take'],
                    'stop_loss_price': entry_price * default_settings['long_stop_loss'],
                    'take_profit_ratio': default_settings['long_profit_take'],
                    'stop_loss_ratio': default_settings['long_stop_loss'],
                    'atr_value': 0,
                    'risk_reward_ratio': 2.0
                }
            else:
                return {
                    'take_profit_price': entry_price * default_settings['short_profit_take'],
                    'stop_loss_price': entry_price * default_settings['short_stop_loss'],
                    'take_profit_ratio': default_settings['short_profit_take'],
                    'stop_loss_ratio': default_settings['short_stop_loss'],
                    'atr_value': 0,
                    'risk_reward_ratio': 2.0
                }


# ================================
# メイン実行部分（統一版・重複禁止）
# ================================
if __name__ == "__main__":
    # DB起動確認（失敗しても落とさない）
    try:
        from db import ping_db_once  # あれば
        ping_db_once()
    except Exception as e:
        logging.getLogger(__name__).warning("DB ping skipped due to error: %s", e)

    # コマンドライン引数
    parser = argparse.ArgumentParser(description="仮想通貨トレーディングボット（live / paper / backtest）")
    parser.add_argument("mode", choices=["backtest", "live", "paper"], help="実行モード (backtest / live / paper)")
    parser.add_argument("--days", type=int, default=60, help="バックテスト日数 (デフォルト: 60)")
    parser.add_argument("--capital", type=int, default=200000, help="初期資金 (円)")
    parser.add_argument("--test", action="store_true", help="テストモード（取引額を半分にする等）")
    parser.add_argument("--notify", action="store_true", help="通知を有効にする")
    parser.add_argument("--debug", action="store_true", help="デバッグログを有効にする")
    parser.add_argument("--api-key", type=str, help="GMO API キー（任意。liveで未設定なら環境変数を使用）")
    parser.add_argument("--api-secret", type=str, help="GMO API シークレット（任意。liveで未設定なら環境変数を使用）")
    parser.add_argument("--reset", action="store_true", help="起動時にポジション情報をリセットする")
    parser.add_argument("--clear-cache", action="store_true", help="キャッシュをクリアしてから実行する")
    parser.add_argument("--user-id", type=int, help="通知/DB設定をひもづけるユーザーID")
    parser.add_argument("--rules-version", type=str, help="signal_rule_thresholds の version を明示指定（未指定なら None）")
    args = parser.parse_args()

    # ボット生成
    bot = CryptoTradingBot(initial_capital=args.capital, test_mode=args.test, user_id=args.user_id)

    # ログレベル
    if args.debug:
        bot.logger.setLevel(logging.DEBUG)
        for h in bot.logger.handlers:
            h.setLevel(logging.DEBUG)
        bot.logger.info("デバッグモードが有効になりました")

    # 通知
    if args.notify:
        bot.notification_settings["enabled"] = True
        bot.logger.info("通知機能が有効になりました")
    else:
        bot.notification_settings["enabled"] = False

    # APIキー（CLI優先で上書き）
    if args.api_key:
        bot.exchange_settings_gmo["api_key"] = args.api_key
    if args.api_secret:
        bot.exchange_settings_gmo["api_secret"] = args.api_secret

    # 事前メンテ（任意フラグ）
    if args.clear_cache:
        try:
            import shutil, os
            if os.path.isdir(bot.cache_dir):
                shutil.rmtree(bot.cache_dir)
                os.makedirs(bot.cache_dir, exist_ok=True)
            bot.logger.info("キャッシュをクリアしました")
        except Exception as e:
            bot.logger.warning("キャッシュクリアに失敗: %s", e)
    if args.reset:
        try:
            # 必要であれば、既存の reset ロジックを呼ぶ
            if hasattr(bot, "reset_positions"):
                bot.reset_positions()
            else:
                # 既存の保存ファイル削除やDBリセットなど
                pass
            bot.logger.info("ポジション情報をリセットしました")
        except Exception as e:
            bot.logger.warning("ポジションリセットに失敗: %s", e)

    # モード実行（単一分岐）
    if args.mode in ("live", "paper"):
        is_live = (args.mode == "live")
        bot.exchange_settings_gmo["live_trade"] = is_live
        bot.source = "real" if is_live else "sim"
        # ← モードに応じた positions ファイルへ切り替え
        bot.positions_path = bot._positions_filepath()
        # ★ SIM（paper）時のみ、ファイルからポジションを復元
        if not is_live:
            bot.load_positions()
        bot.setup_logging()

        msg = "リアルタイムトレード（LIVE）を開始します..." if is_live else "ペーパートレード（SIM）を開始します..."
        print(msg)
        bot.logger.info(msg)

        # live時のみ API を遅延初期化（paperは不要）
        if is_live:
            if hasattr(bot, "_init_gmo_api_if_needed"):
                bot._init_gmo_api_if_needed()
            else:
                # フォールバック初期化
                ak = bot.exchange_settings_gmo.get("api_key") or os.getenv("GMO_API_KEY", "")
                sk = bot.exchange_settings_gmo.get("api_secret") or os.getenv("GMO_API_SECRET", "")
                if not ak or not sk:
                    raise RuntimeError("GMO APIキー/シークレットが未設定です（liveモード）。")
                try:
                    bot.gmo_api = GMOPrivateAPI(ak, sk)  # 片方の実装名
                except NameError:
                    bot.gmo_api = GMOCoinAPI(ak, sk)    # もう片方の実装名

        # 価格キャッシュ Refresher（live/paper 共通）
        try:
            from threading import Thread
            refresher = PriceCacheRefresher(
                bot.symbols,
                bot.get_current_price,
                interval_sec=30,
                logger=bot.logger
            )
            Thread(target=refresher.run_forever, daemon=True).start()
        except Exception as e:
            bot.logger.warning("PriceCacheRefresher 起動に失敗: %s", e)

        # 実行
        bot.run_live()

    elif args.mode == "backtest":
        bot.is_backtest_mode = True
        bot.source = "backtest"
        print(f"{args.days}日間のバックテストを開始します...")
        bot.logger.info("バックテスト開始 days=%s", args.days)

        # backtest エントリの名前差を吸収
        try:
            if hasattr(bot, "backtest"):
                try:
                    bot.backtest(args.days)
                except TypeError:
                    bot.backtest(days_to_test=args.days)
            elif hasattr(bot, "run_backtest"):
                bot.run_backtest(days=args.days)
            else:
                raise RuntimeError("バックテストのエントリ関数が見つかりません（backtest / run_backtest）")
        except Exception:
            bot.logger.exception("バックテスト実行中に例外が発生しました")
            raise
