import requests
import pandas as pd
import numpy as np
from datetime import datetime,date, timedelta
import time
import json
import os
import logging
import smtplib
import shutil
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
import concurrent.futures
import math

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
    
    def _request(self, method, path, params=None, data=None):
        """APIリクエストの共通処理"""
        url = self.base_url + path
        
        # デバッグ：パラメータを確認
        if params:
            print(f"DEBUG: _request params: {params}")
        
        # GETメソッドの場合、パラメータをクエリストリングとして追加
        if method == "GET" and params:
            query_params = []
            for key, value in params.items():
                query_params.append(f"{key}={value}")
            if query_params:
                parameters = "?" + "&".join(query_params)
                url += parameters
        
        timestamp = str(int(time.time() * 1000))
        
        body = ""
        if data:
            body = json.dumps(data, separators=(',', ':'))
        
        # 署名生成（pathのみを使用 - パラメータは含めない）
        headers = {
            "API-KEY": self.api_key,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": self._sign(method, timestamp, path, body)  # パラメータを含めない
        }
        
        # デバッグログの詳細化
        print(f"DEBUG: URL: {url}")
        print(f"DEBUG: Method: {method}")
        print(f"DEBUG: Path for signature: {path}")  # パラメータなしのパス
        print(f"DEBUG: Headers: {headers}")
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            else:
                headers["Content-Type"] = "application/json"
                response = requests.post(url, headers=headers, data=body)
            
            result = response.json()
            print(f"DEBUG: Response: {result}")
            return result
        except Exception as e:
            return {"status": -1, "messages": [{"message_string": str(e)}]}

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
    
    def get_margin_leverage(self, symbol):
        """レバレッジ情報を取得"""
        path = "/v1/account/margin"
        response = self._request("GET", path)
        
        if response.get("status") == 0:
            for item in response.get("data", []):
                if item.get("symbol") == symbol:
                    return item.get("leverage", 2)  # デフォルト2倍
        return 2
    
    def get_balance(self):
        """資産情報を取得"""
        path = "/v1/account/assets"
        return self._request("GET", path)

    def get_order_status(self, order_id):
        """注文状態を取得"""
        path = "/v1/orders"
        params = {"orderId": order_id}
        return self._request("GET", path, params=params)

    def get_closed_orders(self, symbol, date=None):
        """約定済み注文履歴を取得"""
        path = "/v1/closedOrders"
        params = {"symbol": symbol}
        if date:
            params["date"] = date
        return self._request("GET", path, params=params)

    def get_margin_info(self):
        """信用取引の証拠金情報を取得"""
        path = "/v1/account/margin"
        return self._request("GET", path)
    
    def get_account_summary(self):
        """総合的な口座情報を取得"""
        path = "/v1/account/summary"
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
    def get_total_balance(self):
        """
        GMOコインの総資産額を取得する（現物＋信用取引評価額）
        
        Returns:
        float: 総資産額（JPY）
        """
        try:
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
                            self.logger.debug(f"JPY現物残高: {available:,.0f}円")
                        else:
                            # 暗号資産現物
                            available = float(asset.get("available", 0))
                            if available > 0:
                                # 現在価格を取得
                                current_price = self.get_current_price(f"{symbol}_jpy")
                                if current_price > 0:
                                    asset_value = available * current_price
                                    total_balance += asset_value
                                    self.logger.debug(f"{symbol.upper()}現物: {available:.6f} ({asset_value:,.0f}円)")
            
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
                                
                                self.logger.debug(f"建玉 {position_id}: {symbol} {side} "
                                                f"サイズ: {size}, エントリー価格: {price}, "
                                                f"現在価格: {current_price}, 評価損益: {profit:,.0f}円")
            except Exception as e:
                self.logger.debug(f"個別ポジション情報取得時のエラー（無視）: {e}")
            
            self.logger.info(f"GMOコイン総資産額: {total_balance:,.0f}円")
            return total_balance
            
        except Exception as e:
            self.logger.error(f"GMOコイン総資産額取得エラー: {e}", exc_info=True)
            return 0.0

    def __init__(self, initial_capital=100000, test_mode=True):
        """
        トレーディングボットの初期化
        
        Parameters:
        initial_capital (float): 初期資金 (円)
        test_mode (bool): テストモード（少額取引）の場合はTrue
        """
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
                'sender'     : os.environ["EMAIL_ADDRESS_PIP"],
                'password'   : os.environ["EMAIL_PASSWORD_PIP"],
                'recipient'  : "tomokomi1107@gmail.com"
            },
            'send_on_entry': True,
            'send_on_exit' : True,
            'send_on_error': True,
            'daily_report' : True
        }

        self.exchange_settings_gmo = {
            'api_key': os.environ["GMO_API_KEY"],
            'api_secret': os.environ["GMO_API_SECRET"],
            'live_trade': True
        }

        # ポジションID管理を追加
        self.position_ids = {symbol: None for symbol in self.symbols}  # ポジションIDを保存

        # GMO APIの初期化
        self.gmo_api = None
        if self.exchange_settings_gmo['api_key'] and self.exchange_settings_gmo['api_secret']:
            # GMOCoinAPIクラスが同じファイル内にある場合
            self.gmo_api = GMOCoinAPI(
                self.exchange_settings_gmo['api_key'],
                self.exchange_settings_gmo['api_secret']
            )

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
        
        # 保存されたポジション情報を読み込む
        self.load_positions()

        self.logger.info(f"=== ボット初期化完了 （初期資金: {initial_capital:,}円, テストモード: {test_mode}) ===")

    def setup_logging(self):
        """ロギングの設定"""
        self.logger = logging.getLogger('crypto_bot')
        self.logger.setLevel(logging.INFO)
        
        # すでにハンドラーが設定されている場合はクリア
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 時間ベースのログファイル名 (日付と時刻を含む)
        log_filename = os.path.join(self.log_dir, f'crypto_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # ファイルハンドラー（UTF-8エンコーディングを指定）
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラー（UTF-8エンコーディング対応）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # フォーマット
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラーの追加
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
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
            # メール通知
            self._send_email(subject, message)
            
            self.logger.info(f"通知送信完了: {subject}")
        except Exception as e:
            self.logger.error(f"通知送信エラー: {e}")
    
    def _send_email(self, subject, body):
        """メール送信
        
        Parameters:
        subject (str): メールの件名
        body (str): メールの本文
        """
        if not self.notification_settings['enabled']:
            return

        try:
            email_settings = self.notification_settings['email']

            # 必要な設定がすべて揃っているか確認
            required_settings = ['smtp_server', 'smtp_port', 'sender', 'password', 'recipient']
            missing_settings = [s for s in required_settings if not email_settings.get(s)]
            
            if missing_settings:
                self.logger.warning(f"メール設定が不完全です。不足: {missing_settings}")
                return
            
            msg = MIMEMultipart()
            msg['From'] = email_settings['sender']
            msg['To'] = email_settings['recipient']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_settings['smtp_server'], email_settings['smtp_port'])
            server.starttls()
            server.login(email_settings['sender'], email_settings['password'])
            text = msg.as_string()
            server.sendmail(email_settings['sender'], email_settings['recipient'], text)
            server.quit()

            self.logger.info("メール送信成功")
            
        except Exception as e:
            self.logger.error(f"メール送信エラー: {e}")

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
                json.dump({'total_profit': self.total_profit, 'updated_at': datetime.now().isoformat()}, f)
        except Exception as e:
            self.logger.error(f"利益データ保存エラー: {e}")
    
    def _load_last_increase_date(self):
        """前回の増資日を読み込む"""
        increase_file = os.path.join(self.base_dir, 'last_increase.json')
        
        if os.path.exists(increase_file):
            try:
                with open(increase_file, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data.get('last_increase_date', datetime.now().isoformat()))
            except Exception as e:
                self.logger.error(f"増資日データ読み込みエラー: {e}")
                return datetime.now()
        else:
            # ファイルが存在しない場合は現在の日付を設定
            self._save_last_increase_date(datetime.now())
            return datetime.now()

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
        current_date = datetime.now()
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
        """現在のポジション情報をファイルに保存（ポジションID含む）"""
        positions_file = os.path.join(self.base_dir, 'positions.json')
        
        positions_data = {}
        for symbol in self.symbols:
            if self.positions[symbol] is not None:
                positions_data[symbol] = {
                    'position': self.positions[symbol],
                    'entry_price': self.entry_prices[symbol],
                    'entry_time': self.entry_times[symbol].isoformat() if self.entry_times[symbol] else None,
                    'entry_size': self.entry_sizes[symbol],
                    'position_id': self.position_ids.get(symbol)  # ポジションIDを追加
                }
        
        try:
            with open(positions_file, 'w') as f:
                json.dump(positions_data, f)
            self.logger.info("ポジション情報を保存しました")
        except Exception as e:
            self.logger.error(f"ポジション情報の保存エラー: {e}")

    def load_positions(self):
        """保存されたポジション情報を読み込む（ポジションID含む）"""
        positions_file = os.path.join(self.base_dir, 'positions.json')
        
        if os.path.exists(positions_file):
            try:
                with open(positions_file, 'r') as f:
                    positions_data = json.load(f)
                
                for symbol, data in positions_data.items():
                    if symbol in self.symbols:
                        self.positions[symbol] = data['position']
                        self.entry_prices[symbol] = data['entry_price']
                        self.entry_sizes[symbol] = data.get('entry_size', 0)
                        self.position_ids[symbol] = data.get('position_id')  # ポジションIDを読み込む
                        if data['entry_time']:
                            self.entry_times[symbol] = datetime.fromisoformat(data['entry_time'])
                        else:
                            self.entry_times[symbol] = None
                
                self.logger.info("保存されたポジション情報を読み込みました")
                self.print_positions_info()
            except Exception as e:
                self.logger.error(f"ポジション情報の読み込みエラー: {e}")
    
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
                    holding_time = datetime.now() - self.entry_times[symbol]
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
        """GMOコインから最新の価格を取得
        
        Parameters:
        symbol (str): BitBank形式の通貨ペア (例: 'btc_jpy')
        
        Returns:
        float: 最新価格（取得失敗時は0）
        """
        try:
            # シンボルをGMOコイン形式に変換（例: 'btc_jpy' -> 'BTC_JPY'）
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper())
            
            # GMOコインの公開APIを使用
            url = f'https://api.coin.z.com/public/v1/ticker?symbol={gmo_symbol}'
            
            # リクエストヘッダー（オプション）
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # API制限を考慮した遅延
            elapsed = time.time() - self.last_api_call
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                time.sleep(sleep_time)
            
            # リクエスト実行
            self.logger.debug(f"GMOコイン価格取得: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            self.last_api_call = time.time()
            
            # レスポンスを処理
            data = response.json()
            
            if data.get('status') == 0 and 'data' in data:
                # dataがリスト形式の場合の処理
                ticker_data = data['data']
                if isinstance(ticker_data, list):
                    # リストから該当する通貨ペアのデータを見つける
                    for item in ticker_data:
                        if item.get('symbol') == gmo_symbol:
                            last_price = float(item.get('last', 0))
                            self.logger.info(f"{symbol} 現在価格: {last_price}")
                            return last_price
                    
                    # 該当する通貨ペアが見つからなかった場合
                    self.logger.warning(f"{symbol}（{gmo_symbol}）の価格データが見つかりません")
                    return 0
                else:
                    # リストでない場合（単一オブジェクトの場合）
                    last_price = float(ticker_data.get('last', 0))
                    self.logger.debug(f"{symbol} 現在価格: {last_price}")
                    return last_price
            else:
                error_msg = data.get('messages', [{"message_string": "不明なエラー"}])[0].get("message_string", "不明なエラー") if data.get('messages') else "不明なエラー"
                self.logger.error(f"GMOコイン価格取得エラー: {error_msg}")
                return 0
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"価格取得APIリクエストエラー: {e}")
            # リトライを試みる（1回のみ）
            try:
                time.sleep(2)  # 2秒待機
                response = requests.get(url, headers=headers, timeout=10)
                self.last_api_call = time.time()
                data = response.json()
                
                if data.get('status') == 0 and 'data' in data:
                    ticker_data = data['data']
                    if isinstance(ticker_data, list):
                        # リストから該当する通貨ペアのデータを見つける
                        for item in ticker_data:
                            if item.get('symbol') == gmo_symbol:
                                last_price = float(item.get('last', 0))
                                self.logger.debug(f"{symbol} 現在価格（リトライ成功）: {last_price}")
                                return last_price
                    else:
                        last_price = float(ticker_data.get('last', 0))
                        self.logger.debug(f"{symbol} 現在価格（リトライ成功）: {last_price}")
                        return last_price
            except Exception:
                pass
            return 0
        except ValueError as e:
            self.logger.error(f"価格取得JSONパースエラー: {e}")
            return 0
        except Exception as e:
            self.logger.error(f"価格取得未知のエラー: {e}")
            return 0
    
    def create_backup(self):
        """データのバックアップを作成"""
        try:
            # バックアップ時間を更新
            self.last_backup_time = time.time()
            
            # バックアップディレクトリ名
            backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    def api_call(self, url):
        """API制限を考慮したリクエスト関数
        
        Parameters:
        url (str): リクエストURL
        
        Returns:
        dict: JSONレスポンス
        """
        # API制限を守るための遅延
        elapsed = time.time() - self.last_api_call
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            time.sleep(sleep_time)
            
        # リクエスト実行
        try:
            self.logger.debug(f"APIリクエスト: {url}")
            response = requests.get(url, timeout=10)  # タイムアウト設定を追加
            self.last_api_call = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"APIリクエストエラー: {e}")
            # リトライを数回試みる
            for retry in range(3):
                self.logger.info(f"リトライ中... ({retry+1}/3)")
                time.sleep(2)  # リトライ間隔を長めに
                try:
                    response = requests.get(url, timeout=10)
                    self.last_api_call = time.time()
                    return response.json()
                except Exception as retry_e:
                    self.logger.error(f"リトライ失敗: {retry_e}")
            
            # 全てのリトライが失敗した場合
            return {'success': 0, 'data': {'code': -1}}
        except ValueError as e:
            # JSON解析エラー
            self.logger.error(f"JSONパースエラー: {e}")
            return {'success': 0, 'data': {'code': -2}}
        except Exception as e:
            # その他の予期しないエラー
            self.logger.error(f"APIリクエスト未知のエラー: {e}")
            return {'success': 0, 'data': {'code': -3}}
        
    def verify_positions(self):
        """GMOコインの信用取引建玉を使用してポジションを検証する"""
        self.logger.info("取引所ポジション情報の検証を開始...")
        
        # ローカルポジション情報のバックアップ
        local_positions_backup = {
            'positions': self.positions.copy(),
            'entry_prices': self.entry_prices.copy(),
            'entry_times': {k: v for k, v in self.entry_times.items()},
            'entry_sizes': self.entry_sizes.copy()
        }
        
        # 不整合カウンタ
        inconsistencies = 0
        resolved = 0
        
        for symbol in self.symbols:
            try:
                # APIレート制限を考慮して少し待機
                time.sleep(0.5)
                
                # 通貨シンボルからコインを取得
                coin = symbol.split('_')[0]
                
                # GMOコインから建玉情報を取得
                position_details = self.get_position_details(coin)
                
                # 建玉の有無を判断
                has_actual_position = False
                actual_side = None
                actual_size = 0.0
                
                if position_details and position_details["positions"]:
                    # ネットポジションで判断
                    net_size = position_details["net_size"]
                    if abs(net_size) >= self.min_order_sizes.get(symbol, 0.001):
                        has_actual_position = True
                        actual_side = "long" if net_size > 0 else "short"
                        actual_size = abs(net_size)

                # ログ追加: 建玉情報を常に表示
                self.logger.info(f"{symbol}: ボット状態={self.positions[symbol]}, "
                            f"実際の建玉={actual_side if has_actual_position else 'なし'}")

                # ボットの記録と実際のポジションを比較
                if self.positions[symbol] is not None:  # ボットではポジションあり
                    if not has_actual_position:  # 実際にはポジションなし
                        self.logger.warning(f"{symbol}: ボットではポジション有りだが、実際には建玉が見つかりませんでした。")
                        
                        # ポジション情報をリセット
                        self.positions[symbol] = None
                        self.entry_prices[symbol] = 0
                        self.entry_times[symbol] = None
                        self.entry_sizes[symbol] = 0
                        inconsistencies += 1
                        resolved += 1
                    else:
                        # ポジションタイプの確認
                        if self.positions[symbol] != actual_side:
                            self.logger.warning(f"{symbol}: ポジションタイプの不一致。記録: {self.positions[symbol]}, 実際: {actual_side}")
                            self.positions[symbol] = actual_side
                            inconsistencies += 1
                            resolved += 1
                        
                        # ポジションサイズの誤差を確認
                        size_diff_pct = abs(self.entry_sizes[symbol] - actual_size) / max(self.entry_sizes[symbol], actual_size) * 100
                        
                        if size_diff_pct > 5:  # 5%以上の差がある場合
                            self.logger.warning(f"{symbol}: 記録上のポジションサイズ({self.entry_sizes[symbol]:.6f})と"
                                            f"実際のサイズ({actual_size:.6f})に{size_diff_pct:.1f}%の差があります。更新します。")
                            self.entry_sizes[symbol] = actual_size
                            inconsistencies += 1
                            resolved += 1
                
                elif has_actual_position:  # ボットではポジションなし、実際にはポジションあり
                    self.logger.warning(f"{symbol}: ボットではポジションなしだが、実際には{actual_size:.6f}の{actual_side}建玉が見つかりました。")

                    # 現在価格の取得（参照用）
                    current_price = self.get_current_price(symbol)
                    
                    # ポジション情報を作成
                    self.positions[symbol] = actual_side
                    self.entry_prices[symbol] = current_price  # 現在価格を参考値として使用
                    self.entry_times[symbol] = datetime.now() - timedelta(hours=1)  # 1時間前からの保有と仮定
                    self.entry_sizes[symbol] = actual_size
                    
                    inconsistencies += 1
                    resolved += 1
                    
                    self.logger.info(f"{symbol}: ポジション情報を更新しました（推定値）。")
                
            except Exception as e:
                self.logger.error(f"{symbol}のポジション検証中にエラー: {e}", exc_info=True)
        
        # 変更があった場合のみ保存
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
            self.logger.debug(f"Searching for symbol: {gmo_symbol}")
            # 信用取引の建玉情報を取得（symbolパラメータは使わない）
            margin_response = self.gmo_api.get_margin_positions(gmo_symbol)

            # デバッグ：レスポンス全体を表示
            self.logger.debug(f"=== get_balance margin_response ===")
            self.logger.debug(f"Type: {type(margin_response)}")
            self.logger.debug(f"Content: {margin_response}")
            
            # ステータスコードを確認
            status = margin_response.get("status")
            self.logger.debug(f"API status: {status}")
            
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
                
                self.logger.debug(f"Searching for symbol: {gmo_symbol}")
                
                for position in positions:
                    if position.get("symbol") == gmo_symbol:
                        size = float(position.get("size", 0))
                        side = position.get("side")
                        
                        # 買い建玉はプラス、売り建玉はマイナスとして扱う
                        if side == "BUY":
                            total_position_size += size
                        else:  # SELL
                            total_position_size -= size
                        
                        self.logger.debug(f"{gmo_symbol} {side}建玉: {size}")
                
                # 絶対値を返す
                result = abs(total_position_size)
                self.logger.debug(f"{coin}の建玉合計: {result}")
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
        try:
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません")
                return None
            
            # 通貨ペアに変換
            symbol = f"{coin}_jpy"
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
            
            # 建玉情報を取得
            margin_response = self.gmo_api.get_margin_positions(gmo_symbol)
            
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

    def close_margin_position(self, symbol, position_id=None):
        """GMOコインの信用取引ポジションを決済"""
        try:
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません")
                return {'success': False, 'error': "GMO API not initialized"}
            
            # 通貨ペアをGMO形式に変換
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
            
            # ポジションIDが指定されていない場合は保存されているIDを使用
            if position_id is None:
                position_id = self.position_ids.get(symbol)
            
            if position_id is None:
                self.logger.error(f"{symbol}のポジションIDが見つかりません")
                return {'success': False, 'error': "Position ID not found"}

            # 現在のポジション情報を取得
            positions_response = self.gmo_api.get_margin_positions(gmo_symbol)
            
            if positions_response.get("status") != 0:
                error_msg = positions_response.get("messages", [{}])[0].get("message_string", "不明なエラー")
                self.logger.error(f"ポジション情報取得失敗: {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # データ形式を確認
            positions_data = positions_response.get("data", {})
            
            # データが辞書の場合はlistを取得、リストの場合はそのまま使用
            if isinstance(positions_data, dict):
                positions = positions_data.get("list", [])
            else:
                positions = positions_data
            
            # ポジションを探す
            target_position = None
            positions_data = positions_response.get("data", {})
            positions = positions_data.get("list", []) if isinstance(positions_data, dict) else positions_data
            self.logger.warning(f"positions: {positions}")
            for pos in positions:
                if str(pos.get("positionId")) == str(position_id):
                    target_position = pos
                    break
            
            if not target_position:
                self.logger.warning(f"ポジションID {position_id} が見つかりません")
                return {'success': False, 'error': "Position not found"}
            
            # 決済方向の判定
            side = "SELL" if target_position.get("side") == "BUY" else "BUY"
            size = target_position.get("size", 0)
            
            self.logger.info(f"GMOコイン信用取引決済: {gmo_symbol} {side} {size} (ポジションID: {position_id})")
            
            # 決済注文実行（成行注文）
            response = self.gmo_api.close_position(
                symbol=gmo_symbol,
                position_id=int(position_id),
                size=str(size),
                side=side,
                position_type="MARKET"
            )
            
            if response.get("status") == 0:
                order_id = str(response.get("data"))
                self.logger.info(f"決済注文成功: 注文ID={order_id}")
                
                # ポジションIDをクリア
                self.position_ids[symbol] = None
                
                return {'success': True, 'order_id': order_id}
            else:
                error_messages = response.get("messages", [])
                error_msg = error_messages[0].get("message_string", "Unknown error") if error_messages else "Unknown error"
                self.logger.error(f"決済注文エラー: {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            self.logger.error(f"ポジション決済エラー: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
        
    def reset_positions(self):
        """すべてのポジション情報をリセットする（緊急時用）"""
        self.logger.warning("すべてのポジション情報をリセットします")
        
        # すべてのポジションをクリア
        for symbol in self.symbols:
            self.positions[symbol] = None
            self.entry_prices[symbol] = 0
            self.entry_times[symbol] = None
            self.entry_sizes[symbol] = 0
        
        # 保存
        self.save_positions()
        self.logger.info("ポジション情報のリセットが完了しました")

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
        """GMOコインでの信用取引注文を実行する"""
        if not self.exchange_settings_gmo['live_trade']:
            self.logger.info(f"模擬注文（信用取引）: {symbol} {order_type} {size}")
            return {'success': True, 'order_id': 'simulation_order_id', 'executed_size': size}
        
        try:
            # GMO API が利用可能か確認
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません")
                return {'success': False, 'error': "GMO API not initialized", 'executed_size': 0}
            
            # 通貨ペアをGMO形式に変換
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
            
            # 最小注文量の調整と適切なフォーマット
            if symbol == "xrp_jpy":
                size = round(size / 10) * 10
                if size < 10:
                    size = 10
            elif symbol == "eth_jpy":
                size = round(size, 2)
            elif symbol == "ltc_jpy":
                size = int(size)
            elif symbol == "doge_jpy":
                size = round(size / 10) * 10
                if size < 10:
                    size = 10
            elif symbol == "sol_jpy":
                size = round(size, 1)
            elif symbol == "bcc_jpy":  # 追加
                size = round(size, 1)  # BCHは小数点以下2桁
            else:
                size_str = str(size)
            
            # 注文前の既存ポジションを記録（後で新規ポジションを判別するため）
            existing_positions = set()
            try:
                positions_response = self.gmo_api.get_margin_positions(gmo_symbol)
                if positions_response.get("status") == 0:
                    positions_data = positions_response.get("data", {})
                    positions = positions_data.get("list", []) if isinstance(positions_data, dict) else positions_data
                    
                    for pos in positions:
                        if pos.get("symbol") == gmo_symbol:
                            existing_positions.add(pos.get("positionId"))
            except Exception as e:
                self.logger.warning(f"既存ポジション取得エラー: {e}")

            # 新規注文のみを扱う
            side = "BUY" if order_type == "buy" else "SELL"
            self.logger.info(f"GMOコイン信用取引新規注文: {gmo_symbol} {side} {size}")

            # 新規注文データ
            order_data = {
                "symbol": gmo_symbol,
                "side": side,
                "executionType": "MARKET",
                "size": str(size),
                "settlePosition": "OPEN"  # 常に新規
            }
            
            # 新規注文の場合はmarginTradeTypeを指定
            order_data["marginTradeType"] = "SHORT" if side == "SELL" else "LONG"
            
            response = self.gmo_api._request("POST", "/v1/order", data=order_data)
            
            if response.get("status") == 0:
                order_id = str(response.get("data"))
                self.logger.info(f"注文成功: {gmo_symbol} {side} {size}, 注文ID: {order_id}")
                
                # ポジションID取得のリトライメカニズム
                position_id = None
                max_retries = 5
                retry_interval = 2  # 秒
                
                for retry in range(max_retries):
                    time.sleep(retry_interval)
                    
                    try:
                        # 最新のポジション情報を取得
                        positions_response = self.gmo_api.get_margin_positions(gmo_symbol)
                        if positions_response.get("status") == 0:
                            positions_data = positions_response.get("data", {})
                            positions = positions_data.get("list", []) if isinstance(positions_data, dict) else positions_data
                            
                            # 新規ポジションを探す（既存のポジションIDにないもの）
                            for pos in positions:
                                if (pos.get("symbol") == gmo_symbol and 
                                    pos.get("side") == side and
                                    pos.get("positionId") not in existing_positions):
                                    
                                    position_id = pos.get("positionId")
                                    self.logger.info(f"新規ポジションID取得成功: {position_id} (試行: {retry + 1}/{max_retries})")
                                    break
                            
                            if position_id:
                                break
                                
                    except Exception as e:
                        self.logger.warning(f"ポジションID取得試行{retry + 1}失敗: {e}")
                    
                    if retry < max_retries - 1:
                        self.logger.info(f"ポジションID未取得。再試行します... ({retry + 2}/{max_retries})")
                
                # ポジションIDが取得できた場合は保存
                if position_id:
                    self.position_ids[symbol] = position_id
                    self.logger.info(f"ポジションID保存成功: {symbol} = {position_id}")
                else:
                    self.logger.warning(f"ポジションIDを取得できませんでした: {symbol}")
                    # 注文は成功しているので、後でverify_positionsで同期を試みる
                
                return {'success': True, 'order_id': order_id, 'executed_size': size, 'position_id': position_id}
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
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません")
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
            positions_response = self.gmo_api.get_margin_positions(gmo_symbol)
            
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
        """GMOコインでの注文の約定状況を確認する"""
        try:
            # GMOコインAPIが利用可能か確認
            if not self.gmo_api:
                self.logger.error("GMOコインAPIが初期化されていません")
                return 0
            
            # 通貨ペアをGMO形式に変換
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
            
            # 注文情報を取得するエンドポイント設定
            path = "/v1/orders"
            params = {"orderId": str(order_id)}  # 文字列として渡す
            
            # GMOコインのAPIリクエスト
            response = self.gmo_api._request("GET", path, params=params)
            
            # デバッグログ
            self.logger.info(f"GMOコイン注文確認API応答: {response}")
            
            if response.get('status') == 0:
                data = response.get('data', {})
                
                # dataの中のlistを取得
                order_list = data.get('list', [])
                
                if not order_list:
                    self.logger.warning("注文リストが空です")
                    return 0
                
                # 最初の注文情報を取得（通常は1つだけ）
                order_info = order_list[0]
                
                # 注文情報から必要な値を取得
                status = order_info.get('status')
                executed_size = float(order_info.get('executedSize', '0'))
                
                self.logger.info(f"注文詳細: 状態={status}, 約定サイズ={executed_size}")
                
                # GMOコインのステータス: WAITING, ORDERED, EXECUTED, CANCELED, EXPIRED
                if status == 'EXECUTED':
                    self.logger.info(f"注文完全約定: {executed_size}")
                    return executed_size
                elif status == 'ORDERED' and executed_size > 0:
                    self.logger.info(f"注文部分約定: {executed_size}")
                    return executed_size
                elif status == 'WAITING':
                    self.logger.info(f"注文待機中: {status}")
                    return 0
                else:
                    self.logger.info(f"注文状態: {status}")
                    return 0
                    
            else:
                # エラーメッセージの取得
                error_messages = response.get('messages', [])
                if error_messages:
                    error_msg = error_messages[0].get('message_string', '不明なエラー')
                    self.logger.error(f"GMOコイン注文確認エラー: {error_msg}")
                    
                    # エラーコードが10002（注文が存在しない）の場合
                    # 成行注文は即座に約定して履歴に移動することがあるため、取引履歴を確認
                    if response.get('messages') and any(msg.get('message_code') == '10002' for msg in response.get('messages', [])):
                        self.logger.info("注文が履歴に移動した可能性があります。取引履歴を確認します。")
                        
                        # 最新の取引履歴を確認
                        history_path = "/v1/closedOrders"
                        current_date = datetime.now().strftime("%Y%m%d")
                        history_params = {
                            "symbol": gmo_symbol,
                            "date": current_date
                        }
                        
                        history_response = self.gmo_api._request("GET", history_path, params=history_params)
                        
                        if history_response.get('status') == 0:
                            history_data = history_response.get('data', {})
                            orders = history_data.get('list', [])
                            
                            for order in orders:
                                if str(order.get('orderId')) == str(order_id):
                                    executed_size = float(order.get('executedSize', '0'))
                                    if executed_size > 0:
                                        self.logger.info(f"取引履歴から注文を確認: 約定量 {executed_size}")
                                        return executed_size
                    
                    return 0
            
        except Exception as e:
            self.logger.error(f"GMOコイン注文確認エラー: {e}", exc_info=True)
            return 0
            
    def execute_order_with_confirmation(self, symbol, order_type, size, max_retries=3):
        """確実に注文を実行し、ポジションが実際に保有されていることを確認する"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"注文試行 {attempt+1}/{max_retries}: {symbol} {order_type} {size}")
                
                # 現在のポジション状態を確認
                current_position = self.positions.get(symbol)

                # 注文実行（margin=True は不要）
                order_result = self.place_order(symbol, order_type, size)
                
                if not order_result['success']:
                    self.logger.error(f"注文失敗: {order_result.get('error', '不明なエラー')}")
                    time.sleep(2)  # 再試行前に待機
                    continue
                        
                # 注文IDを取得
                order_id = order_result.get('order_id')
                if not order_id:
                    self.logger.error("注文成功したが注文IDがありません")
                    time.sleep(2)
                    continue
                
                # 約定確認（数回試行）
                for check in range(5):
                    time.sleep(3)  # 約定を待つ
                    executed_size = self.check_order_execution(order_id, symbol)
                    
                    if executed_size > 0:
                        self.logger.info(f"注文約定確認完了: {symbol} {order_type} サイズ:{executed_size}")
                        
                        # 決済注文の場合、ポジション情報をクリア
                        if current_position is not None:
                            if (current_position == 'long' and order_type == 'sell') or (current_position == 'short' and order_type == 'buy'):
                                self.logger.info(f"{symbol}のポジションを決済しました")
                                self.positions[symbol] = None
                                self.entry_prices[symbol] = 0
                                self.entry_times[symbol] = None
                                self.entry_sizes[symbol] = 0
                        
                        return {
                            'success': True, 
                            'order_id': order_id, 
                            'executed_size': executed_size,
                            'balance': executed_size
                        }
                    
                    self.logger.info(f"約定待機中... 試行 {check+1}/5")
                
                # 約定が確認できない場合でも、成行注文は通常即座に約定するので注意が必要
                self.logger.warning(f"注文は送信されましたが、約定確認に時間がかかっています: {symbol} {order_type}")
                
                # 念のため、ポジション情報を確認してみる
                time.sleep(2)
                position_details = self.get_position_details(symbol.split('_')[0])
                
                if position_details and position_details.get('positions'):
                    # ポジションが存在する場合は成功とみなす
                    net_size = position_details.get('net_size', 0)
                    if abs(net_size) > 0:
                        self.logger.info(f"ポジション情報から建玉を確認: サイズ {net_size}")
                        return {
                            'success': True, 
                            'order_id': order_id, 
                            'executed_size': abs(net_size),
                            'balance': abs(net_size)
                        }
                
                # それでも確認できない場合
                self.logger.error(f"約定も建玉も確認できませんでした: {symbol} {order_type}")
                
            except Exception as e:
                self.logger.error(f"注文処理中のエラー: {e}")
            
            # 再試行前にランダムな待機時間
            wait_time = 3 + random.uniform(0, 2)
            self.logger.info(f"{wait_time:.1f}秒待機して再試行します")
            time.sleep(wait_time)
        
        # 全ての試行が失敗
        self.logger.error(f"すべての試行が失敗しました: {symbol} {order_type} {size}")
        return {'success': False, 'error': "最大試行回数を超えました", 'executed_size': 0}

    def clear_cache(self):
        """キャッシュデータを全てクリアする"""
        self.logger.info("キャッシュデータをクリアします...")
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
        self.logger.info("キャッシュデータのクリアが完了しました")

    def find_valid_date(self, symbol, timeframe, max_days_back=5):
        """有効なデータが存在する日付を検索し、複数日のデータを組み合わせて必要なデータポイント数を確保する
        
        Parameters:
        symbol (str): 通貨ペア
        timeframe (str): 時間枠
        max_days_back (int): 何日前まで検索するか
        
        Returns:
        tuple: (str, list) 最新の有効な日付文字列（YYYYMMDD）とデータを補完するための追加日付リスト、両方Noneの場合はデータ不足
        """
        # 必要なデータポイント数を取得
        required_data_points = self.get_required_data_points(timeframe)
        self.logger.info(f"{symbol} {timeframe}の必要データポイント数: {required_data_points}")
        
        # 結果格納用変数の初期化
        valid_dates = []
        total_data_points = 0
        
        # キャッシュキー
        cache_key = f"{symbol}_{timeframe}"
        
        # キャッシュされた情報があるか確認
        if cache_key in self.valid_dates:
            last_valid_date = self.valid_dates[cache_key]
            last_check_time = self.valid_dates.get(f"{cache_key}_time", 0)
            
            # 1時間以内のキャッシュなら再利用
            if time.time() - last_check_time < 3600:
                # キャッシュされたデータを取得
                cached_data = self.get_cached_data(symbol, timeframe, last_valid_date)
                
                if not cached_data.empty:
                    total_data_points += len(cached_data)
                    valid_dates.append(last_valid_date)
                    
                    # キャッシュだけで必要データポイント数を満たしている場合
                    if total_data_points >= required_data_points:
                        self.logger.info(f"キャッシュされた有効な日付とデータを使用: {last_valid_date}")
                        return last_valid_date, []
                    
                    self.logger.info(f"キャッシュされたデータ({last_valid_date})のポイント数: {len(cached_data)}")
        
        # 現在日から過去に遡って必要なデータポイント数を確保する
        dates_checked = []
        for days_back in range(max_days_back):
            test_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            
            # すでにチェック済みの日付はスキップ
            if test_date in dates_checked or test_date in valid_dates:
                continue
                
            dates_checked.append(test_date)
            
            # APIからデータを取得
            df = self.get_cached_data(symbol, timeframe, test_date)
            
            if not df.empty:
                points_in_date = 24
                self.logger.info(f"{test_date}のデータポイント数: {points_in_date}")
                
                # 日付とデータポイント数を記録
                valid_dates.append(test_date)
                total_data_points += points_in_date
                
                # データポイント数が十分であれば終了
                if total_data_points >= required_data_points:
                    # 最新の有効な日付をキャッシュに保存
                    newest_date = valid_dates[0]
                    self.valid_dates[cache_key] = newest_date
                    self.valid_dates[f"{cache_key}_time"] = time.time()
                    
                    # 最新日付のみを返す
                    self.logger.info(f"{symbol} {timeframe}の有効な日付を発見: {newest_date}")
                    
                    # 重要な変更点：もはや追加日付は返さない
                    return newest_date
            else:
                self.logger.info(f"{test_date}のデータが空です")
            
            # APIレート制限を考慮して少し待機
            time.sleep(0.5)
        
        # 少なくとも1つの有効な日付が見つかった場合
        if valid_dates:
            newest_date = valid_dates[0]
            self.valid_dates[cache_key] = newest_date
            self.valid_dates[f"{cache_key}_time"] = time.time()
            
            # 最新日付と追加日付リストを返す（データポイント数は不足しているが、部分的なデータとして使用）
            additional_dates = valid_dates[1:] if len(valid_dates) > 1 else []
            self.logger.info(f"{symbol} {timeframe}の日付を発見しましたが、データが不足しています。合計: {total_data_points}/{required_data_points}")
            self.logger.info(f"最新日付: {newest_date}, 追加日付: {additional_dates}")
            return newest_date, additional_dates
        
        # 有効な日付が見つからなかった
        self.logger.warning(f"{symbol} {timeframe}の有効な日付が見つかりませんでした")
        return None, None

    def get_required_data_points(self, timeframe):
        """時間枠に基づいて必要なデータポイント数を返す"""
        if timeframe == '5min':
            return 24  # 2時間分 (5分足 * 12 * 24時間)
        elif timeframe == '30min':
            return 2 * 24  # 1日分 (1時間足 * 24時間 * 1日)
        elif timeframe == '1hour':
            return 24  # 1日分 (1時間足 * 24時間 * 1日)
        elif timeframe == '1day':
            return 7  # 1週間分 (日足 * 7日)
        else:
            return 100  # デフォルト (大半の時間枠で十分な数)

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
            date_str = datetime.now().strftime('%Y%m%d')
            self.logger.info(f"15分足データの日付として本日の日付 {date_str} を試行")
        # 日付が指定されていない場合は有効な日付を検索（検索範囲拡大）
        if date_str is None:
            date_str = self.find_valid_date(symbol, timeframe, max_days_back=7)  # 5→7日に拡大
            if date_str is None:
                self.logger.warning(f"{symbol} {timeframe}の有効な日付が見つかりませんでした")
                return pd.DataFrame()
                
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
            max_retries = 3
            retry_delay = 2  # 初期遅延2秒
            
            while retry_count < max_retries:
                try:
                    # APIからデータを取得
                    url = f'https://public.bitbank.cc/{symbol}/candlestick/{timeframe}/{date_str}'
                    
                    # ユーザーエージェントを追加（サーバー側でブロックされないように）
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    self.logger.debug(f"APIデータを取得: {url}")
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
                                self.logger.debug(f"APIデータをキャッシュに保存: {cache_file}")
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
                    sleep_time = retry_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
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
            # デバッグ情報を出力
            self.logger.debug(f"特徴量計算: データサイズ={len(df)}, カラム={df.columns.tolist()}")
            
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
                    self.logger.debug(f"{col}にNaN値があります。デフォルト値で埋めます。")
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
            
            # 結果の検証
            debug_info = {
                'RSI': {'min': df['RSI'].min(), 'max': df['RSI'].max(), 'mean': df['RSI'].mean()} if 'RSI' in df.columns else {},
                'CCI': {'min': df['CCI'].min(), 'max': df['CCI'].max(), 'mean': df['CCI'].mean()} if 'CCI' in df.columns else {},
                'データ行数': len(df)
            }
            self.logger.debug(f"特徴量計算結果: {debug_info}")
            
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
            
    def debug_technical_indicators(self, symbol):
        """特定の通貨ペアのテクニカル指標値をデバッグ表示"""
        self.logger.info(f"{symbol}のテクニカル指標をデバッグ表示します")
        
        # 最新の有効な日付を取得
        valid_date = self.find_valid_date(symbol, '5min')
        if not valid_date:
            self.logger.warning(f"{symbol}の有効な日付が見つかりませんでした")
            return
        
        # 5分足データの取得
        df_5min = self.get_cached_data(symbol, '5min', valid_date)
        if df_5min.empty:
            self.logger.warning(f"{symbol}のデータを取得できませんでした")
            return
        
        # データフレームの基本情報表示
        self.logger.info(f"データサイズ: {len(df_5min)}行 x {len(df_5min.columns)}列")
        self.logger.info(f"カラム: {df_5min.columns.tolist()}")
        
        # データ型の確認
        self.logger.info("データ型:")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_5min.columns:
                self.logger.info(f"- {col}: {df_5min[col].dtype}")
        
        # 最初の5行を表示
        self.logger.info("最初の5行のデータ:")
        for i in range(min(5, len(df_5min))):
            row = df_5min.iloc[i]
            self.logger.info(f"行 {i}: open={row.get('open', 'N/A')}, high={row.get('high', 'N/A')}, "
                            f"low={row.get('low', 'N/A')}, close={row.get('close', 'N/A')}, "
                            f"volume={row.get('volume', 'N/A')}")
        
        # 特徴量計算前のNaN値を確認
        nan_counts = {col: df_5min[col].isna().sum() for col in df_5min.columns 
                     if col in ['open', 'high', 'low', 'close', 'volume']}
        self.logger.info(f"特徴量計算前のNaN値: {nan_counts}")
        
        # 特徴量計算
        df_with_features = self.build_features(df_5min.copy())
        
        # 特徴量計算後の結果確認
        if 'RSI' in df_with_features.columns:
            rsi_values = df_with_features['RSI'].tail(5).values
            self.logger.info(f"RSI (最後の5件): {rsi_values}")
        
        if 'CCI' in df_with_features.columns:
            cci_values = df_with_features['CCI'].tail(5).values
            self.logger.info(f"CCI (最後の5件): {cci_values}")
        
        # 指標の統計情報
        for indicator in ['RSI', 'CCI', 'EMA_short', 'EMA_long', 'ATR']:
            if indicator in df_with_features.columns:
                stats = {
                    'min': df_with_features[indicator].min(),
                    'max': df_with_features[indicator].max(),
                    'mean': df_with_features[indicator].mean(),
                    'null_count': df_with_features[indicator].isna().sum()
                }
                self.logger.info(f"{indicator} 統計: {stats}")
        
        self.logger.info(f"{symbol}のテクニカル指標デバッグ表示を完了しました")
        
    def get_api_data(self, symbol, timeframe, date_str):
        """APIから直接データを取得する（キャッシュを使わない）
        
        Parameters:
        symbol (str): 通貨ペア
        timeframe (str): 時間枠 (5min, 1hour, 1day など)
        date_str (str): 日付文字列 (YYYYMMDD)
        
        Returns:
        pandas.DataFrame: 価格データ
        """
        # エクスポネンシャルバックオフによるリトライ実装
        retry_count = 0
        max_retries = 3
        retry_delay = 2  # 初期遅延2秒
        
        while retry_count < max_retries:
            try:
                # APIからデータを取得
                url = f'https://public.bitbank.cc/{symbol}/candlestick/{timeframe}/{date_str}'
                
                # ユーザーエージェントを追加（サーバー側でブロックされないように）
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                self.logger.debug(f"APIデータを直接取得: {url}")
                response = requests.get(url, headers=headers, timeout=15)
                
                # レート制限を遵守するための遅延
                time.sleep(self.rate_limit_delay + random.uniform(0, 0.5))
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 成功したらキャッシュに保存（オプション、ここでは保存しない）
                    if data.get('success') == 1 and 'candlestick' in data.get('data', {}):
                        # データの変換処理
                        try:
                            candles = data['data']['candlestick'][0]['ohlcv']
                            
                            # 空のデータをチェック
                            if not candles:
                                self.logger.warning(f"APIからの空データ: {symbol} {timeframe} {date_str}")
                                return pd.DataFrame()
                            
                            # データフレームに変換
                            df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
                            
                            # データ型変換
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            
                            # データの整合性チェック
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
                                
                                # 前方値補完
                                df = df.fillna(method='ffill')
                                
                                # それでも残るNaN値は後方値補完
                                df = df.fillna(method='bfill')
                                
                                # それでも残るNaN値（両端など）は列の平均値で補完
                                for col in ['open', 'high', 'low', 'close', 'volume']:
                                    if df[col].isna().any():
                                        df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 0)
                            
                            return df
                        except Exception as e:
                            self.logger.error(f"データ変換エラー: {e}", exc_info=True)
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
                sleep_time = retry_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                self.logger.info(f"{sleep_time:.1f}秒待機してリトライします")
                time.sleep(sleep_time)
        
        return pd.DataFrame()  # 空のDataFrameを返す（失敗時）


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
            'doge_jpy': {'buy': 0.51, 'sell': 0.51},
            'sol_jpy':  {'buy': 0.51, 'sell': 0.51},
            'xrp_jpy':  {'buy': 0.51, 'sell': 0.51},
            'ltc_jpy':  {'buy': 0.51, 'sell': 0.51},
            'ada_jpy':  {'buy': 0.51, 'sell': 0.51},
            'eth_jpy':  {'buy': 0.51, 'sell': 0.51},
            'bcc_jpy':  {'buy': 0.51, 'sell': 0.51},
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
        atr_score_long = calc_score(df_5min['ATR'], df_5min['ATR'].quantile(0.8), df_5min['ATR'].quantile(0.2), reverse=True)
        atr_score_short = calc_score(df_5min['ATR'], df_5min['ATR'].quantile(0.8), df_5min['ATR'].quantile(0.2), reverse=True)
        
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

        if symbol == 'bcc_jpy':
            df_5min.loc[df_5min['ADX'] < 20, 'buy_signal'] = False
            df_5min.loc[df_5min['ADX'] < 20, 'sell_signal'] = False

            df_5min.loc[df_5min['macd_score_long'] == 0, 'buy_signal'] = False
            df_5min.loc[df_5min['macd_score_short'] == 0, 'sell_signal'] = False

        if symbol == 'doge_jpy':
            df_5min.loc[df_5min['ADX'] < 20, 'buy_signal'] = False
            df_5min.loc[df_5min['ADX'] < 20, 'sell_signal'] = False

        if symbol == 'sol_jpy':
            df_5min.loc[df_5min['ADX'] < 20, 'buy_signal'] = False
            df_5min.loc[df_5min['ADX'] < 20, 'sell_signal'] = False

        if symbol == 'ada_jpy':
            df_5min.loc[df_5min['ADX'] < 20, 'buy_signal'] = False
            df_5min.loc[df_5min['ADX'] < 20, 'sell_signal'] = False

            df_5min.loc[df_5min['atr_score_long'] > 0.65, 'buy_signal'] = False
            df_5min.loc[df_5min['atr_score_short'] > 0.65, 'sell_signal'] = False

        if symbol == 'ltc_jpy':
            df_5min.loc[df_5min['ADX'] < 20, 'buy_signal'] = False
            df_5min.loc[df_5min['ADX'] < 20, 'sell_signal'] = False

        if symbol == 'eth_jpy':
            df_5min.loc[df_5min['ADX'] < 20, 'buy_signal'] = False
            df_5min.loc[df_5min['ADX'] < 20, 'sell_signal'] = False

        if symbol == 'xrp_jpy':
            df_5min.loc[df_5min['ADX'] < 20, 'buy_signal'] = False
            df_5min.loc[df_5min['ADX'] < 20, 'sell_signal'] = False

            df_5min.loc[df_5min['atr_score_short'] > 0.70, 'sell_signal'] = False
            df_5min.loc[df_5min['adx_score_short'] < 0.1, 'sell_signal'] = False


        #5分足データのMAフィルターを適用（修正：各時点の1つ前のレコードのMA25を使用）
        if 'EMA_long' in df_5min.columns and len(df_5min) > 1:  # 少なくとも2レコード必要
            # 最初のレコードはスキップし、2レコード目から処理
            for i in range(1, len(df_5min)):
                # インデックスではなく、位置（整数）でアクセス
                # 現在の行と1つ前の行
                current_row = df_5min.iloc[i]
                previous_row = df_5min.iloc[i-1]
                
                # 1つ前のレコードのMA25と価格を取得
                prev_5min_MA25 = previous_row['EMA_long']
                prev_5min_price = previous_row['close']
                
                # 1つ前の価格がMA25以下の場合は現在の買いシグナルを無効化
                if prev_5min_price <= prev_5min_MA25:
                    df_5min.iloc[i, df_5min.columns.get_loc('buy_signal')] = False
                
                # 1つ前の価格がMA25以上の場合は現在の売りシグナルを無効化
                if prev_5min_price >= prev_5min_MA25:
                    df_5min.iloc[i, df_5min.columns.get_loc('sell_signal')] = False

        return df_5min

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
                'bcc_jpy': {'max_deviation': 1.1},  # 4.5%を超える乖離でシグナル無効化
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
        """複数通貨ペアのバックテスト実行"""
        self.logger.info(f"=== バックテスト開始 ({days_to_test}日間) ===")
        results = {}
        start_profit = self.total_profit  # バックテスト開始時の利益

        # 変更部分: 個別の symbol_balance ではなく全シンボル共通の残高を使用
        total_balance = self.initial_capital + self.total_profit

        # 取引の詳細ログ用のリスト
        trade_logs = []

        def run_backtest(symbol):
            # 変更部分: total_balanceを関数内で参照できるようにする
            nonlocal total_balance
            total_trades = 0
            symbol_profit = 0
            wins = 0

            # ロングとショートの統計分離
            long_trades = 0
            long_wins = 0
            long_profit = 0
            short_trades = 0
            short_wins = 0
            short_profit = 0

            # 各通貨ペアごとの残高管理（これが重要）
            symbol_balance = self.initial_capital + self.total_profit

            self.logger.info(f"=== {symbol.upper()} のバックテスト ===")

            # 日付範囲の設定
            if live_mode:
                day_range = range(days_to_test, 0, -1)
            else:
                day_range = range(days_to_test, 0, -1)

            previous_day_offset = None  # 前回処理した日付のオフセットを記録する変数
            previous_day_data = None    # 前日のデータを保存する変数

            for day_offset in day_range:
                current_date = datetime.now() - timedelta(days=day_offset)
                date_str = current_date.strftime('%Y%m%d')
                
                # 前日の日付を計算
                previous_date = current_date - timedelta(days=1)
                previous_date_str = previous_date.strftime('%Y%m%d')

                # 15分足データの取得（当日分）
                df_5min_current = self.get_cached_data(symbol, '15min', date_str)
                if df_5min_current.empty:
                    self.logger.warning(f"{date_str}の{symbol}の15分足データを取得できませんでした。")
                    continue
                
                # 前日の15分足データを取得
                df_5min_previous = self.get_cached_data(symbol, '15min', previous_date_str)
                
                # 前日データと当日データを結合（前日データが取得できた場合）
                if not df_5min_previous.empty:
                    # 両方のデータフレームを結合
                    df_5min_combined = pd.concat([df_5min_previous, df_5min_current])
                    # タイムスタンプでソート
                    df_5min_combined = df_5min_combined.sort_values('timestamp')
                    # 重複を削除（万が一の場合）
                    df_5min_combined = df_5min_combined.drop_duplicates(subset=['timestamp'])
                    
                    self.logger.info(f"{symbol}の結合データ: 前日={len(df_5min_previous)}本 + 当日={len(df_5min_current)}本 = 合計{len(df_5min_combined)}本")
                    
                    # 特徴量計算に結合データを使用
                    df_5min_full = self.build_features(df_5min_combined.copy())
                else:
                    self.logger.warning(f"{previous_date_str}の{symbol}の前日データを取得できなかったため、当日データのみで処理します。")
                    # 前日データがない場合は当日データのみを使用
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

                # ★修正：シグナル生成後に最新の24時間分（96レコード）のデータを抽出★
                expected_records = 96  # 24時間分の15分足データ

                try:
                    df_hourly = pd.concat(hourly_candles).sort_values('timestamp')
                    # 時間足データの特徴量も計算
                    df_hourly = self.build_features(df_hourly)

                    # 処理中のDataFrameを一時的に保存（_get_entry_reasonで参照するため）
                    self.df_5min = df_5min_full  # 結合された全データを保存
                    
                    # ★修正：シグナル生成を先に行う（結合後の全データで実行）★
                    df_5min_full = self.generate_signals_with_sentiment(symbol, df_5min_full, df_hourly)
                    
                    # データフレームを時間順にソートしてから最新の96レコードを抽出
                    if 'timestamp' in df_5min_full.columns:
                        # タイムスタンプでソート
                        df_5min_full = df_5min_full.sort_values('timestamp').reset_index(drop=True)
                        
                        # 最新の96レコードを抽出（またはデータ量が少ない場合は全て）
                        if len(df_5min_full) > expected_records:
                            df_5min = df_5min_full.iloc[-expected_records:].copy().reset_index(drop=True)
                            self.logger.info(f"シグナル生成後の最新24時間分データ: {len(df_5min)}本（合計{len(df_5min_full)}本から抽出）")
                        else:
                            df_5min = df_5min_full.copy().reset_index(drop=True)
                            self.logger.info(f"データ総数が96レコード未満のため全データを使用: {len(df_5min)}本")
                    else:
                        # timestampカラムがない場合は、インデックスで最新の96レコードを抽出
                        df_5min_full = df_5min_full.reset_index(drop=True)
                        if len(df_5min_full) > expected_records:
                            df_5min = df_5min_full.iloc[-expected_records:].copy().reset_index(drop=True)
                        else:
                            df_5min = df_5min_full.copy().reset_index(drop=True)
                        self.logger.warning(f"タイムスタンプカラムがないため、インデックスで最新データを抽出: {len(df_5min)}本")
                    
                    # データ抽出後の検証
                    if df_5min.empty:
                        self.logger.error(f"データ抽出後にデータフレームが空になりました: {symbol}")
                        continue
                        
                    # 必要なカラムが存在するか確認
                    required_columns = ['buy_signal', 'sell_signal']
                    missing_columns = [col for col in required_columns if col not in df_5min.columns]
                    if missing_columns:
                        self.logger.error(f"必要なカラムが不足しています ({symbol}): {missing_columns}")
                        continue
                        
                    self.logger.info(f"データ抽出完了: {symbol}, 最終データ数: {len(df_5min)}")
                    

                    # 通貨ペアごとの利確・損切り設定
                    profit_loss_settings = self._get_dynamic_profit_loss_settings(symbol, df_5min)
                    long_profit_take = profit_loss_settings['long_profit_take']
                    long_stop_loss = profit_loss_settings['long_stop_loss']
                    short_profit_take = profit_loss_settings['short_profit_take']
                    short_stop_loss = profit_loss_settings['short_stop_loss']

                    # バックテスト処理
                    position = None
                    entry_price = 0
                    entry_time = None
                    entry_rsi = None
                    entry_cci = None
                    entry_reason = ""
                    entry_sentiment = {}
                    self.last_sentiment_time = None

                    # バックテストの主要ループ - 各ローソク足ごとの処理
                    # 当日分のデータのみを処理
                    for i in range(len(df_5min)):
                        row = df_5min.iloc[i]
                        price = row['close']
                        timestamp = row['timestamp'] if 'timestamp' in row else df_5min.index[i]

                        # 現在のテクニカル指標値
                        current_rsi = row['RSI'] if 'RSI' in row else None
                        current_cci = row['CCI'] if 'CCI' in row else None

                        # ポジションがない場合のエントリー判断
                        if position is None:
                            # 修正: 前のローソク足のシグナルもチェック (2連続のシグナルを確認)
                            # 最低でも2つのローソク足が必要
                            if i > 0:
                                previous_row = df_5min.iloc[i-1]
                                
                                # 買いシグナルが現在と前の足で両方Trueの場合のみエントリー
                                if row['buy_signal'] and previous_row['buy_signal']:
                                    # buy_scoreとsell_scoreを取得
                                    buy_score_value = row.get('buy_score', 0)
                                    sell_score_value = row.get('sell_score', 0)

                                    # ここで取引サイズを計算
                                    order_size = self.TRADE_SIZE / price
                                    order_size = self.adjust_order_size(symbol, order_size)
                                    entry_amount = order_size * price

                                    position = 'long'
                                    entry_price = price
                                    entry_time = timestamp
                                    entry_rsi = current_rsi
                                    entry_cci = current_cci
                                    entry_sentiment = self.sentiment.copy()
                                    
                                    self.entry_sizes[symbol] = order_size
                                    
                                    # 変更部分: symbol_balance の代わりに total_balance を使用
                                    balance_before_entry = total_balance
                                    total_balance -= entry_amount
                                    balance_after_entry = total_balance

                                    # エントリー理由の判定
                                    entry_reason = self._get_entry_reason(symbol, row, 'long')

                                    # エントリーログの出力
                                    self.log_entry(symbol, 'long', entry_price, entry_time, entry_rsi, entry_cci, row.get('ATR', 0), row.get('ADX', 0), entry_reason, entry_sentiment)

                                # 売りシグナルが現在と前の足で両方Trueの場合のみエントリー
                                elif row['sell_signal'] and previous_row['sell_signal']:
                                    # buy_scoreとsell_scoreを取得
                                    buy_score_value = row.get('buy_score', 0)
                                    sell_score_value = row.get('sell_score', 0)
                                    
                                    # ここで取引サイズを計算
                                    order_size = self.TRADE_SIZE / price
                                    order_size = self.adjust_order_size(symbol, order_size)
                                    entry_amount = order_size * price
                                    
                                    position = 'short'
                                    entry_price = price
                                    entry_time = timestamp
                                    entry_rsi = current_rsi
                                    entry_cci = current_cci
                                    entry_sentiment = self.sentiment.copy()
                                    
                                    self.entry_sizes[symbol] = order_size
                                    
                                    # 変更部分: symbol_balance の代わりに total_balance を使用
                                    balance_before_entry = total_balance
                                    total_balance -= entry_amount
                                    balance_after_entry = total_balance

                                    # エントリー理由の判定
                                    entry_reason = self._get_entry_reason(symbol, row, 'short')

                                    # エントリーログの出力
                                    self.log_entry(symbol, 'short', entry_price, entry_time, entry_rsi, entry_cci, row.get('ATR', 0), row.get('ADX', 0), entry_reason, entry_sentiment)

                        # ポジションがある場合のイグジット判断
                        elif position == 'long':
                            # 動的なエグジットレベルを計算
                            exit_levels = self.calculate_dynamic_exit_levels(symbol, df_5min, 'long', entry_price)
                            
                            # 計算された価格を使用
                            if price >= exit_levels['take_profit_price'] or price <= exit_levels['stop_loss_price']:
                                # 取引結果の計算
                                exit_price = exit_levels['take_profit_price'] if price >= exit_levels['take_profit_price'] else exit_levels['stop_loss_price']
                                profit = (exit_price - entry_price) / entry_price * self.TRADE_SIZE
                                profit *= 0.9976  # 手数料
                                profit_pct = (exit_price - entry_price) / entry_price * 100
                                
                                # 修正5: エントリー時の金額と決済時の金額を明確に
                                entry_amount = self.entry_sizes[symbol] * entry_price
                                exit_amount = self.entry_sizes[symbol] * exit_price
                                
                                # 修正6: 残高更新方法の改善
                                balance_before_exit = total_balance
                                total_balance += exit_amount  # 売却代金を残高に追加
                                balance_after_exit = total_balance

                                # 保有時間の計算
                                if isinstance(timestamp, pd.Timestamp) and isinstance(entry_time, pd.Timestamp):
                                    holding_time = timestamp - entry_time
                                    hours = holding_time.total_seconds() / 3600
                                else:
                                    hours = 0  # タイムスタンプが不適切な場合

                                # 決済理由の判定
                                if profit > 0:
                                    exit_reason = "利益確定"
                                else:
                                    exit_reason = "損切り"

                                # 詳細ログ出力
                                self.log_exit(symbol, 'long', exit_price, entry_price, timestamp, profit, profit_pct, exit_reason, hours, entry_sentiment)

                                # 取引記録
                                trade_data = {
                                    'symbol': symbol,
                                    'type': 'long',
                                    'entry_price': entry_price,
                                    'entry_time': entry_time,
                                    'entry_rsi': entry_rsi,
                                    'entry_cci': entry_cci,
                                    'buy_score': row.get('buy_score_scaled', 0),  # 修正: buy_score_scaled を使用
                                    'sell_score': row.get('sell_score_scaled', 0),  # 修正: sell_score_scaled を使用
                                    'entry_reason': entry_reason,
                                    'exit_price': exit_price,
                                    'exit_time': timestamp,
                                    'size': self.entry_sizes[symbol],
                                    'entry_amount': entry_amount,
                                    'balance_after_entry': balance_after_entry,
                                    'balance_after_exit': balance_after_exit,
                                    'profit': profit,
                                    'profit_pct': profit_pct,
                                    'exit_reason': exit_reason,
                                    'holding_hours': hours,
                                    'sentiment_bullish': entry_sentiment.get('bullish', 0),
                                    'sentiment_bearish': entry_sentiment.get('bearish', 0),
                                    'sentiment_volatility': entry_sentiment.get('volatility', 0),
                                    # 以下のスコア情報を追加
                                    'rsi_score_long': row.get('rsi_score_long', 0),
                                    'rsi_score_short': row.get('rsi_score_short', 0),
                                    'cci_score_long': row.get('cci_score_long', 0),
                                    'cci_score_short': row.get('cci_score_short', 0),
                                    'volume_score': row.get('volume_score', 0),
                                    'bb_score_long': row.get('bb_score_long', 0),
                                    'bb_score_short': row.get('bb_score_short', 0),
                                    'ma_score_long': row.get('ma_score_long', 0),
                                    'ma_score_short': row.get('ma_score_short', 0),
                                    'adx_score_long': row.get('adx_score_long', 0),
                                    'adx_score_short': row.get('adx_score_short', 0),
                                    'mfi_score_long': row.get('mfi_score_long', 0),
                                    'mfi_score_short': row.get('mfi_score_short', 0),
                                    'atr_score_long': row.get('atr_score_long', 0),
                                    'atr_score_short': row.get('atr_score_short', 0),
                                    'macd_score_long': row.get('macd_score_long', 0),
                                    'macd_score_short': row.get('macd_score_short', 0),
                                    'ema_deviation': row.get('ema_deviation', 0)
                                }
                                trade_logs.append(trade_data)

                                # 統計更新
                                symbol_profit += profit
                                long_profit += profit
                                self.total_profit += profit
                                total_trades += 1
                                long_trades += 1
                                wins += profit > 0
                                long_wins += profit > 0
                                position = None

                        # ショートポジションのイグジット
                        elif position == 'short':
                            # 動的なエグジットレベルを計算
                            exit_levels = self.calculate_dynamic_exit_levels(symbol, df_5min, 'short', entry_price)
                            
                            # 計算された価格を使用
                            if price <= exit_levels['take_profit_price'] or price >= exit_levels['stop_loss_price']:
                                # 取引結果の計算（ショートの場合は反転）
                                exit_price = exit_levels['take_profit_price'] if price <= exit_levels['take_profit_price'] else exit_levels['stop_loss_price']
                                profit = (entry_price - exit_price) / entry_price * self.TRADE_SIZE
                                profit *= 0.9976  # 手数料
                                profit_pct = (entry_price - exit_price) / entry_price * 100
                                
                                # 修正5: エントリー時の金額を明確に
                                entry_amount = self.entry_sizes[symbol] * entry_price
                                
                                # 修正6: 残高更新方法の改善
                                balance_before_exit = total_balance
                                total_balance += entry_amount + profit  # 証拠金返却＋利益/損失
                                balance_after_exit = total_balance

                                # 保有時間の計算
                                if isinstance(timestamp, pd.Timestamp) and isinstance(entry_time, pd.Timestamp):
                                    holding_time = timestamp - entry_time
                                    hours = holding_time.total_seconds() / 3600
                                else:
                                    hours = 0

                                # 決済理由の判定
                                if profit > 0:
                                    exit_reason = "利益確定"
                                else:
                                    exit_reason = "損切り"

                                # 詳細ログ出力
                                self.log_exit(symbol, 'short', exit_price, entry_price, timestamp, profit, profit_pct, exit_reason, hours, entry_sentiment)

                                # 取引記録
                                trade_data = {
                                    'symbol': symbol,
                                    'type': 'short',
                                    'entry_price': entry_price,
                                    'entry_time': entry_time,
                                    'entry_rsi': entry_rsi,
                                    'entry_cci': entry_cci,
                                    'buy_score': row.get('buy_score_scaled', 0),  # 修正: buy_score_scaled を使用
                                    'sell_score': row.get('sell_score_scaled', 0),  # 修正: sell_score_scaled を使用
                                    'entry_reason': entry_reason,
                                    'exit_price': exit_price,
                                    'exit_time': timestamp,
                                    'size': self.entry_sizes[symbol],
                                    'entry_amount': entry_amount,
                                    'balance_after_entry': balance_after_entry,
                                    'balance_after_exit': balance_after_exit,
                                    'profit': profit,
                                    'profit_pct': profit_pct,
                                    'exit_reason': exit_reason,
                                    'holding_hours': hours,
                                    'sentiment_bullish': entry_sentiment.get('bullish', 0),
                                    'sentiment_bearish': entry_sentiment.get('bearish', 0),
                                    'sentiment_volatility': entry_sentiment.get('volatility', 0),
                                    # 以下のスコア情報を追加
                                    'rsi_score_long': row.get('rsi_score_long', 0),
                                    'rsi_score_short': row.get('rsi_score_short', 0),
                                    'cci_score_long': row.get('cci_score_long', 0),
                                    'cci_score_short': row.get('cci_score_short', 0),
                                    'volume_score': row.get('volume_score', 0),
                                    'bb_score_long': row.get('bb_score_long', 0),
                                    'bb_score_short': row.get('bb_score_short', 0),
                                    'ma_score_long': row.get('ma_score_long', 0),
                                    'ma_score_short': row.get('ma_score_short', 0),
                                    'adx_score_long': row.get('adx_score_long', 0),
                                    'adx_score_short': row.get('adx_score_short', 0),
                                    'mfi_score_long': row.get('mfi_score_long', 0),
                                    'mfi_score_short': row.get('mfi_score_short', 0),
                                    'atr_score_long': row.get('atr_score_long', 0), 
                                    'atr_score_short': row.get('atr_score_short', 0), 
                                    'macd_score_long': row.get('macd_score_long', 0),
                                    'macd_score_short': row.get('macd_score_short', 0),
                                    'ema_deviation': row.get('ema_deviation', 0)  # EMA乖離を追加
                                }
                                trade_logs.append(trade_data)

                                # 統計更新
                                symbol_profit += profit
                                short_profit += profit
                                self.total_profit += profit
                                total_trades += 1
                                short_trades += 1
                                wins += profit > 0
                                short_wins += profit > 0
                                position = None


                        # DataFrameの参照を削除
                        if hasattr(self, 'df_5min'):
                            del self.df_5min

                except Exception as e:
                    self.logger.error(f"{date_str}の{symbol}バックテスト中にエラー: {e}")
                    continue

            # 結果の集計
            if total_trades > 0:
                win_rate = wins / total_trades * 100
                avg_profit = symbol_profit / total_trades

                self.logger.info(f"・トレード回数: {total_trades} 回（ロング: {long_trades}回、ショート: {short_trades}回）")
                self.logger.info(f"・勝率: {win_rate:.2f}%")
                if long_trades > 0:
                    self.logger.info(f"・ロング勝率: {(long_wins / long_trades * 100):.2f}%")
                if short_trades > 0:
                    self.logger.info(f"・ショート勝率: {(short_wins / short_trades * 100):.2f}%")
                self.logger.info(f"・平均利益: {avg_profit:.2f} 円")
                self.logger.info(f"・トータル利益: {symbol_profit:.2f} 円（ロング: {long_profit:.2f}円、ショート: {short_profit:.2f}円）")

                return {
                    'trades': total_trades,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'symbol_profit': symbol_profit,
                    'long_trades': long_trades,
                    'long_win_rate': (long_wins / long_trades * 100) if long_trades > 0 else 0,
                    'long_profit': long_profit,
                    'short_trades': short_trades,
                    'short_win_rate': (short_wins / short_trades * 100) if short_trades > 0 else 0,
                    'short_profit': short_profit
                }
            else:
                self.logger.info("😅 トレードが発生しませんでした")
                return {'trades': 0}

        # マルチスレッディングでバックテストを並列実行
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_backtest, symbol): symbol for symbol in self.symbols}
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    self.logger.error(f"{symbol}のバックテスト中にエラー: {e}")

        # 取引詳細の最終サマリー出力
        self.output_trade_summary(trade_logs)

        # Excelファイルに取引ログを保存
        if trade_logs:
            self.save_trade_logs_to_excel(trade_logs)

        # バックテストの全体結果
        backtest_profit = self.total_profit - start_profit
        self.logger.info(f"\n=== バックテスト全体結果 ===")
        self.logger.info(f"・バックテスト収益: {backtest_profit:,.2f} 円")

        # バックテスト結果をファイルに保存
        self.save_backtest_result(results, days_to_test, start_profit)

        return results

    def log_entry(self, symbol, position_type, entry_price, entry_time, entry_rsi, entry_cci, entry_atr, entry_adx, entry_reason, entry_sentiment):
        """エントリー情報のログ出力"""
        # None値のチェックを追加
        rsi_str = f"{entry_rsi:.1f}" if entry_rsi is not None else "N/A"
        cci_str = f"{entry_cci:.1f}" if entry_cci is not None else "N/A"
        atr_str = f"{entry_atr:.2f}" if entry_atr is not None else "N/A"
        adx_str = f"{entry_adx:.1f}" if entry_adx is not None else "N/A"  # ADXを追加
        
        self.logger.info(f"[エントリー] {symbol} {'ロング' if position_type == 'long' else 'ショート'} @ {entry_price:.2f}円 (時刻: {entry_time})")
        self.logger.info(f"  → RSI: {rsi_str}, CCI: {cci_str}, ATR: {atr_str}, ADX: {adx_str}")  # ADXを追加
        self.logger.info(f"  → 理由: {entry_reason}")
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
            self.logger.info(f"  理由: {trade['entry_reason']} → {trade['exit_reason']}")
            self.logger.info("")

    def save_trade_logs_to_excel(self, trade_logs):
        """
        取引ログを通貨ペアごとにシート分割して Excel (.xlsx) で保存する
        追加シートとして各通貨ペアのロング/ショート統計情報も保存
        新機能：ポジション保有状況を横棒グラフで可視化
        """
        # 出力ファイル名を作成
        trade_log_file = os.path.join(
            self.log_dir,
            f'backtest_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )

        # DataFrame化
        df_trades = pd.DataFrame(trade_logs)

        if 'symbol' not in df_trades.columns:
            raise ValueError("'symbol' 列が見つかりません。")

        desired_columns = [
            'symbol', 'type', 'entry_price', 'entry_time', 'entry_rsi', 'entry_cci', 
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
            'macd_score_long', 'macd_score_short'
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
                self._create_overall_timeline_chart_fixed(writer, df_trades)

        self.logger.info(f"取引ログを Excel に保存しました: {trade_log_file}")

    def _create_overall_timeline_chart_fixed(self, writer, df_trades):
        """
        全通貨ペアのポジション保有状況を一つのタイムラインで表示（修正版）
        条件付き書式を正しく適用し、ポジションタイムラインシートは作成しない
        
        Parameters:
        writer: ExcelWriter オブジェクト
        df_trades: 取引データのDataFrame
        """
        try:
            workbook = writer.book
            
            # 時間範囲を決定
            all_times = []
            for _, trade in df_trades.iterrows():
                if pd.notna(trade['entry_time']) and pd.notna(trade['exit_time']):
                    all_times.extend([trade['entry_time'], trade['exit_time']])
            
            if not all_times:
                self.logger.warning("総合タイムライン作成：有効な時間データがありません")
                return
            
            # 時間を datetime に変換
            all_times = [pd.to_datetime(t) if not isinstance(t, pd.Timestamp) else t for t in all_times]
            min_time = min(all_times)
            max_time = max(all_times)
            
            # 総合タイムラインシートを作成
            overall_sheet = workbook.add_worksheet('総合タイムライン')
            
            # 通貨ペアリストを取得
            symbols = sorted(df_trades['symbol'].unique())
            symbol_to_row = {symbol: i + 1 for i, symbol in enumerate(symbols)}
            
            # 時間軸データを準備（1時間単位）
            total_hours = int((max_time - min_time).total_seconds() // 3600) + 1
            max_display_hours = min(total_hours, 2200)  # 最大200時間まで表示（Excel制限対策）
            
            # ヘッダー設定
            headers = ['通貨ペア'] + [f'時間{i}' for i in range(max_display_hours)]
            for col, header in enumerate(headers):
                overall_sheet.write(0, col, header)
            
            # 通貨ペア名を書き込み
            for i, symbol in enumerate(symbols):
                overall_sheet.write(i + 1, 0, symbol)
            
            # 各ポジションの保有状況をマッピング（利益・損失情報付き）
            position_matrix = {}
            profit_matrix = {}  # 利益情報を別途保存
            
            for symbol in symbols:
                position_matrix[symbol] = [0] * max_display_hours  # 0: ポジションなし, 1: ロング, -1: ショート
                profit_matrix[symbol] = [0] * max_display_hours    # 0: ポジションなし, 1: 利益, -1: 損失
            
            # ポジションデータを時間軸にマッピング（利益・損失情報付き）
            for _, trade in df_trades.iterrows():
                if pd.notna(trade['entry_time']) and pd.notna(trade['exit_time']):
                    symbol = trade['symbol']
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    
                    # 基準時刻からの経過時間を計算（時間単位）
                    start_hour = int((entry_time - min_time).total_seconds() // 3600)
                    end_hour = int((exit_time - min_time).total_seconds() // 3600)
                    
                    position_value = 1 if trade['type'] == 'long' else -1
                    profit_value = 1 if trade.get('profit', 0) > 0 else -1  # 利益なら1、損失なら-1
                    
                    for hour in range(start_hour, min(end_hour + 1, max_display_hours)):
                        if 0 <= hour < max_display_hours:
                            position_matrix[symbol][hour] = position_value
                            profit_matrix[symbol][hour] = profit_value
            
            # 条件付き書式用のフォーマットを定義（利益・損失別）
            # ロングポジション
            long_profit_format = workbook.add_format({
                'bg_color': '#00AA00',  # 濃い緑（ロング利益）
                'font_color': '#FFFFFF',  # 白文字
                'align': 'center'
            })
            
            long_loss_format = workbook.add_format({
                'bg_color': '#90EE90',  # ライトグリーン（ロング損失）
                'font_color': '#006400',  # ダークグリーン
                'align': 'center'
            })
            
            # ショートポジション
            short_profit_format = workbook.add_format({
                'bg_color': '#CC0000',  # 濃い赤（ショート利益）
                'font_color': '#FFFFFF',  # 白文字
                'align': 'center'
            })
            
            short_loss_format = workbook.add_format({
                'bg_color': '#FFB6C1',  # ライトピンク（ショート損失）
                'font_color': '#8B0000',  # ダークレッド
                'align': 'center'
            })
            
            no_position_format = workbook.add_format({
                'bg_color': '#F5F5F5',  # ライトグレー
                'align': 'center'
            })
            
            # Excel列参照を正しく生成する関数
            def get_excel_column_letter(col_num):
                """
                数値からExcel列参照を生成する
                例: 0 -> A, 25 -> Z, 26 -> AA, 27 -> AB, ...
                """
                result = ""
                while col_num >= 0:
                    remainder = col_num % 26
                    result = chr(65 + remainder) + result  # 65 = 'A'のASCIIコード
                    col_num = (col_num // 26) - 1
                    if col_num < 0:
                        break
                return result

            # データ範囲を計算
            end_col_letter = get_excel_column_letter(max_display_hours)
            data_range = f'B2:{end_col_letter}{len(symbols) + 1}'
            
            # 修正：利益・損失に応じてフォーマットを適用
            # 各セルを個別にフォーマット（ポジションタイプ＋利益・損失で色分け）
            for row_idx, symbol in enumerate(symbols):
                for col_idx in range(max_display_hours):
                    cell_row = row_idx + 1  # データは2行目から開始（0-indexedなので+1）
                    cell_col = col_idx + 1  # データは2列目から開始（0-indexedなので+1）
                    
                    position_value = position_matrix[symbol][col_idx]
                    profit_value = profit_matrix[symbol][col_idx]
                    
                    # フォーマットの選択
                    if position_value == 1:  # ロングポジション
                        if profit_value == 1:  # 利益
                            format_to_use = long_profit_format
                            display_value = "L+"  # ロング利益
                        else:  # 損失
                            format_to_use = long_loss_format
                            display_value = "L-"  # ロング損失
                    elif position_value == -1:  # ショートポジション
                        if profit_value == 1:  # 利益
                            format_to_use = short_profit_format
                            display_value = "S+"  # ショート利益
                        else:  # 損失
                            format_to_use = short_loss_format
                            display_value = "S-"  # ショート損失
                    else:  # ポジションなし
                        format_to_use = no_position_format
                        display_value = ""  # 空白
                    
                    overall_sheet.write(cell_row, cell_col, display_value, format_to_use)
            
            # カラム幅を調整
            overall_sheet.set_column('A:A', 15)  # 通貨ペア列
            overall_sheet.set_column('B:' + get_excel_column_letter(max_display_hours), 3)  # 時間列（狭くして多くの時間を表示）
            
            # 説明テキストを追加（更新された凡例）
            overall_sheet.write(len(symbols) + 3, 0, '凡例:')
            overall_sheet.write(len(symbols) + 4, 0, 'L+ = ロング利益', long_profit_format)
            overall_sheet.write(len(symbols) + 5, 0, 'L- = ロング損失', long_loss_format)
            overall_sheet.write(len(symbols) + 6, 0, 'S+ = ショート利益', short_profit_format)
            overall_sheet.write(len(symbols) + 7, 0, 'S- = ショート損失', short_loss_format)
            overall_sheet.write(len(symbols) + 8, 0, '空白 = ポジションなし', no_position_format)
            
            # 時間範囲の説明
            overall_sheet.write(len(symbols) + 8, 0, f'期間: {min_time.strftime("%Y-%m-%d %H:%M")} ～ {max_time.strftime("%Y-%m-%d %H:%M")}')
            overall_sheet.write(len(symbols) + 9, 0, f'表示時間: {max_display_hours}時間（総時間: {total_hours}時間）')
            
            self.logger.info("総合タイムライングラフを作成しました")
            
        except Exception as e:
            self.logger.error(f"総合タイムライングラフ作成エラー: {e}", exc_info=True)

    def _create_overall_timeline_chart(self, workbook, timeline_sheet, position_data, min_time, max_time):
        """
        全通貨ペアのポジション保有状況を一つのタイムラインで表示
        
        Parameters:
        workbook: xlsxwriter workbook オブジェクト
        timeline_sheet: タイムラインシート
        position_data: ポジションデータのリスト
        min_time: 最小時刻
        max_time: 最大時刻
        """
        try:
            # 総合タイムラインシートを作成
            overall_sheet = workbook.add_worksheet('総合タイムライン')
            
            # 通貨ペアリストを取得
            symbols = sorted(list(set([p['symbol'] for p in position_data])))
            symbol_to_row = {symbol: i + 1 for i, symbol in enumerate(symbols)}
            
            # ヘッダー設定
            headers = ['通貨ペア'] + [f'時間{i}' for i in range(int((max_time - min_time).total_seconds() // 3600) + 1)]
            for col, header in enumerate(headers):
                overall_sheet.write(0, col, header)
            
            # 通貨ペア名を書き込み
            for i, symbol in enumerate(symbols):
                overall_sheet.write(i + 1, 0, symbol)
            
            # 時間軸データを準備（1時間単位）
            total_hours = int((max_time - min_time).total_seconds() // 3600) + 1
            
            # 各ポジションの保有状況をマッピング
            position_matrix = {}
            for symbol in symbols:
                position_matrix[symbol] = [0] * total_hours  # 0: ポジションなし, 1: ロング, -1: ショート
            
            # ポジションデータを時間軸にマッピング
            for pos in position_data:
                symbol = pos['symbol']
                start_hour = int(pos['entry_hours'])
                end_hour = int(pos['entry_hours'] + pos['duration_hours'])
                
                value = 1 if pos['type'] == 'long' else -1
                
                for hour in range(start_hour, min(end_hour + 1, total_hours)):
                    if hour >= 0 and hour < total_hours:
                        position_matrix[symbol][hour] = value
            
            # データをシートに書き込み
            for i, symbol in enumerate(symbols):
                for hour in range(min(total_hours, 100)):  # 最大100時間まで表示（Excel制限対策）
                    overall_sheet.write(i + 1, hour + 1, position_matrix[symbol][hour])
            
            # 条件付き書式を追加
            # ロングポジション（正の値）を緑色に
            long_format = workbook.add_format({
                'bg_color': '#90EE90',  # ライトグリーン
                'font_color': '#006400',  # ダークグリーン
                'align': 'center'
            })
            
            # ショートポジション（負の値）を赤色に
            short_format = workbook.add_format({
                'bg_color': '#FFB6C1',  # ライトピンク
                'font_color': '#8B0000',  # ダークレッド
                'align': 'center'
            })
            
            # ポジションなし（0）をグレーに
            no_position_format = workbook.add_format({
                'bg_color': '#F5F5F5',  # ライトグレー
                'align': 'center'
            })
            
            # 条件付き書式を適用
            data_range = f'B2:{chr(ord("A") + min(total_hours, 100))}{len(symbols) + 1}'
            
            # 条件付き書式を設定
            overall_sheet.conditional_format(data_range, {
                'type': 'cell',
                'criteria': '>',
                'value': 0,
                'format': long_format
            })
            
            overall_sheet.conditional_format(data_range, {
                'type': 'cell',
                'criteria': '<',
                'value': 0,
                'format': short_format
            })
            
            overall_sheet.conditional_format(data_range, {
                'type': 'cell',
                'criteria': '==',
                'value': 0,
                'format': no_position_format
            })
            
            # カラム幅を調整
            overall_sheet.set_column('A:A', 15)  # 通貨ペア列
            overall_sheet.set_column('B:CV', 3)  # 時間列（狭くして多くの時間を表示）
            
            # 説明テキストを追加
            overall_sheet.write(len(symbols) + 3, 0, '凡例:')
            overall_sheet.write(len(symbols) + 4, 0, '緑色 = ロングポジション')
            overall_sheet.write(len(symbols) + 5, 0, 'ピンク = ショートポジション')
            overall_sheet.write(len(symbols) + 6, 0, 'グレー = ポジションなし')
            
            # 時間範囲の説明
            overall_sheet.write(len(symbols) + 8, 0, f'期間: {min_time.strftime("%Y-%m-%d %H:%M")} ～ {max_time.strftime("%Y-%m-%d %H:%M")}')
            overall_sheet.write(len(symbols) + 9, 0, f'総時間: {total_hours}時間')
            
            self.logger.info("総合タイムライングラフを作成しました")
            
        except Exception as e:
            self.logger.error(f"総合タイムライングラフ作成エラー: {e}", exc_info=True)
    

    def save_backtest_result(self, results, days_to_test, start_profit):
        """バックテスト結果をJSONファイルに保存"""
        backtest_profit = self.total_profit - start_profit
        backtest_result_file = os.path.join(self.log_dir, f'backtest_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        try:
            with open(backtest_result_file, 'w') as f:
                json.dump({
                    'start_date': (datetime.now() - timedelta(days=days_to_test)).strftime("%Y-%m-%d"),
                    'end_date': datetime.now().strftime("%Y-%m-%d"),
                    'days_tested': days_to_test,
                    'total_profit': backtest_profit,
                    'results': {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv 
                                for kk, vv in v.items()} 
                            for k, v in results.items()}
                }, f, indent=4)
            self.logger.info(f"バックテスト結果を保存しました: {backtest_result_file}")
        except Exception as e:
            self.logger.error(f"バックテスト結果保存エラー: {e}")

    def _get_entry_reason(self, symbol, row, position_type):
        """エントリー理由を判定する補助メソッド（修正版）"""
        reasons = []

        def fmt(val, digits=1):
            return f"{val:.{digits}f}"

        # rowがSeriesの場合の値取得を修正
        def get_safe_value(data, key, default=0):
            """Seriesまたは辞書から安全に値を取得"""
            try:
                if hasattr(data, 'get'):
                    value = data.get(key, default)
                else:
                    value = getattr(data, key, default)
                
                # Seriesの場合は最初の値を取得
                if hasattr(value, 'iloc'):
                    return value.iloc[0] if len(value) > 0 else default
                elif hasattr(value, 'item'):
                    return value.item()
                else:
                    return value
            except:
                return default

        if symbol == "ltc_jpy":
            if position_type == 'long':
                reasons.append("常時ロング戦略")
            else:  # short
                if get_safe_value(row, 'is_range_bound', False):
                    reasons.append("レンジ相場")
                
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val > 65:
                    reasons.append(f"RSI過買い({fmt(rsi_val)})")
                
                # 前の価格との比較部分を修正
                if hasattr(row, 'name') and hasattr(self, 'df_5min') and row.name > 0:
                    try:
                        prev_close = self.df_5min.at[row.name - 1, 'close']
                        current_close = get_safe_value(row, 'close', 0)
                        if current_close < prev_close:
                            reasons.append(f"直近下落傾向（prev={fmt(prev_close)}, now={fmt(current_close)}）")
                    except:
                        pass  # エラーが発生した場合はスキップ

        elif symbol == "xrp_jpy":
            if position_type == 'long':
                rsi_val = get_safe_value(row, 'RSI', 0)
                if 50 < rsi_val < 70:
                    reasons.append(f"RSI適正レンジ({fmt(rsi_val)})")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val * 1.1:
                    reasons.append(f"出来高増加（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")

                close_val = get_safe_value(row, 'close', 0)
                open_val = get_safe_value(row, 'open', 0)
                if close_val > 0 and open_val > 0:
                    body = close_val - open_val
                    if body > 0:
                        reasons.append(f"陽線（open={fmt(open_val)}, close={fmt(close_val)}）")
                        high_val = get_safe_value(row, 'high', 0)
                        if high_val > 0:
                            upper_wick = high_val - max(close_val, open_val)
                            if upper_wick < body * 0.3:
                                reasons.append(f"小さな上ヒゲ（wick={fmt(upper_wick)}, body={fmt(body)}）")
            else:  # short
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val > 75:
                    reasons.append(f"RSI過買い({fmt(rsi_val)})")
                
                close_val = get_safe_value(row, 'close', 0)
                ema_short_val = get_safe_value(row, 'EMA_short', 0)
                if close_val < ema_short_val:
                    reasons.append(f"短期EMA下抜け（close={fmt(close_val)}, EMA_short={fmt(ema_short_val)}）")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val * 1.2:
                    reasons.append(f"出来高増加（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")

        elif symbol == "eth_jpy":
            if position_type == 'long':
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val < 40:
                    reasons.append(f"RSI過売り({fmt(rsi_val)})")
                
                cci_val = get_safe_value(row, 'CCI', 0)
                if cci_val < -100:
                    reasons.append(f"CCI過売り({fmt(cci_val)})")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val * 1.2:
                    reasons.append(f"出来高20%増加（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")
            else:  # short
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val > 70:
                    reasons.append(f"RSI過買い({fmt(rsi_val)})")
                
                # 前の価格との比較部分を修正
                if hasattr(row, 'name') and hasattr(self, 'df_5min') and row.name > 0:
                    try:
                        prev_close = self.df_5min.at[row.name - 1, 'close']
                        current_close = get_safe_value(row, 'close', 0)
                        if current_close < prev_close:
                            reasons.append(f"直近下落（prev={fmt(prev_close)}, now={fmt(current_close)}）")
                    except:
                        pass
                
                fib_level = get_safe_value(row, 'fib_level', 0)
                close_val = get_safe_value(row, 'close', 0)
                if fib_level > 0 and close_val < fib_level:
                    reasons.append(f"フィボナッチ0.618下抜け（close={fmt(close_val)}, fib={fmt(fib_level)}）")
                
                highest_high = get_safe_value(row, 'highest_high', 0)
                high_val = get_safe_value(row, 'high', 0)
                if highest_high > 0 and high_val > highest_high * 0.95:
                    reasons.append(f"ダブルトップ圏（high={fmt(high_val)}, hist_high={fmt(highest_high)}）")

        elif symbol == "sol_jpy":
            current_hour = (datetime.now() + timedelta(hours=9)).hour
            if position_type == 'long':
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val < 40:
                    reasons.append(f"RSI過売り({fmt(rsi_val)})")
                
                cci_val = get_safe_value(row, 'CCI', 0)
                if cci_val < -100:
                    reasons.append(f"CCI過売り({fmt(cci_val)})")
                
                if 10 <= current_hour <= 15:
                    reasons.append("日本活発取引時間帯（日中）")
                elif 19 <= current_hour <= 23:
                    reasons.append("日本活発取引時間帯（夜間）")
                
                if current_hour < 10 or current_hour > 23:
                    if rsi_val < 35:
                        reasons.append(f"非活発時間帯・強い過売り（RSI={fmt(rsi_val)}）")
                    volume_val = get_safe_value(row, 'volume', 0)
                    vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                    if volume_val > vol_avg_val * 1.2:
                        reasons.append(f"出来高増加（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")
            else:  # short
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val > 68:
                    reasons.append(f"RSI過買い({fmt(rsi_val)})")
                
                cci_val = get_safe_value(row, 'CCI', 0)
                if cci_val > 120:
                    reasons.append(f"CCI過買い({fmt(cci_val)})")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val * 1.1:
                    reasons.append(f"出来高増加（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")
                
                if 21 <= current_hour or current_hour < 2:
                    reasons.append("夜間セッション（21:00-2:00）")
                    if rsi_val > 68:
                        reasons.append(f"夜間の弱いショート条件（RSI={fmt(rsi_val)}）")
                else:
                    if rsi_val > 78:
                        reasons.append(f"日中の強いショート条件（RSI={fmt(rsi_val)}）")
                    close_val = get_safe_value(row, 'close', 0)
                    ema_short_val = get_safe_value(row, 'EMA_short', 0)
                    if close_val < ema_short_val:
                        reasons.append(f"短期EMA下抜け（close={fmt(close_val)}, EMA_short={fmt(ema_short_val)}）")

        elif symbol == "doge_jpy":
            if position_type == 'long':
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val < 40:
                    reasons.append(f"RSI過売り({fmt(rsi_val)})")
                
                cci_val = get_safe_value(row, 'CCI', 0)
                if cci_val < -100:
                    reasons.append(f"CCI過売り({fmt(cci_val)})")
                
                close_val = get_safe_value(row, 'close', 0)
                ma25_val = get_safe_value(row, 'MA25', 0)
                if close_val > ma25_val:
                    reasons.append(f"25MA上抜け（close={fmt(close_val)}, MA25={fmt(ma25_val)}）")
                
                if 50 <= rsi_val <= 80:
                    reasons.append(f"RSI適正モメンタム({fmt(rsi_val)})")
                
                if cci_val > 50:
                    reasons.append(f"CCIモメンタム上昇({fmt(cci_val)})")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val:
                    reasons.append(f"出来高平均以上（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")
            else:  # short
                if get_safe_value(row, 'downtrend', False):
                    reasons.append("下降トレンド確立")
                
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val > 50:
                    reasons.append(f"一時反発（RSI={fmt(rsi_val)}）")
                
                close_val = get_safe_value(row, 'close', 0)
                ema_long_val = get_safe_value(row, 'EMA_long', 0)
                if close_val < ema_long_val:
                    reasons.append(f"長期EMA下抜け（close={fmt(close_val)}, EMA_long={fmt(ema_long_val)}）")
                
                cci_val = get_safe_value(row, 'CCI', 0)
                if cci_val > 0:
                    reasons.append(f"CCI反発の兆候({fmt(cci_val)})")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val:
                    reasons.append(f"出来高平均以上（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")
                
                trend_strength = get_safe_value(row, 'trend_strength', 0)
                if trend_strength > 0.7:
                    reasons.append(f"強いトレンド({fmt(trend_strength, 2)})")

        elif symbol == "bcc_jpy":  # 新規追加
            if position_type == 'long':
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val < 35:
                    reasons.append(f"RSI過売り({fmt(rsi_val)})")
                
                cci_val = get_safe_value(row, 'CCI', 0)
                if cci_val < -100:
                    reasons.append(f"CCI過売り({fmt(cci_val)})")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val * 1.15:
                    reasons.append(f"出来高15%増加（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")
                
                close_val = get_safe_value(row, 'close', 0)
                ma25_val = get_safe_value(row, 'MA25', 0)
                if close_val > ma25_val:
                    reasons.append(f"25MA上抜け（close={fmt(close_val)}, MA25={fmt(ma25_val)}）")
            else:  # short
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val > 70:
                    reasons.append(f"RSI過買い({fmt(rsi_val)})")
                
                cci_val = get_safe_value(row, 'CCI', 0)
                if cci_val > 100:
                    reasons.append(f"CCI過買い({fmt(cci_val)})")
                
                close_val = get_safe_value(row, 'close', 0)
                ema_short_val = get_safe_value(row, 'EMA_short', 0)
                if close_val < ema_short_val:
                    reasons.append(f"短期EMA下抜け（close={fmt(close_val)}, EMA_short={fmt(ema_short_val)}）")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val * 1.1:
                    reasons.append(f"出来高増加（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")

        elif symbol == "ada_jpy":
            if position_type == 'long':
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val < 35:
                    reasons.append(f"RSI過売り({fmt(rsi_val)})")
                
                cci_val = get_safe_value(row, 'CCI', 0)
                if cci_val < -100:
                    reasons.append(f"CCI過売り({fmt(cci_val)})")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val * 1.15:
                    reasons.append(f"出来高15%増加（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")
                
                close_val = get_safe_value(row, 'close', 0)
                ma25_val = get_safe_value(row, 'MA25', 0)
                if close_val > ma25_val:
                    reasons.append(f"25MA上抜け（close={fmt(close_val)}, MA25={fmt(ma25_val)}）")
                
                ema_short_val = get_safe_value(row, 'EMA_short', 0)
                ema_long_val = get_safe_value(row, 'EMA_long', 0)
                if ema_short_val > 0 and ema_long_val > 0:
                    if ema_short_val > ema_long_val:
                        reasons.append(f"EMAゴールデンクロス（EMA_short={fmt(ema_short_val)}, EMA_long={fmt(ema_long_val)}）")
            else:  # short
                rsi_val = get_safe_value(row, 'RSI', 0)
                if rsi_val > 70:
                    reasons.append(f"RSI過買い({fmt(rsi_val)})")
                
                cci_val = get_safe_value(row, 'CCI', 0)
                if cci_val > 100:
                    reasons.append(f"CCI過買い({fmt(cci_val)})")
                
                close_val = get_safe_value(row, 'close', 0)
                ema_short_val = get_safe_value(row, 'EMA_short', 0)
                if close_val < ema_short_val:
                    reasons.append(f"短期EMA下抜け（close={fmt(close_val)}, EMA_short={fmt(ema_short_val)}）")
                
                volume_val = get_safe_value(row, 'volume', 0)
                vol_avg_val = get_safe_value(row, 'vol_avg', 0)
                if volume_val > vol_avg_val * 1.1:
                    reasons.append(f"出来高増加（vol={fmt(volume_val)}, avg={fmt(vol_avg_val)}）")

        # センチメント要因
        if hasattr(self, 'sentiment'):
            bullish_val = self.sentiment.get('bullish', 50)
            bearish_val = self.sentiment.get('bearish', 50)
            if position_type == 'long' and bullish_val > 60:
                reasons.append(f"強気センチメント優勢（bullish={bullish_val}）")
            elif position_type == 'short' and bearish_val > 60:
                reasons.append(f"弱気センチメント優勢（bearish={bearish_val}）")

            vol = self.sentiment.get('volatility', 50)
            if vol > 70:
                reasons.append(f"高ボラティリティ環境（vol={vol}）")
            elif vol < 30:
                reasons.append(f"低ボラティリティ環境（vol={vol}）")

        return " + ".join(reasons) if reasons else "シグナル検出"


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
        start_date = datetime.now()
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
            'last_report_time': datetime.now()
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
                'last_successful_cycle': datetime.now(),
                'error_log': []
            }

            # バックテストにあわせてlast_sentiment_timeを初期化
            self.last_sentiment_time = None

            # メインループ - より堅牢なエラーハンドリングを追加
            while True:
                loop_start_time = datetime.now()
                
                try:
                    # 現在の日本時間を取得
                    jst_now = datetime.now() + timedelta(hours=9)
                    
                    # システム健全性チェック
                    health_check_counter += 1
                    if health_check_counter >= 60:  # 5時間ごと(60 * 5分)
                        self._perform_health_check()
                        health_check_counter = 0
                    
                    # 定期的なポジション検証（1時間ごと）
                    self.logger.info("定期ポジション検証を実行します")
                    self.verify_positions()
                    
                    # バックアップと増資チェック（1日1回）
                    if jst_now.hour == 0 and jst_now.minute < 5:  # 深夜0時台の最初のサイクル
                        self.logger.info("日次バックアップと増資チェックを実行します")
                        self.check_backup_needed()
                        self.check_monthly_increase()
                    
                    # 日次レポート送信（日本時間の深夜0時）
                    if jst_now.hour == 0 and jst_now.minute < 5 and stats['last_report_time'].day != jst_now.day:
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
                            
                            # 処理中のDataFrameを一時的に保存（_get_entry_reasonで参照するため）- バックテストと同様
                            self.df_5min = df_5min
                            
                            # シグナル生成（センチメント考慮版）
                            df_5min = self.generate_signals_with_sentiment(symbol, df_5min, df_hourly)
                            
                            # 最新のシグナル情報を取得
                            latest_signals = df_5min.iloc[-2]
                            previous_signals = df_5min.iloc[-3]
                            
                            # 現在のポジション状態を確認
                            position = self.positions.get(symbol)
                            
                            # バックテストと同様の手順でポジションを判断
                            
                            # ポジションがない場合のエントリー判断
                            if position is None:

                                # if symbol == 'doge_jpy':
                                #     self.logger.info(f"ETH: 強制売りシグナル設定")
                                #     # DataFrameのコピーを作成してからアクセスする
                                #     latest_signals = latest_signals.copy()
                                #     latest_signals['sell_signal'] = True
                                # elif symbol == 'sol_jpy':
                                #     self.logger.info(f"ETH: 強制売りシグナル設定")
                                #     # DataFrameのコピーを作成してからアクセスする
                                #     latest_signals = latest_signals.copy()
                                #     latest_signals['sell_signal'] = True
                                # else:
                                #     buy_signal = latest_signals.get('buy_signal', False)
    
                                # 買いシグナル検出
                                if latest_signals.get('buy_signal', False) and previous_signals.get('buy_signal', False):
                                    self._handle_entry(symbol, 'long', latest_signals, stats, trade_logs)                                   
                                
                                # 売りシグナル検出
                                elif latest_signals.get('sell_signal', False) and previous_signals.get('sell_signal', False):
                                    self._handle_entry(symbol, 'short', latest_signals, stats, trade_logs)
                            
                            # ポジションがある場合のイグジット判断
                            elif position is not None:
                                self._check_exit_conditions(symbol, stats, trade_logs, df_5min)
                            
                            # DataFrameの参照を削除（バックテストと同様）
                            if hasattr(self, 'df_5min'):
                                del self.df_5min
                            
                        except Exception as e:
                            self.logger.error(f"{symbol}の処理中にエラーが発生: {str(e)}", exc_info=True)
                            error_states['error_log'].append(f"{datetime.now()}: {symbol}処理エラー - {str(e)}")
                            continue
                    
                    # メインループサイクル間のスリープ（5分間隔）
                    elapsed_time = (datetime.now() - loop_start_time).total_seconds()
                    sleep_time = max(70 - elapsed_time, 10)  # 少なくとも10秒は待機
                    self.logger.info(f"次のサイクルまで{sleep_time:.1f}秒待機します")
                    
                    # 成功したのでエラーカウンターをリセット
                    error_states['consecutive_api_errors'] = 0
                    error_states['consecutive_data_errors'] = 0
                    error_states['last_successful_cycle'] = datetime.now()
                    
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
                    error_states['error_log'].append(f"{datetime.now()}: {error_message}")
                    
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
                    error_states['error_log'].append(f"{datetime.now()}: {error_message}")
                    
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
                    time_since_last_success = (datetime.now() - error_states['last_successful_cycle']).total_seconds() / 3600
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
            # 稼働状況のレポートを出力
            self._generate_final_report(start_date, start_balance, stats, trade_logs)
            
            # 取引ログをファイルに保存
            self.save_trade_logs_to_excel(trade_logs)
            
            # 進行中のポジション情報を保存
            self.save_positions()
            
            self.logger.info("プログラムを終了します。")

    def _verify_api_connection(self):
        """APIへの接続を検証する"""
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
                        holding_time = datetime.now() - self.entry_times[symbol]
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
        """市場データを取得する（完全な24時間分の30分足データを確保）
        
        Parameters:
        symbol (str): 通貨ペア
        
        Returns:
        tuple: (30分足データ, 時間足データ) 取得失敗時はNoneを含む
        """
        try:
            # 現在の日付を取得
            current_date = datetime.now()
            date_str = current_date.strftime('%Y%m%d')
            
            # 30分足データの取得（複数日の結合を考慮）
            self.logger.info(f"{symbol}の15分足データ取得を開始します（完全な24時間分を確保）")
            
            # 最新の日付から必要日数分のデータを取得して結合する
            all_30min_data = []
            required_candles = 96  # 24時間分の15分足（96本）
            candles_collected = 0
            days_to_check = 3  # 最大3日分までさかのぼる
            
            for day_offset in range(days_to_check):
                check_date = current_date - timedelta(days=day_offset)
                check_date_str = check_date.strftime('%Y%m%d')
                
                self.logger.debug(f"{symbol}の{check_date_str}の15分足データを確認中")
                df_day = self.get_cached_data(symbol, '15min', check_date_str)
                
                if not df_day.empty:
                    # データの行数（ローソク足の数）を取得
                    day_candles = len(df_day)
                    self.logger.debug(f"{check_date_str}の15分足データ: {day_candles}本")
                    
                    # 最新日のデータ結合方法
                    if day_offset == 0:
                        # 現在の日付の場合は、全データを追加
                        all_30min_data.append(df_day)
                        candles_collected += day_candles
                    else:
                        # 前日以前の場合は、古い方から必要分だけ追加
                        remaining_needed = required_candles - candles_collected
                        if remaining_needed <= 0:
                            # すでに十分なデータがある場合は追加不要
                            break
                            
                        if day_candles > remaining_needed:
                            # 必要な分だけ取得（新しい順）
                            df_day = df_day.sort_index(ascending=False).head(remaining_needed).sort_index()
                            
                        all_30min_data.append(df_day)
                        candles_collected += min(day_candles, remaining_needed)
                
                # すでに十分なデータが集まった場合は終了
                if candles_collected >= required_candles:
                    self.logger.debug(f"十分なデータ({candles_collected}/{required_candles}本)が集まりました")
                    break
                    
            # データが十分に集まったかをチェック
            if candles_collected < required_candles / 2:  # 最低でも半分（24本=12時間分）は欲しい
                self.logger.warning(f"{symbol}の15分足データが不足しています: {candles_collected}/{required_candles}本")
                if candles_collected == 0:
                    return None, None
            
            # 結合したデータを時刻順にソート
            if all_30min_data:
                df_30min = pd.concat(all_30min_data).sort_values('timestamp')
                self.logger.info(f"{symbol}の15分足データ: 合計{len(df_30min)}本取得（目標: {required_candles}本）")
                
                # 最新の48本（24時間分）だけを使用
                if len(df_30min) > required_candles:
                    df_30min = df_30min.iloc[-required_candles:]
                    self.logger.debug(f"{symbol}の15分足データを最新{required_candles}本に制限しました")
            else:
                self.logger.warning(f"{symbol}の15分足データを取得できませんでした")
                df_30min = pd.DataFrame()  # 空のデータフレーム
            
            # 1時間足データの取得（3日分 - バックテストと同様）
            hourly_candles = []
            for h_offset in range(2, -1, -1):
                hourly_date = (current_date - timedelta(days=h_offset)).strftime('%Y%m%d')
                df_hourly_day = self.get_cached_data(symbol, '1hour', hourly_date)
                if not df_hourly_day.empty:
                    hourly_candles.append(df_hourly_day)
            
            # 時間足データの結合
            if hourly_candles:
                df_hourly = pd.concat(hourly_candles).sort_values('timestamp')
                self.logger.debug(f"{symbol}の時間足データ: {len(df_hourly)}本")
            else:
                self.logger.warning(f"{symbol}の時間足データを取得できませんでした")
                df_hourly = pd.DataFrame()  # 空のデータフレーム
            
            return df_30min, df_hourly
                
        except Exception as e:
            self.logger.error(f"{symbol}のデータ取得中にエラーが発生: {str(e)}", exc_info=True)
            return None, None

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
        # テクニカル指標値の取得
        entry_rsi = signal_data.get('RSI', None)
        entry_cci = signal_data.get('CCI', None)
        entry_atr = signal_data.get('ATR', None)
        entry_adx = signal_data.get('ADX', None)

        # スコア情報の取得（新規追加）
        buy_score = signal_data.get('buy_score', 0)
        sell_score = signal_data.get('sell_score', 0)

        # EMA乖離率の取得（新規追加）
        ema_deviation = signal_data.get('ema_deviation', 0)
        
        # エントリーの詳細をログに記録
        entry_reason = self._get_entry_reason(symbol, signal_data, position_type)

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
        
        order_result = self.execute_order_with_confirmation(
            symbol,
            'buy' if position_type == 'long' else 'sell',
            order_size
        )
        
        if order_result['success']:
            executed_size = order_result.get('executed_size', 0)
            # 実際に約定したサイズで金額を再計算
            final_order_amount = executed_size * current_price
            
            # ポジション情報を更新
            self.positions[symbol] = position_type
            self.entry_prices[symbol] = current_price
            self.entry_times[symbol] = datetime.now()
            self.entry_sizes[symbol] = executed_size

            # GMOコインの場合、position_idを取得して保存
            time.sleep(3)  # 注文が約定するまで待機
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))
            positions_response = self.gmo_api.get_margin_positions(gmo_symbol)
            
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
            self.log_entry(symbol, position_type, current_price, datetime.now(), entry_rsi, entry_cci, entry_atr, entry_adx, entry_reason, entry_sentiment)
            
            # 通知送信（修正箇所）
            if self.notification_settings['send_on_entry']:
                # RSI/CCIの値をフォーマット
                rsi_str = f"{entry_rsi:.1f}" if entry_rsi is not None else "N/A"
                cci_str = f"{entry_cci:.1f}" if entry_cci is not None else "N/A"
                atr_str = f"{entry_atr:.2f}" if entry_atr is not None else "N/A"
                adx_str = f"{entry_adx:.1f}" if entry_adx is not None else "N/A"
                
                notification_body = (
                    f"価格: {current_price:.2f}円\n"
                    f"数量: {executed_size:.6f}\n"
                    f"金額: {final_order_amount:.0f}円\n"
                    f"RSI: {rsi_str}\n"
                    f"CCI: {cci_str}\n"
                    f"ATR: {atr_str}\n"
                    f"ADX: {adx_str}\n"  # この行を追加
                    f"理由: {entry_reason}"
                )
                
                self.send_notification(
                    f"{symbol} {position_type}エントリー",
                    notification_body,
                    "entry"
                )

            # スコア値を保存
            self.entry_scores[symbol] = {
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
                'time': datetime.now(),
                'rsi': entry_rsi,
                'cci': entry_cci,
                'ema_deviation': ema_deviation,
                'reason': entry_reason,
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


    def _check_exit_conditions(self, symbol, stats, trade_logs, df_5min):
        """イグジット条件をチェックし、必要に応じて決済を実行する（backtest関数と整合性あり）
        
        Parameters:
        symbol (str): 通貨ペア
        current_price (float): 現在価格
        stats (dict): 統計情報の辞書
        trade_logs (list): 取引ログのリスト
        """
        position = self.positions[symbol]
        entry_price = self.entry_prices[symbol]
        entry_time = self.entry_times[symbol]
        entry_size = self.entry_sizes[symbol]
        
        # 保有時間の計算
        holding_time = datetime.now() - entry_time
        hours = holding_time.total_seconds() / 3600
        
        # 利確・損切りの設定を取得
        settings = self._get_dynamic_profit_loss_settings(symbol, df_5min)
        
        # イグジット条件の計算
        exit_condition = False
        exit_reason = ""

        exit_levels = self.calculate_dynamic_exit_levels(symbol, df_5min, position, entry_price)

        current_price = self.get_current_price(symbol)
        
        if position == 'long':
            # ロングポジションのイグジット判定
            long_profit_take = settings['long_profit_take']
            long_stop_loss = settings['long_stop_loss']
            
            # イグジット条件（バックテストと同じ条件）
            if current_price >= entry_price * long_profit_take:
                exit_condition = True
                # 修正: 実際の損益を計算してから理由を決定
                profit_temp = (current_price - entry_price) * entry_size
                exit_reason = "利益確定" if profit_temp > 0 else "損切り"
            elif current_price <= entry_price * long_stop_loss:
                exit_condition = True
                # 修正: 実際の損益を計算してから理由を決定
                profit_temp = (current_price - entry_price) * entry_size
                exit_reason = "損切り" if profit_temp <= 0 else "利益確定"
                           
        else:  # 'short'
            # ショートポジションのイグジット判定
            short_profit_take = settings['short_profit_take']
            short_stop_loss = settings['short_stop_loss']
            
            # イグジット条件（バックテストと同じ条件）
            if current_price <= entry_price * short_profit_take:
                exit_condition = True
                # 修正: 実際の損益を計算してから理由を決定
                profit_temp = (entry_price - current_price) * entry_size
                exit_reason = "利益確定" if profit_temp > 0 else "損切り"
            elif current_price >= entry_price * short_stop_loss:
                exit_condition = True
                # 修正: 実際の損益を計算してから理由を決定
                profit_temp = (entry_price - current_price) * entry_size
                exit_reason = "損切り" if profit_temp <= 0 else "利益確定"
        
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
            close_results = []
            success_count = 0
            failed_count = 0
            total_executed_size = 0
            total_order_ids = []

            # GMOコインの形式に変換
            gmo_symbol = self.symbol_mapping.get(symbol, symbol.upper().replace('_', ''))

            # ポジションの反対売買方向を決定
            side = "BUY" if position == 'short' else "SELL"

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
            
            # 決済結果の集計
            if success_count > 0:
                # 約定確認（少し待ってから確認）
                time.sleep(3)
                
                # 全約定サイズを取得
                for result in close_results:
                    if result.get('success'):
                        order_id = result.get('order_id')
                        executed_size = self.check_order_execution(order_id, symbol)
                        if executed_size > 0:
                            total_executed_size += float(executed_size)
                
                # 成功した決済がある場合
                self.logger.info(f"{symbol}のポジション決済結果: 成功={success_count}, 失敗={failed_count}, 総約定サイズ={total_executed_size}")
                
                # 損益計算
                if position == 'long':
                    profit = (current_price - entry_price) * entry_size
                    profit_pct = (current_price - entry_price) / entry_price * 100
                else:  # 'short'
                    profit = (entry_price - current_price) * entry_size
                    profit_pct = (entry_price - current_price) / entry_price * 100
                
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
                self.log_exit(symbol, position, current_price, entry_price, datetime.now(), profit, profit_pct, exit_reason, hours, entry_sentiment)
                
                # 通知送信
                if self.notification_settings['send_on_exit']:
                    self.send_notification(
                        f"{symbol} {position}決済",
                        f"価格: {current_price:.2f}円\n"
                        f"損益: {profit:.2f}円 ({profit_pct:+.2f}%)\n"
                        f"保有時間: {hours:.1f}時間\n"
                        f"理由: {exit_reason}\n"
                        f"決済結果: 成功={success_count}, 失敗={failed_count}",
                        "exit"
                    )
                
                # 保存されていたスコア値を取得
                saved_scores = self.entry_scores.get(symbol, {})

                # 取引ログを記録
                trade_log_exit = {
                    'symbol': symbol,
                    'type': position,
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'exit_price': current_price,
                    'exit_time': datetime.now(),
                    'size': entry_size,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason,
                    'holding_hours': hours,
                    # 保存されたスコア値を使用
                    'rsi_score_long': saved_scores.get('rsi_score_long', 0),
                    'rsi_score_short': saved_scores.get('rsi_score_short', 0),
                    'cci_score_long': saved_scores.get('cci_score_long', 0),
                    'cci_score_short': saved_scores.get('cci_score_short', 0),
                    'volume_score': saved_scores.get('volume_score', 0),
                    'bb_score_long': saved_scores.get('bb_score_long', 0),
                    'bb_score_short': saved_scores.get('bb_score_short', 0),
                    'ma_score_long': saved_scores.get('ma_score_long', 0),
                    'ma_score_short': saved_scores.get('ma_score_short', 0),
                    'adx_score_long': saved_scores.get('adx_score_long', 0),
                    'adx_score_short': saved_scores.get('adx_score_short', 0),
                    'mfi_score_long': saved_scores.get('mfi_score_long', 0),
                    'mfi_score_short': saved_scores.get('mfi_score_short', 0),
                    'atr_score_long': saved_scores.get('atr_score_long', 0),
                    'atr_score_short': saved_scores.get('atr_score_short', 0),
                    'macd_score_long': saved_scores.get('macd_score_long', 0),
                    'macd_score_short': saved_scores.get('macd_score_short', 0) 
                }
                trade_logs.append(trade_log_exit)
                
                
                # ポジション情報をリセット
                self.positions[symbol] = None
                self.entry_prices[symbol] = 0
                self.entry_times[symbol] = None
                self.entry_sizes[symbol] = 0
                self.position_ids[symbol] = None  # ポジションIDもクリア
                self.entry_scores[symbol] = {}
                
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
            total, used, free = psutil.disk_usage('/')
            free_gb = free / 1024 / 1024 / 1024
            self.logger.info(f"ディスク空き容量: {free_gb:.2f}GB")
            
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
                                self.logger.debug(f"古いキャッシュファイルを削除: {f}")
                            except Exception as e:
                                self.logger.error(f"キャッシュファイル削除エラー: {str(e)}")
            
            self.logger.info("システム健全性チェックが完了しました")
            return True
        except Exception as e:
            self.logger.error(f"システム健全性チェック中にエラーが発生: {str(e)}")
            return False

    def _send_daily_report(self, stats):
        """日次レポートを送信する
        
        Parameters:
        stats (dict): 統計情報の辞書
        """
        # 前回のレポート時刻
        last_report_time = stats['last_report_time']
        current_time = datetime.now()
        
        # 日付の書式
        date_str = last_report_time.strftime('%Y-%m-%d')
        
        # 勝率の計算
        daily_win_rate = 0
        if stats['daily_trades'] > 0:
            daily_win_rate = (stats['daily_wins'] / stats['daily_trades']) * 100
        
        # トータル勝率の計算
        total_win_rate = 0
        if stats['total_trades'] > 0:
            total_win_rate = (stats['total_wins'] / stats['total_trades']) * 100
        
        # プロフィットファクター
        profit_factor = 0
        if stats['total_loss'] > 0:
            profit_factor = stats['total_profit'] / stats['total_loss']
        
        # 日次収支
        daily_net_profit = stats['daily_profit'] - stats['daily_loss']
        
        # 資金状況
        current_balance = self.get_total_balance()
        
        # レポート本文
        report_body = (
            f"📊 {date_str} 日次取引レポート\n\n"
            f"🔄 取引回数: {stats['daily_trades']}回\n"
            f"✅ 勝ち: {stats['daily_wins']}回\n"
            f"❌ 負け: {stats['daily_losses']}回\n"
            f"📈 勝率: {daily_win_rate:.1f}%\n"
            f"💰 利益: {stats['daily_profit']:,.0f}円\n"
            f"💸 損失: {stats['daily_loss']:,.0f}円\n"
            f"📊 日次収支: {daily_net_profit:+,.0f}円\n\n"
            f"📋 累計成績\n"
            f"🔄 総取引回数: {stats['total_trades']}回\n"
            f"📈 総勝率: {total_win_rate:.1f}%\n"
            f"💹 プロフィットファクター: {profit_factor:.2f}\n"
            f"💵 現在資金: {current_balance:,.0f}円\n"
            f"💰 累計利益: {self.total_profit:,.0f}円\n\n"
            f"🏆 現在のポジション\n"
        )
        
        # ポジション情報を追加
        has_positions = False
        for symbol in self.symbols:
            position = self.positions.get(symbol)
            if position:
                has_positions = True
                entry_price = self.entry_prices[symbol]
                current_price = self.get_current_price(symbol)
                if current_price > 0:
                    if position == 'long':
                        profit_pct = (current_price / entry_price - 1) * 100
                    else:  # short
                        profit_pct = (entry_price / current_price - 1) * 100
                    
                    # 保有時間
                    if self.entry_times[symbol]:
                        holding_time = current_time - self.entry_times[symbol]
                        hours = holding_time.total_seconds() / 3600
                        report_body += f"{symbol}: {position} ({profit_pct:+.2f}%), {hours:.1f}時間保有\n"
                    else:
                        report_body += f"{symbol}: {position} ({profit_pct:+.2f}%)\n"
        
        if not has_positions:
            report_body += "現在ポジションはありません\n"
        
        # 市場センチメント情報を追加
        if hasattr(self, 'sentiment'):
            report_body += f"\n📉 市場センチメント\n"
            report_body += f"強気: {self.sentiment.get('bullish', 0):.1f}%\n"
            report_body += f"弱気: {self.sentiment.get('bearish', 0):.1f}%\n"
            report_body += f"中立: {self.sentiment.get('neutral', 0):.1f}%\n"
            report_body += f"ボラティリティ: {self.sentiment.get('volatility', 0):.1f}%\n"
            report_body += f"トレンド強度: {self.sentiment.get('trend_strength', 0):.1f}%\n"
        
        # レポートを送信
        self.logger.info("日次レポートを送信します")
        self.send_notification("日次取引レポート", report_body, "daily_report")
        self.logger.info("日次レポートを送信しました")


    def _generate_final_report(self, start_date, start_balance, stats, trade_logs):
        """最終レポートを生成する
        
        Parameters:
        start_date (datetime): 開始日時
        start_balance (float): 開始時の資金
        stats (dict): 統計情報の辞書
        trade_logs (list): 取引ログのリスト
        """
        end_balance = self.get_total_balance()
        running_time = datetime.now() - start_date
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
        self.logger.info(f"期間: {start_date.strftime('%Y-%m-%d %H:%M')} ~ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
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
            error_log_file = os.path.join(self.log_dir, f'error_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(error_log_file, 'w') as f:
                f.write("===== トレーディングボットエラーログ =====\n")
                f.write(f"生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
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
                'ltc_jpy': {'low': 60, 'high': 75, 'low_mult': 0.90, 'high_mult_long': 1.10, 'high_mult_short': 1.10},
                'ada_jpy': {'low': 0.50, 'high': 0.57, 'low_mult': 0.90, 'high_mult_long': 1.10, 'high_mult_short': 1.07},
                'xrp_jpy': {'low': 1.5, 'high': 2.1, 'low_mult': 1.00, 'high_mult_long': 1.00, 'high_mult_short': 0.90},
                'eth_jpy': {'low': 2000, 'high': 2500, 'low_mult': 0.95, 'high_mult_long': 1.00, 'high_mult_short': 1.10},
                'sol_jpy': {'low': 100, 'high': 140, 'low_mult': 0.95, 'high_mult_long': 1.10, 'high_mult_short': 1.03},
                'doge_jpy': {'low': 0.20, 'high': 0.24, 'low_mult': 0.95, 'high_mult_long': 0.95, 'high_mult_short': 1.00},
                'bcc_jpy': {'low': 310, 'high': 350, 'low_mult': 0.90, 'high_mult_long': 1.15, 'high_mult_short': 1.10}
            }
            default_setting = {'low': 0.5, 'high': 2.0, 'low_mult': 0.9, 'high_mult_long': 1.1, 'high_mult_short': 1.1}
            config = atr_thresholds.get(symbol, default_setting)

            # adx_thresholds = {
            #     'ltc_jpy':   {'low': 20, 'high': 28, 'low_mult': 0.85, 'high_mult': 1.20},
            #     'ada_jpy':   {'low': 22, 'high': 48, 'low_mult': 0.85, 'high_mult': 1.15},
            #     'xrp_jpy':   {'low': 20, 'high': 30, 'low_mult': 0.90, 'high_mult': 1.15},
            #     'eth_jpy':   {'low': 22, 'high': 30, 'low_mult': 0.85, 'high_mult': 1.15},
            #     'sol_jpy':   {'low': 17, 'high': 32, 'low_mult': 0.85, 'high_mult': 1.15},
            #     'doge_jpy':  {'low': 20, 'high': 35, 'low_mult': 0.90, 'high_mult': 0.95},
            #     'bcc_jpy':   {'low': 22, 'high': 32, 'low_mult': 0.90, 'high_mult': 1.10}
            # }
            adx_thresholds = {
                'ltc_jpy':   {'low': 20, 'high': 50, 'low_mult': 0.85, 'high_mult': 1.15},
                'ada_jpy':   {'low': 22, 'high': 48, 'low_mult': 0.85, 'high_mult': 1.15},
                'xrp_jpy':   {'low': 20, 'high': 50, 'low_mult': 0.80, 'high_mult': 1.20},
                'eth_jpy':   {'low': 20, 'high': 50, 'low_mult': 0.85, 'high_mult': 1.15},
                'sol_jpy':   {'low': 20, 'high': 50, 'low_mult': 0.85, 'high_mult': 1.15},
                'doge_jpy':  {'low': 20, 'high': 50, 'low_mult': 0.80, 'high_mult': 1.20},
                'bcc_jpy':   {'low': 22, 'high': 32, 'low_mult': 0.85, 'high_mult': 1.15}
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

            risk_reward_ratio = tp_pct / sl_pct

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


# メイン実行部分
if __name__ == "__main__":
    # コマンドラインからのモード選択
    parser = argparse.ArgumentParser(description='仮想通貨トレーディングボット（ショート対応・改良版）')
    parser.add_argument('mode', choices=['backtest', 'live'], help='実行モード (backtest または live)')
    parser.add_argument('--days', type=int, default=60, help='バックテスト日数 (デフォルト: 60)')
    parser.add_argument('--capital', type=int, default=100000, help='初期資金 (デフォルト: 100000円)')
    parser.add_argument('--test', action='store_true', help='テストモード（取引額を半分にする）')
    parser.add_argument('--notify', action='store_true', help='通知を有効にする')
    parser.add_argument('--debug', action='store_true', help='デバッグログを有効にする')
    parser.add_argument('--api-key', type=str, help='bitbank API キー')
    parser.add_argument('--api-secret', type=str, help='bitbank API シークレット')
    parser.add_argument('--reset', action='store_true', help='起動時にポジション情報をリセットする')
    parser.add_argument('--clear-cache', action='store_true', help='キャッシュをクリアしてから実行する')
    
    args = parser.parse_args()
    
    # ボットのインスタンス作成
    bot = CryptoTradingBot(initial_capital=args.capital, test_mode=args.test)
    
    # デバッグモード設定
    if args.debug:
        bot.logger.setLevel(logging.DEBUG)
        for handler in bot.logger.handlers:
            handler.setLevel(logging.DEBUG)
        bot.logger.info("デバッグモードが有効になりました")
    
    # 通知設定
    if args.notify:
        bot.notification_settings['enabled'] = True
        bot.logger.info("通知機能が有効になりました")
    
    # API認証情報を設定
    if args.api_key and args.api_secret:
        bot.exchange_settings_gmo['api_key'] = args.api_key
        bot.exchange_settings_gmo['api_secret'] = args.api_secret
        bot.logger.info("API認証情報を設定しました")
    
    # キャッシュクリア
    if args.clear_cache:
        bot.clear_cache()
        bot.logger.info("キャッシュをクリアしました")
    
    # ポジションリセット
    if args.reset:
        bot.reset_positions()
        bot.logger.info("ポジション情報をリセットしました")
    
    # 実行モード
    if args.mode == "backtest":
        print(f"{args.days}日間のバックテストを開始します...")
        bot.backtest(days_to_test=args.days)
    elif args.mode == "live":
        print("リアルタイムトレードモードを開始します...")
        bot.run_live()