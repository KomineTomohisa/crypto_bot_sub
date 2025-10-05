# notifiers/line_messaging.py
import os
import time
import logging
from typing import Sequence, Optional, Callable, Tuple

import requests

# APIエンドポイント
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_MULTICAST_URL = "https://api.line.me/v2/bot/message/multicast"

# 失敗時に呼ぶコールバック型（任意）
OnSendFail = Callable[[int, str, Optional[str]], None]
# 引数: (http_status, response_text, line_user_id_or_None)


class LineMessaging:
    """
    LINE Messaging API クライアント（Push/MultiCast）
    - 既定: 環境変数 LINE_CHANNEL_ACCESS_TOKEN を使用
    - 将来: DB保管トークンを使うなら provider_key と decryptor を渡す
      * decryptor は関数: (enc_bytes) -> str （例: security.crypto.decrypt_token）
      * token_fetcher は関数: (provider_key) -> Optional[bytes] （例: db.get_line_channel_token）
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        *,
        provider_key: Optional[str] = None,
        token_fetcher: Optional[Callable[[str], Optional[bytes]]] = None,
        decryptor: Optional[Callable[[bytes], str]] = None,
        timeout: int = 10,
        max_retries: int = 2,
        on_send_fail: Optional[OnSendFail] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        token = access_token

        # DB運用フック（任意）
        if token is None and provider_key and token_fetcher and decryptor:
            enc = token_fetcher(provider_key)
            if enc:
                token = decryptor(enc)

        # ENVフォールバック
        if token is None:
            token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

        if not token:
            raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN が見つかりません（ENV or DB）")

        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.on_send_fail = on_send_fail
        self.session = session or requests.Session()
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # ---- 内部: POST + リトライ ----
    def _post(self, url: str, payload: dict) -> Tuple[int, str]:
        last_text = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(url, headers=self.headers, json=payload, timeout=self.timeout)
                last_text = r.text
                # 成功 / 再試行不要の失敗を返す
                if r.status_code in (200, 201):
                    return r.status_code, r.text

                # 429/5xx はリトライ、それ以外は即返す
                if r.status_code not in (429, 500, 502, 503, 504):
                    return r.status_code, r.text

                self.logger.warning(
                    f"LINE API {r.status_code}: {r.text[:200]} retry {attempt}/{self.max_retries}"
                )
            except requests.RequestException as e:
                last_text = str(e)
                self.logger.warning(f"LINE API exception: {e} retry {attempt}/{self.max_retries}")

            time.sleep(1.0 * attempt)  # 素朴なバックオフ

        # リトライ尽きた
        return 599, last_text

    # ---- 公開: 1人宛て Push ----
    def send_text(self, line_user_id: str, text: str) -> bool:
        payload = {"to": line_user_id, "messages": [{"type": "text", "text": text}]}
        code, body = self._post(LINE_PUSH_URL, payload)
        if code != 200 and self.on_send_fail:
            self.on_send_fail(code, body, line_user_id)
        return code == 200

    # ---- 公開: 複数宛て MultiCast（最大500件/回・要プラン確認）----
    def send_text_bulk(self, line_user_ids: Sequence[str], text: str) -> bool:
        ids = list(line_user_ids or [])
        if not ids:
            return True
        payload = {"to": ids, "messages": [{"type": "text", "text": text}]}
        code, body = self._post(LINE_MULTICAST_URL, payload)
        if code != 200 and self.on_send_fail:
            self.on_send_fail(code, body, None)
        return code == 200
