import tiktoken
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


from config import MEMORY_LOG_DIR


class MemoryManager:
    """
    管理特定對話 session 的短期和長期記憶。
    只儲存 user 和 assistant 的對話歷史。
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_tokens: int = 2000,
        max_turns: int = 10
    ):
        """
        初始化一個對話 session。
        如果提供了 session_id，則嘗試載入現有歷史；否則，建立一個新的。
        """
        MEMORY_LOG_DIR.mkdir(exist_ok=True)  # 確保資料夾存在

        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"⚠️ 警告：無法載入 tiktoken。基於 Token 的截斷功能將被禁用。錯誤：{e}")
            self.tokenizer = None

        if session_id and self._load_session_from_disk(session_id):
            print(f"✅ 成功恢復記憶 session，ID 為：{self.session_id}")
        else:
            self.session_id = str(uuid.uuid4())
            self.start_time = datetime.now()
            print(f"✅ 已建立新的記憶 session，ID 為：{self.session_id}")

    def _get_session_path(self, session_id: str) -> Path:
        """取得特定 session 的檔案路徑。"""
        return MEMORY_LOG_DIR / f"{session_id}.json"

    def _load_session_from_disk(self, session_id: str) -> bool:
        """從磁碟載入一個現有的 session 歷史。"""
        session_path = self._get_session_path(session_id)
        if not session_path.exists():
            return False

        try:
            with open(session_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)
                self.session_id = session_data["session_id"]
                self.start_time = datetime.fromisoformat(
                    session_data["start_time"])
                self.history = session_data["messages"]
                self._truncate_history()
            return True
        except (Exception, KeyError, json.JSONDecodeError) as e:
            print(f"❌ 載入 session {session_id} 時出錯：{e}")
            return False

    def add_message(self, role: str, content: str):
        """將一條新訊息加入歷史紀錄，並自動儲存。"""
        if role not in ["user", "assistant"]:
            print(f"⚠️ 警告：嘗試加入無效的角色 '{role}' 到記憶體中。已忽略。")
            return
        self.history.append({"role": role, "content": content})
        self._truncate_history()
        self.save_session_to_disk()

    def save_session_to_disk(self):
        """將當前的對話 session 儲存到其對應的檔案中。"""
        session_path = self._get_session_path(self.session_id)

        session_data = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "last_updated": datetime.now().isoformat(),
            "messages": self.history
        }

        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            print(f"[Memory] Session {self.session_id} 已成功更新。")
        except Exception as e:
            print(f"❌ [Memory] 儲存 session 時出錯：{e}")

    def get_history(self) -> List[Dict[str, str]]:
        """
        回傳純粹的 user 和 assistant 對話歷史。
        """
        return self.history

    def _truncate_history(self):
        """
        檢查歷史紀錄是否超出限制，如果超出，則從最舊的對話開始移除。
        """
        # (截斷邏輯維持不變)
        if len(self.history) > self.max_turns * 2:
            messages_to_remove = len(self.history) - self.max_turns * 2
            self.history = self.history[messages_to_remove:]

        if self.tokenizer:
            current_tokens = self.get_token_count()
            while current_tokens > self.max_tokens:
                if len(self.history) > 2:
                    self.history = self.history[2:]
                    current_tokens = self.get_token_count()
                else:
                    break

    def get_token_count(self) -> int:
        """計算目前歷史紀錄中的總 token 數量。"""
        if not self.tokenizer:
            return 0

        num_tokens = 0
        # 只計算 user 和 assistant 的訊息
        for message in self.history:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(self.tokenizer.encode(value))
        return num_tokens

    def clear(self):
        """清空當前對話的記憶。"""
        self.history = []
        print("記憶體已清除。")
