# memory/manager.py
from __future__ import annotations
from pydantic import PrivateAttr
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain.memory import CombinedMemory
from langchain.memory.buffer import ConversationBufferMemory

from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple
import uuid

# ---------- Time helpers (timezone-aware UTC) ----------
UTC = timezone.utc


def utcnow() -> datetime:
    """Timezone-aware current UTC datetime."""
    return datetime.now(UTC)


def utcnow_iso(trim_seconds: bool = True) -> str:
    """ISO 8601 UTC string with 'Z' suffix."""
    dt = utcnow()
    if trim_seconds:
        dt = dt.replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


# ---------- LangChain (0.2+) ----------

# Pydantic for BaseMemory private field


PairStatus = Literal["open", "deferred", "closed", "canceled"]


@dataclass
class TurnPair:
    """
    一個代理的單個子任務 / 回合（user, agent 的成對互動 + 狀態）
    可附帶 slots（欄位收集）與 events（update/cancel/tool_result 等）。
    """
    pair_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_msg: Optional[str] = None
    agent_at: Optional[datetime] = None
    user_msg: Optional[str] = None
    user_at: Optional[datetime] = None
    status: PairStatus = "open"
    slots: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, str]]:
        """
        轉為簡單 {role, content} 訊息序列（近似實際順序）。
        預設假設 agent 先說話（常見於澄清問題）。
        """
        msgs: List[Dict[str, str]] = []
        if self.agent_msg:
            msgs.append({"role": "assistant", "content": self.agent_msg})
        if self.user_msg:
            msgs.append({"role": "user", "content": self.user_msg})
        return msgs

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.agent_at:
            d["agent_at"] = self.agent_at.astimezone(
                UTC).isoformat().replace("+00:00", "Z")
        if self.user_at:
            d["user_at"] = self.user_at.astimezone(
                UTC).isoformat().replace("+00:00", "Z")
        return d


class StaticGlobalContextMemory(BaseMemory):
    """
    BaseMemory 子類：提供固定字串的全域摘要/事實（唯讀）。
    能被 CombinedMemory 驗證接受。
    """
    memory_key: str = "global_context"
    _text: str = PrivateAttr(default="")

    # ---- 必要：回報這個 Memory 會注入哪些變數名 ----
    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def __init__(self, text: str, **data):
        super().__init__(**data)
        self._text = text

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return {self.memory_key: self._text}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        # 唯讀，不在此更新
        return

    def clear(self) -> None:
        self._text = ""


class MemoryManager:
    """
    Mode-B（LangChain 版）記憶管理：
      - 每代理維護 (user, agent) pair 與狀態：open / deferred / closed / canceled
      - 支援 active_agent 與（必要時的）pending_user
      - slots / events：結構化欄位與事件紀錄
      - apply_updates()：回覆配對器將一句話切段後，批次綁回不同 pair
      - as_langchain_combined_memory()：輸出 global_context + chat_history 供代理使用
    """

    def __init__(
        self,
        max_pairs_per_agent: int = 8,
        max_global_log: int = 800,
        max_summary_chars: int = 2000,
        timezone_name: str = "America/Los_Angeles",
    ):
        self.max_pairs_per_agent = max_pairs_per_agent
        self.max_global_log = max_global_log
        self.max_summary_chars = max_summary_chars
        self.timezone = timezone_name

        # 每代理：deque[TurnPair]
        self._pairs: Dict[str, Deque[TurnPair]] = {}

        # pair_id -> (agent_name, TurnPair)
        self._idx: Dict[str, Tuple[str, TurnPair]] = {}

        # 全域事件流（user/assistant 記錄）
        self._global_log: Deque[Dict[str, Any]] = deque(
            maxlen=self.max_global_log)

        # 全域摘要與事實
        self._global_summary: str = ""
        self._global_facts: Dict[str, str] = {}

        # 代理切換狀態
        self._active_agent: Optional[str] = None
        self._pending_user: Optional[str] = None

    # ----------------------------
    # 基本紀錄（全域 log）
    # ----------------------------
    def record_user(self, text: str) -> None:
        self._global_log.append(
            {"ts": utcnow_iso(), "role": "user", "content": text})

    def record_agent_log(self, agent_name: str, text: str) -> None:
        """僅寫全域日誌；不影響 pair 狀態"""
        self._global_log.append(
            {"ts": utcnow_iso(), "role": "assistant",
             "agent": agent_name, "content": text}
        )

    # ----------------------------
    # Pair 相關操作
    # ----------------------------
    def start_pair(
        self,
        agent_name: str,
        agent_msg: str,
        *,
        expects_user: bool = True,
        slots: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        pair_id: Optional[str] = None,
    ) -> str:
        """代理提出澄清/子任務：建立一個新 pair"""
        p = TurnPair(
            pair_id=pair_id or str(uuid.uuid4()),
            agent_msg=agent_msg,
            agent_at=utcnow(),
            status="open" if expects_user else "closed",
            slots=dict(slots or {}),
            meta=dict(meta or {}),
        )
        buf = self._pairs.setdefault(
            agent_name, deque(maxlen=self.max_pairs_per_agent))
        buf.append(p)
        self._idx[p.pair_id] = (agent_name, p)
        if expects_user:
            self._active_agent = agent_name
        self.record_agent_log(agent_name, agent_msg)
        return p.pair_id

    def attach_user_to_open_pair(
        self,
        agent_name: str,
        user_msg: str,
        *,
        pair_id: Optional[str] = None,
        close: bool = True,
        slot_updates: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        將使用者回覆綁回該代理的 open/最近 deferred pair。
        - pair_id 指定：直接綁定
        - 未指定：找該代理最近一個 open 或 deferred 的 pair
        """
        pair: Optional[TurnPair] = None
        if pair_id:
            _, pair = self._idx.get(pair_id, (None, None))
        else:
            for p in reversed(self._pairs.get(agent_name, deque())):
                if p.status in ("open", "deferred"):
                    pair = p
                    break
        if pair is None:
            return None

        pair.user_msg = user_msg
        pair.user_at = utcnow()
        if slot_updates:
            pair.slots.update(slot_updates)
        pair.status = "closed" if close else "open"

        # active_agent 規則：若回合已關閉，清空；否則仍然等待
        self._active_agent = None if close else agent_name

        self._global_log.append(
            {"ts": utcnow_iso(), "role": "user", "content": user_msg,
             "bind_to": pair.pair_id}
        )
        return pair.pair_id

    def agent_followup(
        self,
        agent_name: str,
        agent_msg: str,
        *,
        same_pair: bool = False,
        target_pair_id: Optional[str] = None,
        expects_user: bool = True,
        slot_updates: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        代理後續訊息：
          - same_pair=True：更新既有 pair（如補充/工具結果）
          - same_pair=False：開新 pair（新的澄清/子任務）
        """
        pair: Optional[TurnPair] = None
        if same_pair:
            if target_pair_id:
                _, pair = self._idx.get(target_pair_id, (None, None))
            else:
                for p in reversed(self._pairs.get(agent_name, deque())):
                    pair = p
                    break
            if pair is None:
                same_pair = False

        if same_pair and pair:
            pair.agent_msg = (pair.agent_msg or "") + \
                ("\n" if pair.agent_msg else "") + agent_msg
            pair.agent_at = utcnow()
            if slot_updates:
                pair.slots.update(slot_updates)
            pair.status = "open" if expects_user else "closed"
            self._active_agent = agent_name if expects_user else None
            self.record_agent_log(agent_name, agent_msg)
            return pair.pair_id

        # 開新 pair
        return self.start_pair(
            agent_name,
            agent_msg,
            expects_user=expects_user,
            slots=slot_updates,
        )

    def update_pair(
        self,
        pair_id: str,
        *,
        slot_updates: Optional[Dict[str, Any]] = None,
        event: Optional[Dict[str, Any]] = None,
        status: Optional[PairStatus] = None,
    ) -> bool:
        ref = self._idx.get(pair_id)
        if not ref:
            return False
        agent, pair = ref
        if slot_updates:
            pair.slots.update(slot_updates)
        if event:
            e = dict(event)
            e["ts"] = utcnow_iso()
            pair.events.append(e)
        if status:
            pair.status = status
        pair.agent_at = utcnow()
        return True

    def cancel_pair(self, pair_id: str, reason: str = "user") -> bool:
        return self.update_pair(pair_id, event={"type": "cancel", "by": reason}, status="canceled")

    def close_pair(self, pair_id: str) -> bool:
        return self.update_pair(pair_id, status="closed")

    # ----------------------------
    # 代理切換（active / deferred）
    # ----------------------------
    def set_active_agent(self, agent_name: Optional[str]) -> None:
        """設定目前等待使用者回覆的代理；None 代表沒有代理在等。"""
        self._active_agent = agent_name

    def get_active_agent(self) -> Optional[str]:
        """回傳目前等待使用者回覆的代理名稱；若無則為 None。"""
        return self._active_agent

    def defer_agent(self, agent_name: str) -> None:
        """把該代理最近的 open pair 標為 deferred（切去別的代理時用）"""
        for p in reversed(self._pairs.get(agent_name, deque())):
            if p.status == "open":
                p.status = "deferred"
                break
        if self._active_agent == agent_name:
            self._active_agent = None

    def resume_agent(self, agent_name: str) -> None:
        """把該代理最近的 deferred pair 恢復為 open，並設為 active"""
        for p in reversed(self._pairs.get(agent_name, deque())):
            if p.status == "deferred":
                p.status = "open"
                self._active_agent = agent_name
                break

    # ----------------------------
    # 回覆配對器入口（把一句話切成多個 updates 綁回不同 pair）
    # updates 形如：
    #   [{"pair_id":"A_table","slots":{"size":4}},
    #    {"pair_id":"B_table","slots":{"size":6,"diet":"vegetarian"}}]
    # ----------------------------
    def apply_updates(self, updates: List[Dict[str, Any]]) -> None:
        for upd in updates:
            pid = upd.get("pair_id")
            if not pid:
                continue
            slots = upd.get("slots") or {}
            event = upd.get("event")
            status = upd.get("status")
            self.update_pair(pid, slot_updates=slots,
                             event=event, status=status)

    # ----------------------------
    # 取得上下文 / 匯出給 LLM
    # ----------------------------
    def get_context_for_agent(
        self,
        agent_name: str,
        *,
        include_global_summary: bool = True,
        include_facts: bool = True,
        max_pairs: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """回傳 (summary_text, history_messages)"""
        parts: List[str] = []
        if include_global_summary and self._global_summary:
            parts.append(f"[GLOBAL SUMMARY]\n{self._global_summary}")
        if include_facts and self._global_facts:
            facts = "\n".join(f"- {k}: {v}" for k,
                              v in self._global_facts.items())
            parts.append(f"[GLOBAL FACTS]\n{facts}")
        summary_text = "\n\n".join(parts)

        buf = list(self._pairs.get(agent_name, deque()))
        if max_pairs is not None:
            buf = buf[-max_pairs:]

        messages: List[Dict[str, str]] = []
        for p in buf:
            messages.extend(p.to_messages())
        return summary_text, messages

    def as_langchain_combined_memory(self, agent_name: str):
        # 1) global_context
        parts = []
        if self._global_summary:
            parts.append(f"[GLOBAL SUMMARY]\n{self._global_summary}")
        if self._global_facts:
            facts = "\n".join(f"- {k}: {v}" for k,
                              v in self._global_facts.items())
            parts.append(f"[GLOBAL FACTS]\n{facts}")
        global_text = "\n".join(parts) if parts else ""
        global_mem = StaticGlobalContextMemory(
            text=global_text, memory_key="global_context")

        # 2) chat_history（加入 input_key/output_key）
        chat_mem = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
        )
        for p in self._pairs.get(agent_name, deque()):
            if p.agent_msg:
                chat_mem.chat_memory.add_ai_message(p.agent_msg)
            if p.user_msg:
                chat_mem.chat_memory.add_user_message(p.user_msg)

        # 3) CombinedMemory
        return CombinedMemory(memories=[global_mem, chat_mem])

    # ----------------------------
    # 全域摘要 / 事實（用 LangChain LLM）
    # ----------------------------
    def set_global_summary(self, text: str) -> None:
        self._global_summary = (text or "")[: self.max_summary_chars]

    def get_global_summary(self) -> str:
        return self._global_summary

    def upsert_fact(self, key: str, value: str) -> None:
        self._global_facts[key] = value

    def get_all_facts(self) -> Dict[str, str]:
        return dict(self._global_facts)

    def update_global_summary(
        self,
        llm: BaseChatModel,
        *,
        extra_instructions: str = "",
        take_last_n: int = 200,
    ) -> str:
        """
        使用 LangChain ChatModel 將全域日誌濃縮成摘要。
        你可以把 ChatOpenAI / GPT-4o-mini / 其他相容模型傳進來。
        """
        logs = list(self._global_log)[-take_last_n:]
        lines = []
        for rec in logs:
            who = rec.get(
                "agent", "ASSISTANT") if rec["role"] == "assistant" else "USER"
            lines.append(f"{who}: {rec['content']}")
        latest_log = "\n".join(lines)

        sys = "You summarize multi-agent conversations into concise, factual notes for future turns."
        if extra_instructions:
            sys += " " + extra_instructions

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys +
                 f"\nTimezone: {self.timezone}\nMax {self.max_summary_chars} chars."),
                MessagesPlaceholder("existing"),
                ("human",
                 "Latest log:\n{log}\n\nUpdate the running summary. Keep key facts, decisions, TODOs."),
            ]
        )

        existing_msgs = []
        if self._global_summary:
            existing_msgs = [SystemMessage(
                content=f"[PRIOR SUMMARY]\n{self._global_summary}")]

        chain = prompt | llm
        out = chain.invoke({"existing": existing_msgs, "log": latest_log})
        text = out.content.strip() if hasattr(out, "content") else str(out).strip()
        self._global_summary = text[: self.max_summary_chars]
        return self._global_summary

    # ----------------------------
    # 診斷 / 公用
    # ----------------------------
    def list_pairs(self, agent_name: str) -> List[Dict[str, Any]]:
        return [p.as_dict() for p in self._pairs.get(agent_name, deque())]

    def get_pair(self, pair_id: str) -> Optional[TurnPair]:
        ref = self._idx.get(pair_id)
        return ref[1] if ref else None

    def clear_agent(self, agent_name: str) -> None:
        for p in self._pairs.get(agent_name, deque()):
            self._idx.pop(p.pair_id, None)
        self._pairs.pop(agent_name, None)
        if self._active_agent == agent_name:
            self._active_agent = None

    def clear_all(self) -> None:
        self._pairs.clear()
        self._idx.clear()
        self._global_log.clear()
        self._global_summary = ""
        self._global_facts.clear()
        self._active_agent = None
        self._pending_user = None
