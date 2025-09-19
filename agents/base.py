# agents/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Dict, Any, Union
import json
import time
import traceback
from openai import OpenAI
from prompts.manager import PROMPT_MANAGER
from memory.manager import MemoryManager
from utils.conversation import build_messages_openai, trim_messages_by_chars
from config import OPENAI_API_KEY


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class BaseAgentConfig:
    id: str
    prompt_name: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    history_limit_pairs: int = 10
    max_chars: int = 12000
    # optional: output_mode="text"|"json" 讓 base 可幫忙 JSON 解析（參數抽取類很實用）
    output_mode: str = "text"


class AgentBase(ABC):
    """供所有 Agent 共用的骨架：單輪 LLM 回合、串流/非串流、組 messages、logging。"""

    def __init__(self, config: BaseAgentConfig):
        self.config = config

    # ---------- 公開 API ----------

    def process_request(self, message: Union[str, Dict[str, Any]], memory: Optional[MemoryManager]) -> Union[str, Dict[str, Any]]:
        """
        Accepts either a plain string or a dict payload.
        Dict payload will be serialized to a compact JSON string before sending to the LLM.
        Returns parsed JSON if the model returned valid JSON; otherwise returns raw text.
        """
        try:
            if isinstance(message, dict):
                try:
                    # 序列化成緊湊 JSON，避免 prompt token 浪費
                    message = json.dumps(
                        message, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    # 萬一序列化失敗就退回安全字串
                    message = str(message)

            chunks = self._respond(message, memory, stream=False)
            text = "".join(chunks)
            return self._maybe_parse(text)
        except Exception as e:
            self._log_error("process_request", e)
            # 回傳一致的錯誤格式，方便上游判斷
            return {"error": "agent_error", "message": "抱歉，我現在有點問題，沒辦法回答您。"}

    def stream_request(self, message: Union[str, Dict[str, Any]], memory: Optional[MemoryManager]) -> Iterator[str]:

        try:
            if isinstance(message, dict):
                try:
                    message = json.dumps(
                        message, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    message = str(message)
            yield from self._respond(message, memory, stream=True)
        except Exception as e:
            self._log_error("stream_request", e)
            yield "抱歉，我現在有點問題，沒辦法回答您。"

    # ---------- 核心路徑 ----------
    def _respond(self, user_text: str, memory: Optional[MemoryManager], *, stream: bool) -> Iterator[str]:
        t0 = _now_ms()
        system_prompt = PROMPT_MANAGER.get(self.config.prompt_name)

        messages = build_messages_openai(
            system_prompt=system_prompt,
            agent_name=self.config.id,
            user_text=user_text,
            memory=memory,
            history_limit_pairs=self.config.history_limit_pairs,
            include_global_summary=True,
            include_facts=True,
        )
        messages = trim_messages_by_chars(messages, self.config.max_chars)

        # 前後 hook（可供子類覆寫）
        self.before_invoke(messages)

        if stream:
            for delta in self._invoke_model(messages, stream=True):
                yield delta
            self.after_invoke(None)
            self._log_ok("stream", _now_ms() - t0, -1)
        else:
            text = "".join(self._invoke_model(messages, stream=False))
            self.after_invoke(text)
            self._log_ok("non-stream", _now_ms() - t0, len(text))
            yield text

    # ---------- Provider 需實作 ----------
    @abstractmethod
    def _invoke_model(self, messages: List[Dict[str, str]], *, stream: bool) -> Iterator[str]:
        """呼叫實際模型，回傳字串片段（非串流也以單一 chunk 形式回傳）。"""
        ...

    # ---------- Hooks ----------
    def before_invoke(self, messages: List[Dict[str, str]]) -> None:
        pass

    def after_invoke(self, text: Optional[str]) -> None:
        pass

    # ---------- 輔助 ----------
    def _maybe_parse(self, text: str) -> str | Dict[str, Any]:
        if self.config.output_mode == "json":
            try:
                return json.loads(text)
            except Exception:
                # 依需求可在這裡強制回覆可解析錯誤格式
                return {"error": "model_output_not_json", "raw": text}
        return text

    def _log_ok(self, mode: str, elapsed_ms: int, out_len: int) -> None:
        print(f"[{self.config.id}] ok mode={mode} model={self.config.model} elapsed_ms={elapsed_ms} out_len={out_len}")

    def _log_error(self, where: str, err: Exception) -> None:
        print(f"[{self.config.id}] ERROR at {where}: {err}\n{traceback.format_exc()}")


class OpenAIChatAgentBase(AgentBase):
    """OpenAI 供應商專用的 base，實作 _invoke_model。"""

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        self.client = OpenAI()

    def _invoke_model(self, messages: List[Dict[str, str]], *, stream: bool) -> Iterator[str]:
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature
        }

        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if stream:
            kwargs["stream"] = True
            stream_obj = self.client.chat.completions.create(**kwargs)
            for chunk in stream_obj:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
        else:
            resp = self.client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            yield text
