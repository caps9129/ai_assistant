from __future__ import annotations
from typing import List, Dict, Optional, Tuple

try:
    from memory.manager import MemoryManager  # type: ignore
except Exception:
    MemoryManager = None  # 避免循環相依造成 import 失敗


def _coerce_history_limit(limit: Optional[int]) -> int:
    if limit is None:
        return 0
    try:
        return max(0, int(limit))
    except Exception:
        return 0


def build_messages_openai(
    *,
    system_prompt: str,
    agent_name: str,
    user_text: str,
    memory: Optional["MemoryManager"] = None,
    history_limit_pairs: Optional[int] = 10,
    include_global_summary: bool = True,
    include_facts: bool = True,
    attach_summary_as_system: bool = True,
) -> List[Dict[str, str]]:
    """
    產生 OpenAI Chat Completions API 可用的 messages 陣列。
    組合規則：
      [ system(system_prompt) ]
      [ system(global_summary_and_facts) ]  (有內容才附，加在 system 下方一次即可)
      [ ...history messages... ]            (由 MemoryManager 轉出，最多 N 組 pair，可選)
      [ user(user_text) ]

    備註：
    - 不直接存取 MemoryManager 內部資料結構；只呼叫 get_context_for_agent()
    - history_limit_pairs=0 代表不帶歷史（只吃單輪）
    """
    messages: List[Dict[str, str]] = []
    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 取摘要 + 歷史
    if memory is not None:
        summary_text, history_msgs = memory.get_context_for_agent(
            agent_name,
            include_global_summary=include_global_summary,
            include_facts=include_facts,
            max_pairs=_coerce_history_limit(history_limit_pairs) or None,
        )

        # 將摘要與事實以 system 再附一則（若你希望改成 assistant 也可以）
        if attach_summary_as_system and summary_text.strip():
            messages.append(
                {"role": "system", "content": summary_text.strip()})

        # 追加歷史對話（MemoryManager 已轉為 [{role, content}]）
        if history_msgs:
            messages.extend(history_msgs)

    # 最後加本輪使用者訊息
    messages.append({"role": "user", "content": user_text})
    return messages


def trim_messages_by_chars(messages: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    """
    以「字元數」粗估切尾，避免超長。之後若要換成 token-aware，可再替換實作。
    保留最前面的 system 區塊，優先刪最早的歷史訊息。
    """
    if max_chars <= 0:
        return messages

    def _len(msgs: List[Dict[str, str]]) -> int:
        return sum(len(m.get("content") or "") for m in msgs)

    if _len(messages) <= max_chars:
        return messages

    # 保留頭兩則 system（人設與摘要），從第三則開始砍舊歷史
    head = []
    tail = []
    for m in messages:
        if len(head) < 2 and m.get("role") == "system":
            head.append(m)
        else:
            tail.append(m)

    # 從 tail 開頭一路砍到長度夠
    i = 0
    while i < len(tail) and _len(head + tail[i:]) > max_chars:
        i += 1
    return head + tail[i:]
