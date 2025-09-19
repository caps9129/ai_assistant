# tests/test_memory_mode_b.py
import pytest
from typing import List, Dict, Any

from memory.manager import MemoryManager


# -------------------------
# 極簡列印：只印 role + content
# -------------------------
def print_simple_history(m: MemoryManager, agent: str, title: str = "") -> List[Dict[str, str]]:
    _sep = "=" * 72
    summary, msgs = m.get_context_for_agent(agent)
    print("\n" + _sep)
    print(f"[{agent} HISTORY] {title}")
    print("-" * 72)
    for i, msg in enumerate(msgs, 1):
        print(f"{i:02d}. {msg['role']}: {msg['content']}")
    print(_sep)
    return msgs


# -------------------------
# 建立一個管理器（小上限即可）
# -------------------------
def make_manager() -> MemoryManager:
    return MemoryManager(max_pairs_per_agent=5, max_global_log=200)


# -------------------------
# 測試 1：模擬 A → B → C → A → C 的往返
# 只檢查各自 agent 的 history 是否正確獨立與順序無誤
# -------------------------
def test_cross_agent_simple_history_print():
    m = make_manager()

    # 使用者先跟 A
    m.start_pair("AgentA", "A: 需要地點嗎？", expects_user=True,
                 slots={"need": "location"})
    m.attach_user_to_open_pair("AgentA", "信義區", close=True)

    # 切去 B
    m.start_pair("AgentB", "B: 需要人數嗎？", expects_user=True,
                 slots={"need": "count"})
    m.attach_user_to_open_pair("AgentB", "2 人", close=True)

    # 切去 C
    m.start_pair("AgentC", "C: 要翻譯成哪種語言？",
                 expects_user=True, slots={"need": "lang"})
    m.attach_user_to_open_pair("AgentC", "英文", close=True)

    # 回到 A（新一回合）
    m.agent_followup("AgentA", "A: 還需要時間嗎？",
                     same_pair=False, expects_user=True)
    m.attach_user_to_open_pair("AgentA", "明天 15:00", close=True)

    # 再回到 C（新一回合）
    m.agent_followup("AgentC", "C: 請提供要翻譯的句子。",
                     same_pair=False, expects_user=True)
    m.attach_user_to_open_pair("AgentC", "「歡迎光臨」", close=True)

    # ---- 印出各 agent 的簡單歷史（僅 role + content）----
    msgs_a = print_simple_history(m, "AgentA", "A 往返兩回合")
    msgs_b = print_simple_history(m, "AgentB", "B 一回合")
    msgs_c = print_simple_history(m, "AgentC", "C 兩回合")

    # ---- 斷言：每個 agent 的訊息數量與順序 ----
    # AgentA：兩個 pair → 4 則訊息（A問/U答 × 2）
    assert len(msgs_a) == 4
    assert msgs_a[0]["role"] == "assistant" and "需要地點" in msgs_a[0]["content"]
    assert msgs_a[1]["role"] == "user" and "信義區" in msgs_a[1]["content"]
    assert msgs_a[2]["role"] == "assistant" and "需要時間" in msgs_a[2]["content"]
    assert msgs_a[3]["role"] == "user" and "15:00" in msgs_a[3]["content"]

    # AgentB：一個 pair → 2 則訊息
    assert len(msgs_b) == 2
    assert msgs_b[0]["role"] == "assistant" and "需要人數" in msgs_b[0]["content"]
    assert msgs_b[1]["role"] == "user" and "2" in msgs_b[1]["content"]

    # AgentC：兩個 pair → 4 則訊息
    assert len(msgs_c) == 4
    assert msgs_c[0]["role"] == "assistant" and "哪種語言" in msgs_c[0]["content"]
    assert msgs_c[1]["role"] == "user" and "英文" in msgs_c[1]["content"]
    assert msgs_c[2]["role"] == "assistant" and "提供要翻譯的句子" in msgs_c[2]["content"]
    assert msgs_c[3]["role"] == "user" and "歡迎光臨" in msgs_c[3]["content"]


# -------------------------
# 測試 2：同一句話同時回答兩個子任務（apply_updates）
# 仍然只列印 role + content，便於人工檢查
# -------------------------
def test_apply_updates_single_utterance_two_pairs_simple_print():
    m = make_manager()

    # 建立同一代理下的兩個未完成子任務（A 桌、B 桌）
    pid_a = m.start_pair(
        "ReservationsAgent", "A 桌：請給人數/偏好？",
        expects_user=True, slots={"party": "A", "date": "Fri", "time": "19:00"}
    )
    pid_b = m.start_pair(
        "ReservationsAgent", "B 桌：請給人數/偏好？",
        expects_user=True, slots={"party": "B", "date": "Sat", "time": "18:00"}
    )

    # 使用者「同一句話」回覆兩件事（實務上會由配對器拆段）
    user_utterance = "A 桌 4 人；B 桌 6 人、素食可。"
    # 這裡為了讓各 pair 的 history 可見，將「同一句」拆成兩段並綁回各自 pair
    m.attach_user_to_open_pair(
        "ReservationsAgent", "A 桌 4 人", pair_id=pid_a, close=True)
    m.attach_user_to_open_pair(
        "ReservationsAgent", "B 桌 6 人、素食可。", pair_id=pid_b, close=True)

    # 同步填入 slots（這一步代表配對器產生的結構化 updates）
    m.apply_updates([
        {"pair_id": pid_a, "slots": {"size": 4}},
        {"pair_id": pid_b, "slots": {"size": 6, "diet": "vegetarian"}},
    ])

    # 印出 ReservationsAgent 的簡單歷史
    msgs = print_simple_history(m, "ReservationsAgent", "同一句話回答兩個子任務")

    # 斷言順序與內容（兩個 pair → 4 則訊息）
    assert len(msgs) == 4
    assert msgs[0]["role"] == "assistant" and "A 桌" in msgs[0]["content"]
    assert msgs[1]["role"] == "user" and "A 桌 4 人" in msgs[1]["content"]
    assert msgs[2]["role"] == "assistant" and "B 桌" in msgs[2]["content"]
    assert msgs[3]["role"] == "user" and "B 桌 6 人" in msgs[3]["content"]

    # 斷言 slots 已更新
    p_a = m.get_pair(pid_a)
    p_b = m.get_pair(pid_b)
    assert p_a.slots.get("size") == 4
    assert p_b.slots.get("size") == 6 and p_b.slots.get("diet") == "vegetarian"


if __name__ == "__main__":
    # 直接執行時也會印出簡單 history
    test_cross_agent_simple_history_print()
    test_apply_updates_single_utterance_two_pairs_simple_print()
