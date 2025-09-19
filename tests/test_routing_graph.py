# tests/test_routing_graph.py
import pytest

from agents.graphs.routing_graph import run_routing

# 只比 SIMPLE/COMPLEX 與 cand_tools（= graph 輸出的 tools）
TEST_CASES = [
    # === SIMPLE_TOOL ===
    ("Find coffee shops near me.", "SIMPLE_TOOL",
     ["google_maps_search_places"]),
    ("找一下附近有沒有在營業的超市", "SIMPLE_TOOL", ["google_maps_search_places"]),
    ("What's the address of the OSU Kelley Engineering Center?", "SIMPLE_TOOL",
     ["google_maps_search_places"]),

    ("How long does it to walk from my current location to the library?",
     "SIMPLE_TOOL", ["google_maps_directions"]),
    ("開車到台北101要多久？", "SIMPLE_TOOL", ["google_maps_directions"]),
    ("Navigate to 2501 SW Jefferson Way, Corvallis, OR",
     "SIMPLE_TOOL", ["google_maps_directions"]),

    ("Schedule a meeting for tomorrow at 3pm titled 'Project Sync'",
     "SIMPLE_TOOL", ["google_calendar_create_event"]),
    ("幫我預約下週三早上十點的牙醫回診",
     "SIMPLE_TOOL", ["google_calendar_create_event"]),

    ("Remind me to buy milk", "SIMPLE_TOOL", ["google_tasks_create_task"]),
    ("新增一個待辦事項：完成報告", "SIMPLE_TOOL", ["google_tasks_create_task"]),
    ("Add a to-do to call John by this Friday at 5pm",
     "SIMPLE_TOOL", ["google_tasks_create_task"]),

    # === COMPLEX_TOOL ===
    ("Find the highest-rated pizza place nearby and take me there.",
     "COMPLEX_TOOL", ["google_maps_search_places", "google_maps_directions"]),
    ("導航到附近最近的加油站", "COMPLEX_TOOL",
     ["google_maps_search_places", "google_maps_directions"]),
    ("I need to find a pharmacy that's open now, what's the ETA?",
     "COMPLEX_TOOL", ["google_maps_search_places", "google_maps_directions"]),
    ("幫我找一家便宜的餐廳並告訴我怎麼走",
     "COMPLEX_TOOL", ["google_maps_search_places", "google_maps_directions"]),
    ("Search for parks within 5km and show me the route to the one with the best rating.",
     "COMPLEX_TOOL", ["google_maps_search_places", "google_maps_directions"]),
    ("What's the closest grocery store and how long will it take to drive there?",
     "COMPLEX_TOOL", ["google_maps_search_places", "google_maps_directions"]),

    # 下列 GENERAL_CHAT/EXIT 在本檔不比（只測 SIMPLE/COMPLEX）：
    ("Tell me a joke.", "GENERAL_CHAT", []),
    ("Who is the current president of the United States?", "GENERAL_CHAT", []),
    ("今天天氣如何？", "GENERAL_CHAT", []),
    ("Set a timer for 10 minutes.", "GENERAL_CHAT", []),
    ("Thanks, that's all for now.", "EXIT", []),
    ("好，掰掰", "EXIT", []),
]


@pytest.mark.parametrize("user_input, expected_route, expected_tools", TEST_CASES)
def test_routing_graph_main_only(user_input, expected_route, expected_tools):
    """
    以 main_only 模式測 Stage-1 子圖（只跑 MainRouter -> FuseDecision）。
    只比對 SIMPLE/COMPLEX 的類型與 cand_tools（= tools）。
    """
    assert expected_route in {"SIMPLE_TOOL", "COMPLEX_TOOL"}

    out = run_routing(user_text=user_input, mode="main_only")
    assert "final_route" in out and "tools" in out

    assert out["final_route"] == expected_route, \
        f"[main_only] route mismatch. expected={expected_route}, got={out['final_route']}, text={user_input}"

    assert sorted(out["tools"]) == sorted(expected_tools), \
        f"[main_only] tools mismatch. expected={expected_tools}, got={out['tools']}, text={user_input}"


@pytest.mark.parametrize("user_input, expected_route, expected_tools", TEST_CASES)
def test_routing_graph_fusion(user_input, expected_route, expected_tools):
    """
    以 fusion 模式測 Stage-1 子圖（MainRouter -> QueryRouter -> FuseDecision）。
    只比對 SIMPLE/COMPLEX 的類型與 cand_tools（= tools）。
    """
    assert expected_route in {"SIMPLE_TOOL", "COMPLEX_TOOL"}

    out = run_routing(user_text=user_input, mode="fusion")
    assert "final_route" in out and "tools" in out

    assert out["final_route"] == expected_route, \
        f"[fusion] route mismatch. expected={expected_route}, got={out['final_route']}, text={user_input}"

    assert sorted(out["tools"]) == sorted(expected_tools), \
        f"[fusion] tools mismatch. expected={expected_tools}, got={out['tools']}, text={user_input}"


# --- 允許直接執行 ---
if __name__ == "__main__":
    pytest.main([__file__])
