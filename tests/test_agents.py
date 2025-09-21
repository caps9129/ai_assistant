# test_agents.py
import pytest

# 直接測合併後的 graph（routing → planning）
from agents.graphs.pipeline_graph import run_agent

TEST_CASES = [
    # === 1. SIMPLE_TOOL (11) ===
    ("Find coffee shops near me.", "SIMPLE_TOOL",
     ["google_maps_search_places"]),
    ("找一下附近有沒有在營業的超市", "SIMPLE_TOOL", ["google_maps_search_places"]),
    ("What's the address of the OSU Kelley Engineering Center?",
     "SIMPLE_TOOL", ["google_maps_search_places"]),

    ("How long does it to walk from my current location to the library?",
     "SIMPLE_TOOL", ["google_maps_directions"]),
    ("開車到台北101要多久？", "SIMPLE_TOOL", ["google_maps_directions"]),
    ("Navigate to 2501 SW Jefferson Way, Corvallis, OR",
     "SIMPLE_TOOL", ["google_maps_directions"]),

    ("Schedule a meeting for tomorrow at 3pm titled 'Project Sync'",
     "SIMPLE_TOOL", ["google_calendar_create_event"]),
    ("幫我預約下週三早上十點的牙醫回診", "SIMPLE_TOOL", ["google_calendar_create_event"]),

    ("Remind me to buy milk", "SIMPLE_TOOL", ["google_tasks_create_task"]),
    ("新增一個待辦事項：完成報告", "SIMPLE_TOOL", ["google_tasks_create_task"]),
    ("Add a to-do to call John by this Friday at 5pm",
     "SIMPLE_TOOL", ["google_tasks_create_task"]),

    # === 2. COMPLEX_TOOL (6) ===
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

    # === 3. GENERAL_CHAT (4) ===
    ("Tell me a joke.", "GENERAL_CHAT", []),
    ("Who is the current president of the United States?", "GENERAL_CHAT", []),
    ("今天天氣如何？", "GENERAL_CHAT", []),  # No weather tool is available.
    # No timer tool available.
    ("Set a timer for 10 minutes.", "GENERAL_CHAT", []),

    # === 4. EXIT (2) ===
    ("Thanks, that's all for now.", "EXIT", []),
    ("好，掰掰", "EXIT", []),
]


@pytest.mark.parametrize("user_input, expected_route, expected_tools", TEST_CASES)
def test_pipeline_graph_routing_and_planning(user_input, expected_route, expected_tools):
    """
    測試合併後的 graph（routing→planning）
    - 以 mode='main_only' 執行：僅跑 MainRouter + FuseDecision（與你原本測法一致）
    - 檢查 routing 輸出：final_route 與 tools
    - 若為 SIMPLE/COMPLEX，檢查 planning 輸出：plan 應為 dict；否則可為 None/缺省
    """
    # WHEN: 執行合併後的 pipeline
    res = run_agent(
        user_input,
        memory=None,
        mode="main_only",  # 與原本 ManagerConfig(run_mode="main_only") 對齊
    )

    # THEN: 路由決策存在且正確
    assert "final_route" in res, "The 'final_route' key should exist in the result"
    assert res["final_route"] == expected_route, f"Expected route '{expected_route}' for input: '{user_input}'"

    # 工具集合（與預期比對，不考慮順序）
    actual_tools = sorted(res.get("tools", []))
    assert actual_tools == sorted(expected_tools), \
        f"Mismatch in user-facing tools expected={expected_tools} actual={actual_tools}"

    # 規劃輸出：只有在需要工具時才強制檢查
    plan = res.get("plan")
    if expected_route in ("SIMPLE_TOOL", "COMPLEX_TOOL"):
        assert isinstance(
            plan, dict), f"Expected a planning output (dict) for route={expected_route}, got: {type(plan)}"
        # 可選：基本 key 存在就好（視你的 Simple/ComplexPlannerNode 輸出而定）
        # for k in ("type", "steps", "confidence"):
        #     assert k in plan, f"Plan missing key: {k}"
    else:
        # GENERAL_CHAT / EXIT 不強制有 plan
        assert plan is None or isinstance(
            plan, dict), "Plan should be None or dict for non-tool routes"


# 允許直接以 python 執行
if __name__ == "__main__":
    pytest.main([__file__])
