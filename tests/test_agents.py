import pytest
from agents.manager import ManagerConfig, AgentManager

# Test cases are designed based on the logic and examples in main_router.txt.
# The 25 questions below cover the five primary routing scenarios:
# 1. SIMPLE_TOOL: A single user-facing tool is required.
# 2. COMPLEX_TOOL: Two or more user-facing tools are needed, typically search + navigation.
# 3. GENERAL_CHAT: No specific tool is relevant to the user's query.
# 4. EXIT: The user indicates they want to end the conversation.
# 5. EDGE_CASES: Handles empty, ambiguous, or unsupported requests.

TEST_CASES = [
    # === 1. SIMPLE_TOOL Scenarios (11 Cases) ===
    # --- google_maps_search_places ---
    ("Find coffee shops near me.", "SIMPLE_TOOL",
     ["google_maps_search_places"]),
    ("找一下附近有沒有在營業的超市", "SIMPLE_TOOL", ["google_maps_search_places"]),
    ("What's the address of the OSU Kelley Engineering Center?", "SIMPLE_TOOL", [
     "google_maps_search_places"]),

    # # # --- google_maps_directions ---
    ("How long does it to walk from my current location to the library?",
     "SIMPLE_TOOL", ["google_maps_directions"]),
    ("開車到台北101要多久？", "SIMPLE_TOOL", ["google_maps_directions"]),
    ("Navigate to 2501 SW Jefferson Way, Corvallis, OR", "SIMPLE_TOOL",
     ["google_maps_directions"]),

    # # # --- google_calendar_create_event ---
    ("Schedule a meeting for tomorrow at 3pm titled 'Project Sync'",
     "SIMPLE_TOOL", ["google_calendar_create_event"]),
    ("幫我預約下週三早上十點的牙醫回診", "SIMPLE_TOOL", ["google_calendar_create_event"]),

    # # # --- google_tasks_create_task ---
    ("Remind me to buy milk", "SIMPLE_TOOL", ["google_tasks_create_task"]),
    ("新增一個待辦事項：完成報告", "SIMPLE_TOOL", ["google_tasks_create_task"]),
    ("Add a to-do to call John by this Friday at 5pm",
     "SIMPLE_TOOL", ["google_tasks_create_task"]),

    # # # === 2. COMPLEX_TOOL Scenarios (6 Cases) ===
    ("Find the highest-rated pizza place nearby and take me there.",
     "COMPLEX_TOOL", ["google_maps_search_places", "google_maps_directions"]),
    ("導航到附近最近的加油站", "COMPLEX_TOOL", [
     "google_maps_search_places", "google_maps_directions"]),
    ("I need to find a pharmacy that's open now, what's the ETA?", "COMPLEX_TOOL", [
     "google_maps_search_places", "google_maps_directions"]),
    ("幫我找一家便宜的餐廳並告訴我怎麼走", "COMPLEX_TOOL", [
     "google_maps_search_places", "google_maps_directions"]),
    ("Search for parks within 5km and show me the route to the one with the best rating.",
     "COMPLEX_TOOL", ["google_maps_search_places", "google_maps_directions"]),
    ("What's the closest grocery store and how long will it take to drive there?",
     "COMPLEX_TOOL", ["google_maps_search_places", "google_maps_directions"]),

    # # # === 3. GENERAL_CHAT Scenarios (4 Cases) ===
    ("Tell me a joke.", "GENERAL_CHAT", []),
    ("Who is the current president of the United States?", "GENERAL_CHAT", []),
    ("今天天氣如何？", "GENERAL_CHAT", []),  # No weather tool is available.
    # No timer tool is available.
    ("Set a timer for 10 minutes.", "GENERAL_CHAT", []),

    # # # === 4. EXIT Scenarios (2 Cases) ===
    ("Thanks, that's all for now.", "EXIT", []),
    ("好，掰掰", "EXIT", []),
]


@pytest.mark.parametrize(
    "user_input, expected_route, expected_tools",
    TEST_CASES
)
def test_main_router_logic(user_input, expected_route, expected_tools):
    """
    Tests the main router by calling run_once with a variety of user inputs
    and asserting that the output matches the expected route and tool selections.
    """
    # GIVEN a user input string

    # WHEN the main router graph is invoked
    mgr = AgentManager(ManagerConfig(
        embedding_config_path="embeddings/config.json", run_mode="main_only"))
    state = mgr.plan(user_input)
    # print(87, state)

    # THEN the output state should contain the correct routing decision
    assert "final_route" in state, "The 'final_route' key should be in the final state"
    router_output = state["final_route"]

    assert router_output == expected_route, f"Expected route '{expected_route}' for input: '{user_input}'"

    # Sort lists to ensure comparison is order-independent
    actual_tools = sorted(state["tools"])

    # if (router_output == 'GENERAL_CHAT' or router_output == 'EXIT'):
    #     actual_tools = []

    assert actual_tools == sorted(
        expected_tools), f"Mismatch in user-facing tools expected_tools: {expected_tools} actual_tools: {actual_tools}"


# --- 新增的執行入口 ---
if __name__ == "__main__":
    """
    This block allows the test file to be executed directly as a module
    by invoking the pytest runner programmatically.
    """
    # We pass the current file's path (__file__) to pytest's main function.
    pytest.main([__file__])
