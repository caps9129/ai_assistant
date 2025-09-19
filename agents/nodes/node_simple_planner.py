from __future__ import annotations
from typing import Any, Dict

from agents.simple_planner_agent import SimplePlannerAgent


class SimplePlannerNode:
    """
    Thin LangGraph node wrapper for SimplePlannerAgent.

    Input state (required):
      - user_text: str
      - cand_tools: list[str] | list[dict]   # e.g., ["tool_a", ...] or [{"name":"tool_a","desc":"..."}]

    Input state (optional):
      - memory: MemoryManager | None         # passed through to agent

    Output:
      - state["plan"]: dict                  # { type, steps, clarification, confidence, raw }
    """

    def __init__(self) -> None:
        self.agent = SimplePlannerAgent()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # --- minimal validations ---
        if not isinstance(state, dict):
            raise ValueError("SimplePlannerNode expected a dict state.")
        if "user_text" not in state or not isinstance(state["user_text"], str):
            raise ValueError(
                "SimplePlannerNode requires state['user_text']: str.")
        if "cand_tools" not in state or not isinstance(state["cand_tools"], list):
            raise ValueError(
                "SimplePlannerNode requires state['cand_tools']: list[str|dict].")

        user_text = state["user_text"]
        cand_tools = state["cand_tools"]
        memory = state.get("memory")

        # Delegate to agent (agent handles normalization & JSON parsing)
        plan = self.agent.plan(
            user_text=user_text,
            cand_tools=cand_tools,
            memory=memory,
        )

        # Merge back into state
        out = dict(state)
        out["plan"] = plan
        return out


if __name__ == "__main__":
    # Minimal smoke test
    from memory.manager import MemoryManager
    import json

    node = SimplePlannerNode()
    state1 = {
        "user_text": "Show 24/7 pharmacies around Shinjuku Station.",
        "cand_tools": [
            {
                "name": "google_maps_search_places",
                "desc": "Search for nearby places centered on the user's current location (cafes, gyms, supermarkets, etc.) with filters."
            }
        ],
        "memory": MemoryManager()
    }
    state2 = {
        "user_text": "Take me there right now.",
        "cand_tools": [
            {
                "name": "google_maps_directions",
                "desc": "Route from origin (default current location) to destination."
            },
            {
                "name": "google_maps_geocode_address",
                "desc": "Supporting tool to disambiguate place strings into addresses for directions."
            }
        ],
        "memory": MemoryManager()
    }
    state3 = {
        "user_text": "What's the phone number for Din Tai Fung?",
        "cand_tools": [
            {
                "name": "google_maps_get_place_details",
                "desc": "Fetch details (hours, phone, website, address, rating) for a single identified place by name or place_id."
            }
        ],
        "memory": MemoryManager()
    }
    for state in [state1, state2, state3]:
        print("="*20)
        new_state = node(state)
        print(json.dumps(new_state["plan"], ensure_ascii=False, indent=2))
