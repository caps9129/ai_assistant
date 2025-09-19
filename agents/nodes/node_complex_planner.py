# agents/nodes/node_complex_planner.py
from __future__ import annotations
from typing import Any, Dict

from agents.complex_planner_agent import ComplexPlannerAgent


class ComplexPlannerNode:
    """
    Thin LangGraph node wrapper for ComplexPlannerAgent.

    Input state (required):
      - user_text: str
      - cand_tools: list[str] | list[dict]   # e.g., ["tool_a", ...] or [{"name":"tool_a","desc":"..."}]
    Input state (optional):
      - memory: MemoryManager | None         # passed through to agent

    Output:
      - state["plan"]: dict                  # { type, steps, clarification, confidence, raw }
    """

    def __init__(self) -> None:
        self.agent = ComplexPlannerAgent()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # --- minimal validations ---
        if not isinstance(state, dict):
            raise ValueError("ComplexPlannerNode expected a dict state.")
        if "user_text" not in state or not isinstance(state["user_text"], str):
            raise ValueError(
                "ComplexPlannerNode requires state['user_text']: str.")
        if "cand_tools" not in state or not isinstance(state["cand_tools"], list):
            raise ValueError(
                "ComplexPlannerNode requires state['cand_tools']: list[str|dict].")

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

    node = ComplexPlannerNode()
    state = {
        "user_text": "Find good restaurants and take me to the closest.",
        "cand_tools": [
            {"name": "google_maps_search_places", "desc": "Find nearby places."},
            {"name": "google_maps_directions",
                "desc": "Route from origin to destination."},
        ],
        "memory": MemoryManager(),
    }
    new_state = node(state)
    print(json.dumps(new_state["plan"], ensure_ascii=False, indent=2))
