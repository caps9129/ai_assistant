# agents/nodes/node_main_router.py
from __future__ import annotations
from typing import Any, Dict

from agents.main_router_agent import MainRouterAgent


class MainRouterNode:
    """
    Thin LangGraph node wrapper for MainRouterAgent.

    Input state (required):
      - user_text: str

    Input state (optional):
      - memory: MemoryManager | None   # passed through to agent

    Output (merged back into state):
      - route: str                     # "SIMPLE_TOOL" | "COMPLEX_TOOL" | "GENERAL_CHAT" | "EXIT"
      - needs: list[str]               # normalized tool names (may be empty)
      - router_reason: str | None      # optional, for debugging/telemetry
      - router_raw: dict | None        # optional, original model JSON for debugging
    """

    def __init__(self) -> None:
        self.agent = MainRouterAgent()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # --- minimal validations ---
        if not isinstance(state, dict):
            raise ValueError("MainRouterNode expected a dict state.")
        if "user_text" not in state or not isinstance(state["user_text"], str):
            raise ValueError(
                "MainRouterNode requires state['user_text']: str.")

        user_text = state["user_text"]
        memory = state.get("memory")

        # Delegate to agent
        out = self.agent.classify(user_text, memory=memory)
        # out shape: {"route": ..., "needs": [...], "reason": str, "raw": ...}

        new_state = dict(state)
        new_state["route"] = out.get("route")
        new_state["needs"] = out.get("needs", []) or []
        # keep debug fields separate to avoid clobbering other 'reason' keys
        new_state["router_reason"] = out.get("reason")
        new_state["router_raw"] = out.get("raw")
        return new_state


if __name__ == "__main__":
    # Minimal smoke test (optional)
    from memory.manager import MemoryManager
    import json

    node = MainRouterNode()
    s = {
        "user_text": "Find coffee near PDX",
        "memory": MemoryManager(),
    }
    s2 = node(s)
    print(json.dumps({k: s2[k] for k in ("route", "needs",
          "router_reason")}, ensure_ascii=False, indent=2))
