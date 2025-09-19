# agents/nodes/node_query_router.py
from __future__ import annotations
from typing import Any, Dict, Optional
from config import QUERY_ROUTER_CONFIG
import os

from agents.query_router import QueryRouter, RouterConfig


class QueryRouterNode:
    """
    Thin LangGraph node wrapper for embedding/FAISS-based QueryRouter.

    Initialization options:
      - Pass an existing `router` instance, OR
      - Provide `config_path` to build one, OR
      - Leave both None to read from env var QUERY_ROUTER_CONFIG.

    Input state (required):
      - user_text: str

    Input state (optional):
      - memory: MemoryManager | None   # preserved but unused here

    Output (merged back into state):
      - topk: list[dict]               # router's "results" (top-K per card)
      - qr_mode: str | None            # router.decision.mode ("simple" | "complex" | "none")
      - qr_selected: list[str] | None  # router.decision.selected (list of card ids)
      - qr_raw: dict                   # full router output for debugging/telemetry
    """

    def __init__(
        self,
        *,
        router: Optional[QueryRouter] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if router is not None:
            self.router = router
            return

        # resolve config path: explicit > env > error
        cfg_path = QUERY_ROUTER_CONFIG
        if not cfg_path:
            raise ValueError(
                "QueryRouterNode requires either a `router` instance or `config_path` "
                "or env var QUERY_ROUTER_CONFIG."
            )

        cfg = RouterConfig.from_json(cfg_path)
        self.router = QueryRouter(cfg)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # --- minimal validations ---
        if not isinstance(state, dict):
            raise ValueError("QueryRouterNode expected a dict state.")
        if "user_text" not in state or not isinstance(state["user_text"], str):
            raise ValueError(
                "QueryRouterNode requires state['user_text']: str.")

        user_text = state["user_text"]

        # Delegate to router (embeddings + FAISS search + scoring/decision)
        out = self.router.route(user_text)
        # out includes: {"query","results","results_all","decision", ...}

        decision = out.get("decision") or {}
        new_state = dict(state)
        new_state["topk"] = out.get("results", [])            # for fuse stage
        new_state["qr_mode"] = decision.get("mode")
        new_state["qr_selected"] = decision.get("selected")
        # keep full payload for debugging/telemetry
        new_state["qr_raw"] = out
        return new_state


if __name__ == "__main__":
    # Minimal smoke test (requires a valid QUERY_ROUTER_CONFIG and FAISS indices)
    import json

    try:
        node = QueryRouterNode()
        s = {"user_text": "Find coffee near PDX"}
        s2 = node(s)
        print(json.dumps(
            {k: s2[k] for k in ("qr_mode", "qr_selected")},
            ensure_ascii=False, indent=2
        ))
        print(f"TopK: {len(s2.get('topk', []))} items")
    except Exception as e:
        print(f"[node_query_router] init/run failed: {e}")
