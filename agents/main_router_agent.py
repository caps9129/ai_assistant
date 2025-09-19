# agents/main_router_agent.py
from __future__ import annotations
from typing import Dict, Any, Optional, List

from agent_config import AGENT_REGISTRY
from agents.base import BaseAgentConfig, OpenAIChatAgentBase

_ALLOWED_ROUTES = {"SIMPLE_TOOL", "COMPLEX_TOOL", "GENERAL_CHAT", "EXIT"}

# Allowed capabilities now map to REAL tool function names from the router output.
# (Added google_maps_geocode_address to align with new needs schema.)
_ALLOWED_CAPABILITIES = {
    "google_maps_search_places",
    "google_maps_directions",
    "google_calendar_create_event",
    "google_tasks_create_task",
    "google_maps_get_place_details"
}

# NOTE: _NEEDS_SYNONYMS removed per new router contract:
# Router now outputs needs as a boolean dict keyed by REAL function names.
# e.g.
# "needs": {
#   "google_maps_search_places": true/false,
#   "google_maps_directions": true/false,
#   "google_calendar_create_event": true/false,
#   "google_tasks_create_task": true/false,
#   "google_maps_get_place_details": true/false,
#   "google_maps_geocode_address": true/false
# }


class MainRouterAgent(OpenAIChatAgentBase):
    """
    LLM-based intention classifier.
    Returns:
      - route: SIMPLE_TOOL / COMPLEX_TOOL / GENERAL_CHAT / EXIT
      - needs: list[str] of enabled tool function names from the allowed set
    """

    def __init__(
        self,
        *,
        default_model: str = "gpt-4o-mini",
        default_temperature: float = 0.0,
        default_history_limit_pairs: int = 0,
    ) -> None:
        cfg = AGENT_REGISTRY.get_config("MainRouterAgent")
        base_cfg = BaseAgentConfig(
            id="MainRouterAgent",
            prompt_name=(getattr(cfg, "prompt_name", None) or "main_router"),
            model=(getattr(cfg, "model", None) or default_model),
            temperature=float(
                getattr(cfg, "temperature", default_temperature)),
            output_mode="json"
            # force_json=True,
        )
        super().__init__(base_cfg)
        print("✅ MainRouterAgent ready.")

    # ----- Public API -----
    def classify(self, user_text: str, memory=None) -> Dict[str, Any]:
        """
        Returns:
        {
          "route": "SIMPLE_TOOL" | "COMPLEX_TOOL" | "GENERAL_CHAT" | "EXIT",
          "needs": ["google_maps_search_places", "google_maps_directions", ...],  # may be empty
          "reason": "<optional string>",
          "raw": <LLM raw output or fallback info>
        }
        """
        raw = self.process_request(user_text, memory)

        # 1) Model returned non-JSON (e.g., a bare token like "SIMPLE_TOOL")
        if isinstance(raw, dict) and raw.get("error") == "model_output_not_json":
            s = (raw.get("raw") or "").strip()
            route = self._coerce_route_from_string(s)
            needs: List[str] = []  # no JSON → do not guess needs
            return {"route": route, "needs": needs, "reason": "coerced_from_plain_text", "raw": {"route": s}}

        # 2) Model returned a plain string
        if isinstance(raw, str):
            route = self._coerce_route_from_string(raw)
            needs: List[str] = []
            return {"route": route, "needs": needs, "reason": "coerced_from_plain_text", "raw": {"route": raw}}

        # 3) Normal JSON: normalize route & needs
        route, reason = self._normalize_route_and_reason(raw)
        needs = self._normalize_needs(raw)
        return {"route": route, "needs": needs, "reason": reason, "raw": raw}

    # ----- Helpers -----
    def _normalize_route_and_reason(self, raw: Any) -> tuple[str, str]:
        route = "GENERAL_CHAT"
        reason = ""
        if isinstance(raw, dict):
            r = raw.get("route")
            if isinstance(r, str):
                r_up = r.strip().upper()
                if r_up in _ALLOWED_ROUTES:
                    route = r_up
            rr = raw.get("reason")
            if isinstance(rr, str):
                reason = rr.strip()

        if route not in _ALLOWED_ROUTES and isinstance(raw, dict):
            alias = (raw.get("route") or "").strip().lower()
            if alias in {"simple", "simple_tool"}:
                route = "SIMPLE_TOOL"
            elif alias in {"complex", "complex_tool"}:
                route = "COMPLEX_TOOL"
            elif alias in {"chat", "general", "general_chat"}:
                route = "GENERAL_CHAT"
            elif alias in {"quit", "exit"}:
                route = "EXIT"
            else:
                route = "GENERAL_CHAT"
        return route, reason

    def _normalize_needs(self, raw: Any) -> List[str]:
        """
        Extract & normalize needs from model JSON.
        Supports:
        1) Boolean dict:
           {"needs": {
             "google_maps_search_places": true, "google_maps_directions": false, ...
           }}
        2) Array of strings/objects:
           {"needs": ["google_maps_search_places", {"capability":"google_maps_directions"}]}
        3) Same keys under "capabilities", "need", "required_capabilities", "need_capabilities".
        """
        if not isinstance(raw, dict):
            return []

        candidates: List[str] = []

        # Handle boolean dicts under "needs" or "capabilities"
        for k_dict in ("needs", "capabilities"):
            v = raw.get(k_dict)
            if isinstance(v, dict):
                for cap, flag in v.items():
                    if isinstance(cap, str) and bool(flag):
                        candidates.append(cap)

        # Handle array-like shapes
        possible_keys = ["needs", "capabilities", "need",
                         "required_capabilities", "need_capabilities"]
        for key in possible_keys:
            v = raw.get(key)
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, str):
                        candidates.append(it)
                    elif isinstance(it, dict):
                        cap = it.get("capability") or it.get(
                            "name") or it.get("type")
                        if isinstance(cap, str):
                            candidates.append(cap)
            elif isinstance(v, str):
                candidates.append(v)

        # Final normalization: whitelist + de-dup (preserve order)
        out: List[str] = []
        seen = set()
        for s in candidates:
            k = s.strip()
            # Names are already real function names; lower/upper not required,
            # but keep case-insensitive safety:
            k_lower = k.lower()
            # Match allowed set in a case-insensitive manner
            match = None
            for allowed in _ALLOWED_CAPABILITIES:
                if allowed.lower() == k_lower:
                    match = allowed
                    break
            if match and match not in seen:
                out.append(match)
                seen.add(match)

        return out

    def _coerce_route_from_string(self, s: str) -> str:
        t = s.strip().lower()
        if t in {"simple", "simple_tool", "simpletool"}:
            return "SIMPLE_TOOL"
        if t in {"complex", "complex_tool", "complextool"}:
            return "COMPLEX_TOOL"
        if t in {"chat", "general", "general_chat"}:
            return "GENERAL_CHAT"
        if t in {"exit", "quit"}:
            return "EXIT"
        # Support already-uppercase enums
        t_up = s.strip().upper()
        if t_up in _ALLOWED_ROUTES:
            return t_up
        return "GENERAL_CHAT"


# ---- ad-hoc quick test ----
if __name__ == '__main__':
    from memory.manager import MemoryManager
    from typing import List

    agent = MainRouterAgent()
    memory = MemoryManager()

    samples: List[str] = [
        "Find coffee shops near me",
    ]
    for text in samples:
        res = agent.classify(text, memory)
        print("---")
        print(text)
        print(res)
