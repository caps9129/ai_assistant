from __future__ import annotations
from typing import Any, Dict, List, Optional

from agent_config import AGENT_REGISTRY
from agents.base import BaseAgentConfig, OpenAIChatAgentBase

_ALLOWED_SUPPORT = {"google_maps_geocode_address"}


class ComplexPlannerAgent(OpenAIChatAgentBase):
    """
    Multi-step planner for COMPLEX (with possible downshift to SIMPLE).
    - Primary steps MUST be chosen ONLY from cand_tools (exact name match).
    - Supporting steps are owned here (only google_maps_geocode_address).
    - Type rule:
        * if any supporting step OR >=2 primaries -> type="COMPLEX_TOOL"
        * if exactly one primary and NO support -> type="SIMPLE_TOOL"
    """

    def __init__(self):
        cfg = AGENT_REGISTRY.get_config("ComplexPlannerAgent")
        if not cfg:
            raise ValueError("Config for 'ComplexPlannerAgent' not found.")

        base_cfg = BaseAgentConfig(
            id="ComplexPlannerAgent",
            prompt_name=cfg.prompt_name,   # e.g. "complex_planner"
            model=cfg.model,
            temperature=getattr(cfg, "temperature", 0.0),
            max_tokens=getattr(cfg, "max_tokens", None),
            history_limit_pairs=getattr(cfg, "history_limit_pairs", 0),
            max_chars=getattr(cfg, "max_chars", 12000),
            output_mode="json",
        )
        super().__init__(base_cfg)
        print("✅ ComplexPlannerAgent ready.")

    def plan(
        self,
        user_text: str,
        cand_tools: List[Any],
        memory=None,
    ) -> Dict[str, Any]:
        norm_cand = self._normalize_cand_tools(cand_tools)

        payload = {
            "user_text": user_text,
            "cand_tools": norm_cand,
        }

        raw = self.process_request(payload, memory)

        if isinstance(raw, str):
            return {
                "type": "COMPLEX_TOOL",
                "steps": [],
                "clarification": raw.strip() or "Could you clarify which specific action you want me to plan?",
                "confidence": 0.0,
                "raw": {"fallback": raw},
            }

        out = self._normalize_output(raw, norm_cand)
        out["raw"] = raw
        return out

    # ---------- helpers ----------
    def _normalize_cand_tools(self, cand_tools: List[Any]) -> List[Dict[str, str]]:
        norm: List[Dict[str, str]] = []
        if not isinstance(cand_tools, list):
            return norm
        for item in cand_tools:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    norm.append({"name": name})
            elif isinstance(item, dict):
                name = (item.get("name") or "").strip()
                if name:
                    desc = item.get("desc")
                    norm.append(
                        {"name": name, **({"desc": desc} if isinstance(desc, str) else {})})
        return norm

    def _normalize_output(self, raw: Any, cand_tools: List[Dict[str, str]]) -> Dict[str, Any]:
        result = {
            "type": "COMPLEX_TOOL",
            "steps": [],
            "clarification": None,
            "confidence": 0.0,
        }
        if not isinstance(raw, dict):
            return result

        cand_names = [c["name"] for c in cand_tools]
        cand_set = set(cand_names)

        typ = raw.get("type")
        steps_in = raw.get("steps")
        clar = raw.get("clarification")
        conf = raw.get("confidence")

        steps_clean: List[Dict[str, Any]] = []
        has_support = False
        primary_count = 0

        if isinstance(steps_in, list):
            for s in steps_in:
                if not isinstance(s, dict):
                    continue
                tool = s.get("tool")
                use = s.get("use")
                if not isinstance(tool, str) or not isinstance(use, str):
                    continue
                tname = tool.strip()
                use_l = use.strip().lower()

                if use_l == "primary":
                    if tname in cand_set:
                        steps_clean.append(s)
                        primary_count += 1
                elif use_l == "support":
                    if tname in _ALLOWED_SUPPORT:
                        steps_clean.append(s)
                        has_support = True

        enforced = "COMPLEX_TOOL" if has_support or primary_count >= 2 else "SIMPLE_TOOL"
        if typ not in ("COMPLEX_TOOL", "SIMPLE_TOOL"):
            typ = enforced
        else:
            if typ == "SIMPLE_TOOL" and (has_support or primary_count >= 2):
                typ = "COMPLEX_TOOL"
            elif typ == "COMPLEX_TOOL" and (primary_count == 1 and not has_support):
                typ = "SIMPLE_TOOL"

        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        conf_f = 0.0 if conf_f < 0 else 1.0 if conf_f > 1.0 else conf_f

        clar_s = (clar.strip() if isinstance(clar, str) else None) or None
        if not steps_clean and clar_s is None:
            clar_s = "Could you clarify which specific action you want me to plan?"

        return {
            "type": typ,
            "steps": steps_clean,
            "clarification": clar_s,
            "confidence": conf_f,
        }


if __name__ == '__main__':
    # Minimal local test
    import json
    from memory.manager import MemoryManager

    agent = ComplexPlannerAgent()
    memory = MemoryManager()

    # Example inputs
    user_text = "Find good restaurants and take me to the closest."
    cand_tools = [
        {"name": "google_maps_search_places", "desc": "Find nearby places."},
        {"name": "google_calendar_create_event",
            "desc": "Create a simple Google Calendar event (all-day or 1-hour timed) in the primary calendar."},
        {"name": "google_maps_directions",
            "desc": "Route from origin to destination."},

    ]

    payload = {
        "user_text": user_text,
        "cand_tools": cand_tools,
    }
    # Call as a normal request (non-streaming)

    raw = agent.process_request(payload, memory)
    out = agent._normalize_output(raw, agent._normalize_cand_tools(cand_tools))
    # out["raw"] = raw
    print(json.dumps(out, ensure_ascii=False, indent=2))
