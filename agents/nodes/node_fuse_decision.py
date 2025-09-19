from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


# -------------------------
# Config & Utilities
# -------------------------

@dataclass
class FuseDecisionConfig:
    """
    Dynamic policy for fusing MainRouter 'needs' and QueryRouter candidates.

    Attributes:
        max_tools: Maximum number of tools to return in 'tools'.
        require_user_facing: If True, only allow tools that pass `is_user_facing(tool)` (when provided).
        min_qr_score: Minimum score for a QR candidate to be considered.
        adopt_qr_when_needs_empty: If True, allow using QR-only suggestions when MainRouter needs is empty.
        order_policy: How to order tools ("needs_first" | "qr_first" | "merge_by_score").
        prefer_exact_needs: If True, always keep needs tools even if QR scores are higher.
        collapse_duplicates: If True, de-duplicate tools while preserving order.
        simple_max_primary: If route == SIMPLE_TOOL, enforce returning <=1 tool.
        complex_min_primary: If route == COMPLEX_TOOL, ensure at least 2 tools if available (bounded by max_tools).
        allowed_capabilities: Optional allowlist (if provided, tools not in it will be filtered out early).
    """
    max_tools: int = 3
    require_user_facing: bool = True
    min_qr_score: float = 0.30
    adopt_qr_when_needs_empty: bool = True
    order_policy: str = "needs_first"  # or "qr_first" or "merge_by_score"
    prefer_exact_needs: bool = True
    collapse_duplicates: bool = True
    simple_max_primary: int = 1
    complex_min_primary: int = 2
    allowed_capabilities: Optional[Set[str]] = None


def _sanitize_needs(needs: Any) -> List[str]:
    """
    Accepts:
      - dict like {"tool_a": true, "tool_b": false}
      - list like ["tool_a", "tool_b"]
      - None
    Returns a list of tool names (truthy in dict case).
    """
    if needs is None:
        return []
    if isinstance(needs, dict):
        return [k for k, v in needs.items() if bool(v)]
    if isinstance(needs, list):
        return [t for t in needs if isinstance(t, str)]
    # Unknown format
    return []


def _sanitize_qr_candidates(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accepts either state["qr_candidates"] or legacy state["topk"] as the QR output.
    Each candidate is expected to contain at least:
      - "tool": str
      - "score": float
    Extra fields are preserved in debug only.
    """
    if isinstance(state.get("qr_candidates"), list):
        return state["qr_candidates"]
    if isinstance(state.get("topk"), list):
        return state["topk"]
    return []


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -------------------------
# Node Implementation
# -------------------------

class NodeFuseDecision:
    """
    Fuse MainRouter 'needs' and QueryRouter candidates into final {route, tools}.
    Keep this node strategy-only. Query retrieval/ranking should live in node_query_router.

    Input state (required):
      - route: str  # "SIMPLE_TOOL" | "COMPLEX_TOOL" | "GENERAL_CHAT" | "EXIT"
      - needs: dict|list|None

    Input state (optional):
      - qr_candidates or topk: list[{"tool": str, "score": float, ...}]
      - memory: MemoryManager | None

    Output (merged into state):
      - final_route: str
      - tools: list[str]
      - debug["fuse"]: dict   # rules applied, inputs snapshot, reasons, etc.

    Constructor deps (DI-friendly):
      - is_user_facing: Optional[Callable[[str], bool]]
      - config: FuseDecisionConfig
    """

    def __init__(
        self,
        *,
        is_user_facing: Optional[Callable[[str], bool]] = None,
        config: Optional[FuseDecisionConfig] = None,
    ) -> None:
        self.is_user_facing = is_user_facing
        self.cfg = config or FuseDecisionConfig()

    # ------------- Public API -------------

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Basic validation
        if not isinstance(state, dict):
            raise ValueError("NodeFuseDecision expected a dict state.")
        route = state.get("route")
        if not isinstance(route, str):
            raise ValueError("NodeFuseDecision requires state['route']: str")

        needs_raw = state.get("needs")
        needs_list = _sanitize_needs(needs_raw)

        qr_cands = _sanitize_qr_candidates(state)
        # Pre-filter QR by score
        qr_cands = [
            c for c in qr_cands
            if isinstance(c, dict)
            and isinstance(c.get("tool"), str)
            and isinstance(c.get("score"), (int, float))
            and float(c["score"]) >= self.cfg.min_qr_score
        ]

        # Apply allowlist + user-facing filter
        def _allowed(tool: str) -> bool:
            if self.cfg.allowed_capabilities is not None and tool not in self.cfg.allowed_capabilities:
                return False
            if self.cfg.require_user_facing and self.is_user_facing is not None:
                if not self.is_user_facing(tool):
                    return False
            return True

        needs_list = [t for t in needs_list if _allowed(t)]

        qr_by_tool: Dict[str, float] = {}
        for c in qr_cands:
            t = c["tool"]
            if not _allowed(t):
                continue
            sc = float(c["score"])
            # keep the best score per tool
            if t not in qr_by_tool or sc > qr_by_tool[t]:
                qr_by_tool[t] = sc

        # Determine candidate union based on policy
        selected_tools, rules_applied = self._select_tools(
            route=route,
            needs=needs_list,
            qr_scores=qr_by_tool,
        )

        # Enforce SIMPLE/COMPLEX constraints
        final_tools = self._enforce_route_cardinality(route, selected_tools)

        debug = {
            "inputs": {
                "route": route,
                "needs": needs_raw,
                "needs_sanitized": needs_list,
                "qr_candidates_len": len(qr_cands),
                "qr_scores": qr_by_tool,
                "order_policy": self.cfg.order_policy,
            },
            "rules_applied": rules_applied,
            "selected_before_enforce": selected_tools,
            "selected_after_enforce": final_tools,
        }

        out = dict(state)
        out["final_route"] = route
        out["tools"] = final_tools
        fuse_dbg = out.get("debug", {})
        fuse_dbg["fuse"] = debug
        out["debug"] = fuse_dbg
        return out

    # ------------- Selection Strategy -------------

    def _select_tools(
        self,
        *,
        route: str,
        needs: List[str],
        qr_scores: Dict[str, float],
    ) -> Tuple[List[str], List[str]]:
        """Return (selected_tools, rules_applied)."""
        rules: List[str] = []

        qr_sorted = sorted(qr_scores.items(),
                           key=lambda kv: kv[1], reverse=True)
        qr_order = [t for t, _ in qr_sorted]

        # Base orders by policy
        if self.cfg.order_policy == "qr_first":
            merged = qr_order + needs
            rules.append("order_policy=qr_first")
        elif self.cfg.order_policy == "merge_by_score":
            # Interleave needs (score=+inf) and qr by score
            score_map = {t: s for t, s in qr_sorted}
            needs_scored = [(t, float("inf")) for t in needs]
            merged_pairs = sorted(
                needs_scored + list(score_map.items()), key=lambda kv: kv[1], reverse=True)
            merged = [t for t, _ in merged_pairs]
            rules.append("order_policy=merge_by_score")
        else:
            # needs_first (default)
            merged = needs + qr_order
            rules.append("order_policy=needs_first")

        # If needs empty and allowed, adopt QR-only suggestions
        if not needs and self.cfg.adopt_qr_when_needs_empty and qr_order:
            rules.append("adopt_qr_when_needs_empty=True")
            base = merged
        else:
            base = merged

        if self.cfg.collapse_duplicates:
            base = _dedupe_preserve_order(base)
            rules.append("collapse_duplicates=True")

        # Optional: favor exact needs even if QR suggests others
        if self.cfg.prefer_exact_needs and needs:
            # Keep needs at their relative order, then fill with others
            rest = [t for t in base if t not in needs]
            base = needs + rest
            rules.append("prefer_exact_needs=True")

        # Apply max cap (soft here; route cardinality enforcement happens later)
        cap = max(1, int(self.cfg.max_tools))
        chosen = base[:cap]
        rules.append(f"cap={cap}")

        return chosen, rules

    def _enforce_route_cardinality(self, route: str, tools: List[str]) -> List[str]:
        """Honor simple/complex constraints without changing the declared route."""
        if route == "SIMPLE_TOOL":
            cap = max(1, int(self.cfg.simple_max_primary))
            return tools[:cap] if tools else []
        if route == "COMPLEX_TOOL":
            # Keep up to max_tools; try to ensure at least complex_min_primary if available
            cap = max(1, int(self.cfg.max_tools))
            trimmed = tools[:cap]
            if len(trimmed) < self.cfg.complex_min_primary:
                # If not enough tools available, just return what we have
                return trimmed
            return trimmed
        # For GENERAL_CHAT / EXIT, return empty list (manager/executor will handle)
        return []


# -------------------------
# Example usage (optional)
# -------------------------

if __name__ == "__main__":
    # A minimal is_user_facing callback and allowlist (optional)
    ALLOW = {
        "google_maps_search_places",
        "google_maps_directions",
        "google_calendar_create_event",
        "google_tasks_create_task",
        "google_maps_get_place_details",
    }

    def is_user_facing(tool: str) -> bool:
        return tool in ALLOW

    node = NodeFuseDecision(
        is_user_facing=is_user_facing,
        config=FuseDecisionConfig(
            max_tools=3,
            require_user_facing=True,
            min_qr_score=0.35,
            order_policy="needs_first",
            adopt_qr_when_needs_empty=True,
            allowed_capabilities=ALLOW,
        ),
    )

    # Simulated state
    state = {
        "route": "COMPLEX_TOOL",
        "needs": {"google_maps_search_places": True, "google_maps_directions": True},
        "qr_candidates": [
            {"tool": "google_maps_search_places", "score": 0.82},
            {"tool": "google_maps_get_place_details", "score": 0.65},
            {"tool": "not_in_allowlist", "score": 0.9},
        ],
    }

    out = node(state)
    import json
    print(json.dumps({
        "final_route": out["final_route"],
        "tools": out["tools"],
        "debug": out["debug"]["fuse"]
    }, ensure_ascii=False, indent=2))
