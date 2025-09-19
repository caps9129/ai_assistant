# agents/graphs/planning_graph.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable, TypedDict

from langgraph.graph import StateGraph, END

# use existing nodes (single source of truth)
from agents.nodes.node_simple_planner import SimplePlannerNode
from agents.nodes.node_complex_planner import ComplexPlannerNode


class PlanState(TypedDict, total=False):
    # inputs (from routing_graph)
    user_text: str
    memory: Any
    # "SIMPLE_TOOL" | "COMPLEX_TOOL" | "GENERAL_CHAT" | "EXIT"
    final_route: str
    tools: List[str]                 # names chosen by routing_graph

    # derived (this graph)
    cand_tools: List[Dict[str, str]]  # [{"name","desc"?}] for planner
    # {"type","steps","clarification","confidence","raw"?}
    plan: Dict[str, Any]
    debug: Dict[str, Any]            # optional debug bucket


def _make_desc_lookup(get_desc: Optional[Callable[[str], Optional[str]]] = None) -> Callable[[str], Optional[str]]:
    """
    Return a safe function name -> short description.
    Priority:
      1) user-supplied get_desc(name)
      2) try import tool_registry.* (common patterns)
      3) fallback: None
    """
    if callable(get_desc):
        return get_desc

    reg_funcs: List[Callable[[str], Optional[str]]] = []
    try:
        import tool_registry as tr  # type: ignore
        if hasattr(tr, "get_desc") and callable(getattr(tr, "get_desc")):
            reg_funcs.append(getattr(tr, "get_desc"))
        if hasattr(tr, "get_tool_desc") and callable(getattr(tr, "get_tool_desc")):
            reg_funcs.append(getattr(tr, "get_tool_desc"))
        if hasattr(tr, "get_short_desc") and callable(getattr(tr, "get_short_desc")):
            reg_funcs.append(getattr(tr, "get_short_desc"))
        for key in ("DESCRIPTIONS", "TOOL_DESCRIPTIONS", "TOOL_CARDS"):
            if hasattr(tr, key) and isinstance(getattr(tr, key), dict):
                d: Dict[str, Any] = getattr(tr, key)
                reg_funcs.append(lambda name, _d=d: _d.get(
                    name) if isinstance(_d.get(name), str) else None)
    except Exception:
        pass

    if reg_funcs:
        def _lookup(name: str) -> Optional[str]:
            for fn in reg_funcs:
                try:
                    v = fn(name)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                except Exception:
                    continue
            return None
        return _lookup

    def _none(_name: str) -> Optional[str]:
        return None
    return _none


class _PrepCandToolsNode:
    """
    Build cand_tools = [{"name","desc"?}] from tools[] using a description lookup.
    """

    def __init__(self, get_desc: Callable[[str], Optional[str]]) -> None:
        self.get_desc = get_desc

    def __call__(self, state: PlanState) -> PlanState:
        names: List[str] = state.get("tools", []) or []
        cand: List[Dict[str, str]] = []
        for n in names:
            if not isinstance(n, str):
                continue
            entry: Dict[str, str] = {"name": n}
            try:
                desc = self.get_desc(n)
            except Exception:
                desc = None
            if isinstance(desc, str) and desc.strip():
                entry["desc"] = desc.strip()
            cand.append(entry)

        new_state = dict(state)
        new_state["cand_tools"] = cand
        dbg = dict(new_state.get("debug", {}))
        dbg.setdefault("planning", {})
        dbg["planning"]["cand_tools_len"] = len(cand)
        new_state["debug"] = dbg
        return new_state


def build_planning_graph(
    *,
    get_desc: Optional[Callable[[str], Optional[str]]] = None,
):
    """
    Build a LangGraph app that:
      - takes {user_text, memory, final_route, tools}
      - produces {plan} using Simple/Complex planner (depending on final_route)
      - preserves original fields in state
    """
    desc_lookup = _make_desc_lookup(get_desc)
    prep = _PrepCandToolsNode(desc_lookup)

    # Reuse the canonical nodes
    n_simple = SimplePlannerNode()
    n_complex = ComplexPlannerNode()

    sg = StateGraph(PlanState)

    sg.add_node("prep_cand_tools", prep)
    sg.add_node("simple_planner", n_simple)    # <- imported node
    sg.add_node("complex_planner", n_complex)  # <- imported node

    sg.set_entry_point("prep_cand_tools")

    def _route_decision(state: PlanState) -> str:
        r = (state.get("final_route") or "").upper()
        if r == "SIMPLE_TOOL":
            return "SIMPLE"
        if r == "COMPLEX_TOOL":
            return "COMPLEX"
        return "NONE"

    sg.add_conditional_edges(
        "prep_cand_tools",
        _route_decision,
        {
            "SIMPLE": "simple_planner",
            "COMPLEX": "complex_planner",
            "NONE": END,  # GENERAL_CHAT / EXIT → no planning
        },
    )

    sg.add_edge("simple_planner", END)
    sg.add_edge("complex_planner", END)

    return sg.compile()


def run_planning(
    *,
    user_text: str,
    final_route: str,
    tools: List[str],
    memory: Any = None,
    get_desc: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper to run the planning subgraph.

    Inputs:
      - user_text, final_route, tools, memory

    Returns (merged shape):
      {
        "final_route": <same as input>,
        "tools": [...],
        "plan": {type, steps, clarification, confidence, raw?},
        "debug": {...}
      }
    """
    app = build_planning_graph(get_desc=get_desc)

    init: PlanState = {
        "user_text": user_text,
        "final_route": final_route,
        "tools": tools,
        "memory": memory,
    }
    out: PlanState = app.invoke(init)

    return {
        "final_route": out.get("final_route", final_route),
        "tools": out.get("tools", tools),
        "plan": out.get("plan"),
        "debug": out.get("debug", {}),
    }


if __name__ == "__main__":
    from memory.manager import MemoryManager
    import json

    mem = MemoryManager()

    # SIMPLE example
    s1 = run_planning(
        user_text="Navigate to Taipei 101.",
        final_route="SIMPLE_TOOL",
        tools=["google_maps_directions"],
        memory=mem,
    )
    print("[SIMPLE]")
    print(json.dumps(s1, ensure_ascii=False, indent=2))

    # COMPLEX example
    s2 = run_planning(
        user_text="Find good restaurants and take me to the closest.",
        final_route="COMPLEX_TOOL",
        tools=["google_maps_search_places", "google_maps_directions"],
        memory=mem,
    )
    print("[COMPLEX]")
    print(json.dumps(s2, ensure_ascii=False, indent=2))
