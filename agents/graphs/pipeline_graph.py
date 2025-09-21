# agents/graphs/pipeline_graph.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict, Set, Callable  # ← 加入 Callable

from langgraph.graph import StateGraph, END

from agents.graphs.routing_graph import build_routing_graph
from agents.graphs.planning_graph import build_planning_graph


class AgentState(TypedDict, total=False):
    # inputs
    user_text: str
    memory: Any

    # routing outputs
    route: str  # "SIMPLE_TOOL" | "COMPLEX_TOOL" | "GENERAL_CHAT" | "EXIT"
    needs: Any
    router_reason: Optional[str]
    router_raw: Optional[Dict[str, Any]]
    qr_candidates: List[Any]          # ← 明確元素型別（保守）
    topk: List[Any]                   # ← 明確元素型別（保守）
    final_route: str
    tools: List[str]
    debug: Dict[str, Any]

    # planning outputs
    cand_tools: List[Dict[str, str]]  # [{"name","desc"?}]
    # {"type","steps","clarification","confidence","raw"}
    plan: Dict[str, Any]


def build_main_graph(
    *,
    # routing_graph kwargs
    mode: str = "fusion",
    allowed_capabilities: Optional[Set[str]] = None,
    min_qr_score: float = 0.35,
    max_tools: int = 3,
    order_policy: str = "needs_first",
    prefer_exact_needs: bool = True,
    adopt_qr_when_needs_empty: bool = True,
    require_user_facing: bool = True,
    simple_max_primary: int = 1,
    complex_min_primary: int = 2,

    # planning_graph kwargs
    get_desc: Optional[Callable[[str], Optional[str]]] = None,  # ← 精確型別
):
    """
    建立合併圖（routing → planning）並回傳已編譯的 LangGraph app。
    - 第 1 階段：routing_graph 產生 final_route / tools
    - 第 2 階段：planning_graph 依 final_route 與 tools 生成 plan
    """
    routing_app = build_routing_graph(
        mode=mode,
        allowed_capabilities=allowed_capabilities,
        min_qr_score=min_qr_score,
        max_tools=max_tools,
        order_policy=order_policy,
        prefer_exact_needs=prefer_exact_needs,
        adopt_qr_when_needs_empty=adopt_qr_when_needs_empty,
        require_user_facing=require_user_facing,
        simple_max_primary=simple_max_primary,
        complex_min_primary=complex_min_primary,
    )
    planning_app = build_planning_graph(get_desc=get_desc)

    def n_routing(state: AgentState) -> AgentState:
        return routing_app.invoke(dict(state))  # 保留傳入欄位

    def n_planning(state: AgentState) -> AgentState:
        # GENERAL_CHAT/EXIT 會在 planning_graph 走 NONE→END，不出 plan
        return planning_app.invoke(dict(state))

    sg = StateGraph(AgentState)
    sg.add_node("routing", n_routing)
    sg.add_node("planning", n_planning)

    sg.set_entry_point("routing")
    sg.add_edge("routing", "planning")
    sg.add_edge("planning", END)

    return sg.compile()


def run_routing_planning(
    user_text: str,
    *,
    memory: Any = None,
    get_desc: Optional[Callable[[str], Optional[str]]] = None,  # ← 精確型別
    **routing_kwargs,
) -> Dict[str, Any]:
    """
    便捷執行函式：一次跑完整個 pipeline，回傳合併後狀態。
    routing_kwargs 會轉交給 build_main_graph 的 routing 參數。  # ← 修 docstring
    """
    app = build_main_graph(get_desc=get_desc, **routing_kwargs)

    init: AgentState = {"user_text": user_text, "memory": memory}
    out: AgentState = app.invoke(init)

    return {
        "final_route": out.get("final_route"),
        "tools": out.get("tools", []),
        "plan": out.get("plan"),
        "route": out.get("route"),
        "needs": out.get("needs"),
        "qr_candidates": out.get("qr_candidates", out.get("topk", [])),
        "debug": out.get("debug", {}),
    }


# 對外穩定別名（不綁階段順序，未來 pipeline 變動也不用改對外 API）
build_agent_graph = build_main_graph   # ← 新增這行別名
run_agent = run_routing_planning
