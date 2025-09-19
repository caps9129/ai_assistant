# agents/graphs/routing_graph.py
from __future__ import annotations
from typing import Any, Dict, Optional, Set, TypedDict

from langgraph.graph import StateGraph, END

# nodes
from agents.nodes.node_main_router import MainRouterNode
from agents.nodes.node_query_router import QueryRouterNode
from agents.nodes.node_fuse_decision import (
    NodeFuseDecision,
    FuseDecisionConfig,
)

# -----------------------
# User-facing allowlist
# （不包含 supporting tools，例如 geocode）
# -----------------------
DEFAULT_ALLOWED: Set[str] = {
    "google_maps_search_places",
    "google_maps_directions",
    "google_calendar_create_event",
    "google_tasks_create_task",
    "google_maps_get_place_details",
}


def _is_user_facing_factory(allowed: Set[str]):
    def _is_user_facing(tool: str) -> bool:
        return tool in allowed
    return _is_user_facing


# -----------------------
# LangGraph State schema
# -----------------------
class RouteState(TypedDict, total=False):
    user_text: str
    memory: Any

    # MainRouter output
    route: str  # "SIMPLE_TOOL" | "COMPLEX_TOOL" | "GENERAL_CHAT" | "EXIT"
    needs: Any  # dict|list|None
    router_reason: Optional[str]
    router_raw: Optional[Dict[str, Any]]

    # QueryRouter output
    qr_candidates: list                 # 規範化鍵（若 node 輸出 topk，會轉到這裡）
    topk: list                          # 部分舊版 node 的鍵；僅作 fallback
    qr_raw: Optional[Dict[str, Any]]

    # FuseDecision output
    final_route: str
    tools: list[str]
    debug: Dict[str, Any]


# -----------------------
# Builder
# -----------------------
def build_routing_graph(
    *,
    mode: str = "fusion",  # "fusion" | "main_only"
    allowed_capabilities: Optional[Set[str]] = None,
    min_qr_score: float = 0.35,
    max_tools: int = 3,
    # "needs_first" | "qr_first" | "merge_by_score"
    order_policy: str = "needs_first",
    prefer_exact_needs: bool = True,
    adopt_qr_when_needs_empty: bool = True,
    require_user_facing: bool = True,
    simple_max_primary: int = 1,
    complex_min_primary: int = 2,
    # 依賴注入（可選）
    main_node: Optional[MainRouterNode] = None,
    query_node: Optional[QueryRouterNode] = None,
    fuse_node: Optional[NodeFuseDecision] = None,
):
    """
    回傳已編譯的 LangGraph app。

    用法：
        app = build_routing_graph(mode="fusion")
        state = app.invoke({"user_text": "...", "memory": MemoryManager()})
        print(state["final_route"], state["tools"])
    """
    allowed = allowed_capabilities or DEFAULT_ALLOWED

    # 1) instantiate nodes
    main_node = main_node or MainRouterNode()
    query_node = query_node or QueryRouterNode()

    if fuse_node is None:
        cfg = FuseDecisionConfig(
            max_tools=max_tools,
            require_user_facing=require_user_facing,
            min_qr_score=min_qr_score,
            adopt_qr_when_needs_empty=adopt_qr_when_needs_empty,
            order_policy=order_policy,
            prefer_exact_needs=prefer_exact_needs,
            collapse_duplicates=True,
            simple_max_primary=simple_max_primary,
            complex_min_primary=complex_min_primary,
            allowed_capabilities=allowed,
        )
        fuse_node = NodeFuseDecision(
            is_user_facing=_is_user_facing_factory(allowed),
            config=cfg,
        )

    # 2) wrap node callables to LangGraph node fns
    def n_main(state: RouteState) -> RouteState:
        return main_node(dict(state))

    def n_query(state: RouteState) -> RouteState:
        # 小優化：若判定為一般對話或退出，就不跑 QR
        if state.get("route") in ("GENERAL_CHAT", "EXIT"):
            return dict(state)

        try:
            s = query_node(dict(state))
        except Exception as e:
            # 不要讓 QR 問題中斷流程
            dbg = dict(state.get("debug", {}))
            dbg["qr_error"] = str(e)
            s = dict(state)
            s["debug"] = dbg
            s["qr_candidates"] = []
            return s

        # --- 正規化 key：若只有 topk，轉為 qr_candidates ---
        if "qr_candidates" not in s and "topk" in s:
            s["qr_candidates"] = s.get("topk", [])
        return s

    def n_fuse(state: RouteState) -> RouteState:
        return fuse_node(dict(state))

    # 3) build graph
    sg = StateGraph(RouteState)

    sg.add_node("main_router", n_main)
    if mode == "fusion":
        sg.add_node("query_router", n_query)
    sg.add_node("fuse_decision", n_fuse)

    sg.set_entry_point("main_router")
    if mode == "fusion":
        # 先後順序執行（之後要平行也可改）
        sg.add_edge("main_router", "query_router")
        sg.add_edge("query_router", "fuse_decision")
    else:  # main_only
        sg.add_edge("main_router", "fuse_decision")

    sg.add_edge("fuse_decision", END)

    app = sg.compile()
    return app


# -----------------------
# Helper: simple runner
# -----------------------
def run_routing(user_text: str, *, memory=None, **kwargs) -> Dict[str, Any]:
    """
    便利函式：建圖並跑一次，輸出與舊版 AgentManager.plan() 對齊。
    """
    app = build_routing_graph(**kwargs)
    init: RouteState = {"user_text": user_text, "memory": memory}
    out: RouteState = app.invoke(init)

    # 回傳時也做一次 fallback：若沒有 qr_candidates 就用 topk
    qrc = out.get("qr_candidates", out.get("topk", []))

    debug = out.get("debug", {})
    debug.update({
        "mode": kwargs.get("mode", "fusion"),
        "router_reason": out.get("router_reason"),
        "qr_raw": out.get("qr_raw"),
    })

    return {
        "final_route": out.get("final_route"),
        "tools": out.get("tools", []),
        "debug": debug,
        "route": out.get("route"),
        "needs": out.get("needs"),
        "qr_candidates": qrc,
    }


# -----------------------
# Main (ad-hoc test)
# -----------------------
if __name__ == "__main__":
    from memory.manager import MemoryManager
    import json

    res = run_routing(
        "Find coffee near PDX",
        memory=MemoryManager(),
        mode="fusion",
        order_policy="needs_first",
    )
    print(json.dumps({
        "final_route": res["final_route"],
        "tools": res["tools"],
        "route": res["route"],
        "needs": res["needs"],
        "qr_k": len(res["qr_candidates"]),
        "rules": res["debug"].get("fuse", {}).get("rules_applied"),
    }, ensure_ascii=False, indent=2))
