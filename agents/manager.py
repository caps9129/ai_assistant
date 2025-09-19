# agents/manager.py
from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

# --- External components already in the repo ---
from agents.main_router_agent import MainRouterAgent
from agents.query_router import QueryRouter, RouterConfig


# ----------------------------
# Helper dataclasses
# ----------------------------
@dataclass
class ManagerConfig:
    # Orchestration mode:
    #   - "fusion": run MainRouter + QueryRouter in parallel, then fuse (default)
    #   - "main_only": run only MainRouter, tools taken directly from needs (no mapping)
    run_mode: str = "fusion"

    # embeddings / router config (fusion mode only)
    embedding_config_path: str = "embeddings/config.json"

    # QueryRouter settings (fusion mode only)
    top_k: int = 3  # Always ask QueryRouter for top K distinct user-facing tools
    score_floor: float = 0.0  # we rely on needs coverage, not absolute thresholds

    # Consistency override (fusion mode)
    upgrade_to_complex_if_multi_need: bool = True  # if >=2 needs, force complex

    # Logging
    enable_debug_logging: bool = True

    def __post_init__(self):
        # Normalize run_mode
        self.run_mode = (self.run_mode or "fusion").lower().strip()
        if self.run_mode not in ("fusion", "main_only"):
            self.run_mode = "fusion"


def _setup_logger(enable_debug: bool = True) -> logging.Logger:
    logger = logging.getLogger("AgentManager")
    if not logger.handlers:
        level_name = os.getenv("AGENTMGR_LOGLEVEL",
                               "DEBUG" if enable_debug else "INFO")
        level = getattr(logging, level_name.upper(),
                        logging.DEBUG if enable_debug else logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    logger.setLevel(logging.DEBUG if enable_debug else logging.INFO)
    return logger


def _log_json(logger: logging.Logger, title: str, obj: Any, level: int = logging.DEBUG):
    try:
        logger.log(level, "%s: %s", title, json.dumps(obj, ensure_ascii=False))
    except Exception:
        logger.log(level, "%s (repr): %r", title, obj)


def _brief_results(results: List[Dict[str, Any]], max_items: int = 3) -> List[Dict[str, Any]]:
    out = []
    for r in results[:max_items]:
        out.append({
            "card": r.get("card"),
            "score": round(float(r.get("score", 0.0)), 4),
            "max_pos": round(float(r.get("max_pos", 0.0)), 4),
            "avg_topk": round(float(r.get("avg_topk", 0.0)), 4),
            "max_neg": round(float(r.get("max_neg", 0.0)), 4),
        })
    return out


# ----------------------------
# Core manager (fusion layer)
# ----------------------------
class AgentManager:
    """
    Stage-1 Orchestrator with two modes:

    1) fusion  (default):
       - 並行跑 MainRouter (route+needs) 與 QueryRouter (Top-K 候選)
       - 規則融合：一致性覆寫 + needs 精確匹配 + 兜底為 needs 名稱（無映射）
       - 回傳 {final_route, tools, debug}

    2) main_only:
       - 只跑 MainRouter（最低延遲）
       - 直接以 needs（真實 user-facing tool 名稱）決定 tools（不查 embedding、不做映射）
       - 回傳格式與 fusion 完全一致，便於 A/B 測
    """

    def __init__(self, cfg: Optional[ManagerConfig] = None):
        self.cfg = cfg or ManagerConfig()
        self.log = _setup_logger(self.cfg.enable_debug_logging)

        self.log.info("Initializing AgentManager... mode=%s",
                      self.cfg.run_mode)

        # MainRouter is always needed
        t0 = perf_counter()
        self.main_router = MainRouterAgent()
        t1 = perf_counter()
        self.log.info("MainRouterAgent initialized in %.1f ms",
                      (t1 - t0) * 1000)

        # Capability table for user-facing check in fusion
        self.card_cap: Dict[str, Dict[str, Any]] = {}

        # QueryRouter + capability map are only needed in fusion mode
        if self.cfg.run_mode == "fusion":
            t2 = perf_counter()
            self.router_cfg = RouterConfig.from_json(
                self.cfg.embedding_config_path)
            self.qrouter = QueryRouter(self.router_cfg)
            t3 = perf_counter()
            self.log.info("QueryRouter initialized in %.1f ms",
                          (t3 - t2) * 1000)

            # Load capability map from embeddings/config.json
            with open(self.cfg.embedding_config_path, "r", encoding="utf-8") as f:
                _cfg_raw = json.load(f)
            self.card_cap = _cfg_raw.get("capability", {})
            _log_json(self.log, "Loaded capability map", self.card_cap)
        else:
            self.log.info("Fusion components skipped (main_only mode).")

        self.log.info("AgentManager ready.")

    # ---------- public API ----------
    def plan(self, user_text: str) -> Dict[str, Any]:
        """
        Wrapper that dispatches to the selected run_mode.
        Returns: {final_route, tools, debug}
        """
        if self.cfg.run_mode == "main_only":
            return self._plan_main_only(user_text)
        # default fusion
        return self._plan_fusion(user_text)

    # ---------- MAIN ONLY ----------
    def _plan_main_only(self, user_text: str) -> Dict[str, Any]:
        self.log.info("=== PLAN (main_only) START ===")
        self.log.debug("User text: %s", user_text)

        t0 = perf_counter()
        mr_out = {}
        try:
            mr_out = self.main_router.classify(user_text)
        except Exception as e:
            self.log.exception("[MainRouter] error: %s", e)
            mr_out = {"route": "GENERAL_CHAT", "needs": [], "error": str(e)}
        t1 = perf_counter()
        self.log.info("[MainRouter] ok elapsed_ms=%.0f", (t1 - t0) * 1000)
        _log_json(self.log, "[MainRouter] out", mr_out)

        route = (mr_out or {}).get("route") or "GENERAL_CHAT"
        needs: List[str] = (mr_out or {}).get("needs") or []
        self.log.debug("Main-only normalized route: %s", route)
        self.log.debug("Main-only normalized needs: %s", needs)

        tools: List[str] = []
        debug: Dict[str, Any] = {"route_from_main": route, "needs": needs, "topk": [],
                                 "coverage": [], "fallbacks": []}

        # 直接從 needs 產生 tools（SIMPLE → 取首個；COMPLEX → 全部去重保序）
        if route == "SIMPLE_TOOL":
            if needs:
                tools = [needs[0]]
                debug["coverage"].append(
                    {"need": needs[0], "card": needs[0], "how": "need_exact"})
        elif route == "COMPLEX_TOOL":
            seen = set()
            for n in needs:
                if n not in seen:
                    tools.append(n)
                    seen.add(n)
                    debug["coverage"].append(
                        {"need": n, "card": n, "how": "need_exact"})
        else:
            # GENERAL_CHAT / EXIT → no tools
            pass

        result = {
            "final_route": route,
            "tools": tools,
            "debug": debug
        }
        _log_json(self.log, "FINAL PLAN (main_only)",
                  result, level=logging.INFO)
        self.log.info("=== PLAN (main_only) END ===")
        return result

    # ---------- FUSION (parallel main + query) ----------
    def _plan_fusion(self, user_text: str) -> Dict[str, Any]:
        self.log.info("=== PLAN (fusion) START ===")
        self.log.debug("User text: %s", user_text)

        mr_out, qr_out = self._run_parallel(user_text)

        # Normalize main_router output
        route = (mr_out or {}).get("route") or "GENERAL_CHAT"
        needs = (mr_out or {}).get("needs") or []
        self.log.debug("MainRouter normalized route: %s", route)
        self.log.debug("MainRouter normalized needs: %s", needs)

        # Normalize query_router output
        qr_results = (qr_out or {}).get("results", [])
        self.log.debug("QueryRouter Top-K count: %d", len(qr_results))
        self.log.debug("QueryRouter Top-K (brief): %s",
                       _brief_results(qr_results, max_items=3))

        # Consistency override：needs >= 2 一律 complex
        if self.cfg.upgrade_to_complex_if_multi_need and len(needs) >= 2 and route == "SIMPLE_TOOL":
            self.log.info("Override route -> COMPLEX_TOOL (needs >= 2)")
            route = "COMPLEX_TOOL"

        # Make final selection per rules
        final_route, chosen, debug = self._fuse_decision(
            route, needs, qr_results)

        result = {
            "final_route": final_route,
            "tools": chosen,           # user-facing only
            "debug": {
                "route_from_main": route,
                "needs": needs,
                "topk": qr_results,
                **debug
            }
        }
        _log_json(self.log, "FINAL PLAN (fusion)", result, level=logging.INFO)
        self.log.info("=== PLAN (fusion) END ===")
        return result

    # ---------- internal ----------
    def _run_parallel(self, user_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """在同一個 stage 並行跑 main_router 與 query_router，等兩者完成後回傳"""
        def call_main():
            t0 = perf_counter()
            try:
                out = self.main_router.classify(user_text)
                t1 = perf_counter()
                self.log.info("[MainRouter] ok elapsed_ms=%.0f out_keys=%s",
                              (t1 - t0) * 1000, list(out.keys()) if isinstance(out, dict) else type(out))
                _log_json(self.log, "[MainRouter] out", out)
                return out
            except Exception as e:
                t1 = perf_counter()
                self.log.exception(
                    "[MainRouter] error after %.0f ms: %s", (t1 - t0) * 1000, e)
                return {"route": "GENERAL_CHAT", "needs": [], "error": str(e)}

        def call_query():
            t0 = perf_counter()
            try:
                out = self.qrouter.route(user_text, top_k=self.cfg.top_k)
                t1 = perf_counter()
                self.log.info(
                    "[QueryRouter] ok elapsed_ms=%.0f", (t1 - t0) * 1000)
                _log_json(self.log, "[QueryRouter] out.brief", {
                    "results_brief": _brief_results(out.get("results", []), max_items=self.cfg.top_k),
                    "n_all": len(out.get("results_all", []))
                })
                return out
            except Exception as e:
                t1 = perf_counter()
                self.log.exception(
                    "[QueryRouter] error after %.0f ms: %s", (t1 - t0) * 1000, e)
                return {"results": [], "error": str(e)}

        self.log.debug("Launching parallel routers...")
        outs = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_mr = ex.submit(call_main)
            fut_qr = ex.submit(call_query)

            for fut in as_completed([fut_mr, fut_qr]):
                try:
                    res = fut.result()
                    if res and isinstance(res, dict) and "results" in res:
                        outs["qr"] = res
                    else:
                        outs["mr"] = res
                except Exception as e:
                    if fut is fut_mr:
                        outs["mr"] = {"route": "GENERAL_CHAT",
                                      "needs": [], "error": str(e)}
                    else:
                        outs["qr"] = {"results": [], "error": str(e)}
        self.log.debug("Parallel routers finished. Keys: %s",
                       list(outs.keys()))
        return outs.get("mr", {}), outs.get("qr", {})

    def _is_user_facing(self, card: str) -> bool:
        meta = self.card_cap.get(card) or {}
        return not bool(meta.get("is_support", False))

    # --- simplified selection helpers (no capability mapping) ---

    def _pick_exact_match_from_topk(
        self, needed_tool: str, candidates: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        從 Top-K 候選中找與 needed_tool 完全相同名稱的卡（且為 user-facing）。
        """
        for r in candidates:
            card = r.get("card")
            if card and card == needed_tool and self._is_user_facing(card):
                return card
        return None

    def _fuse_decision(
        self,
        route: str,
        needs: List[str],
        results: List[Dict[str, Any]],
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        依規則做融合決策，產生最終 tools（只含 user-facing）
        規則（無映射、無 fallback_by_capability）：
          - SIMPLE: 若 Top-K 有與 needs[0] 同名卡 → 用它；否則用 needs[0]；若無 needs → 用 Top-K 第一張 user-facing（若有）
          - COMPLEX: 對每個 need 依序：若 Top-K 有同名卡則用之；否則直接用 need；最後去重保序
        """
        debug: Dict[str, Any] = {"coverage": [], "fallbacks": []}
        chosen: List[str] = []

        self.log.info("Fusion start: route=%s needs=%s", route, needs)
        self.log.debug("Candidate results (brief): %s",
                       _brief_results(results, max_items=5))

        # SIMPLE
        if route == "SIMPLE_TOOL":
            if needs:
                need0 = needs[0]
                pick = self._pick_exact_match_from_topk(need0, results)
                if pick:
                    chosen = [pick]
                    debug["coverage"].append(
                        {"need": need0, "card": pick, "how": "topk_exact"})
                    self.log.info("SIMPLE -> choose topk exact: %s", pick)
                else:
                    chosen = [need0]
                    debug["coverage"].append(
                        {"need": need0, "card": need0, "how": "need_exact"})
                    self.log.info("SIMPLE -> choose need directly: %s", need0)
            else:
                # 無 needs → 取第一張 user-facing（若存在）
                top1 = None
                for r in results:
                    if r.get("card") and self._is_user_facing(r["card"]):
                        top1 = r["card"]
                        break
                if top1:
                    chosen = [top1]
                    debug["coverage"].append(
                        {"need": None, "card": top1, "how": "topk_first"})
                    self.log.warning(
                        "SIMPLE guard pick first available: %s", top1)
            return "SIMPLE_TOOL", chosen, debug

        # COMPLEX
        if route == "COMPLEX_TOOL":
            ordered: List[str] = []
            for need in needs:
                pick = self._pick_exact_match_from_topk(need, results)
                if pick:
                    ordered.append(pick)
                    debug["coverage"].append(
                        {"need": need, "card": pick, "how": "topk_exact"})
                else:
                    ordered.append(need)
                    debug["coverage"].append(
                        {"need": need, "card": need, "how": "need_exact"})

            # 去重保序
            seen = set()
            for card in ordered:
                if card not in seen and self._is_user_facing(card):
                    chosen.append(card)
                    seen.add(card)

            # 若完全沒有 needs（少見），就選前 2 張 user-facing
            if not needs and not chosen:
                for r in results:
                    c = r.get("card")
                    if c and self._is_user_facing(c):
                        chosen.append(c)
                    if len(chosen) >= 2:
                        break
                self.log.info(
                    "COMPLEX no-needs -> choose first two user-facing: %s", chosen)

            self.log.info("COMPLEX chosen tools: %s", chosen)
            _log_json(self.log, "COMPLEX coverage", debug["coverage"])
            return "COMPLEX_TOOL", chosen, debug

        # GENERAL_CHAT / EXIT：不使用工具
        self.log.info("Route=%s -> no tools", route)
        return route, [], debug


# ----------------------------
# Minimal CLI test
# ----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Stage-1 Manager test (fusion or main_only).")
    ap.add_argument("--text", required=True, help="user input")
    ap.add_argument("--config", default="embeddings/config.json")
    ap.add_argument("--mode", default="fusion",
                    choices=["fusion", "main_only"])
    args = ap.parse_args()

    mgr = AgentManager(ManagerConfig(
        run_mode=args.mode,
        embedding_config_path=args.config
    ))
    out = mgr.plan(args.text)

    print("\n=== FINAL PLAN ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))
