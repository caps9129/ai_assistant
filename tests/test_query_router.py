# tests/test_query_router.py
# -*- coding: utf-8 -*-

import argparse
import json
from typing import List, Tuple, Dict, Any

from agents.query_router import QueryRouter, RouterConfig


# ------------------------------
# 測試查詢（10 SIMPLE + 10 COMPLEX）
# ------------------------------
def get_test_queries():
    simple: List[Tuple[str, str]] = [
        ("Find highly rated coffee shops near me, open now (≥4.5).",
         "google_maps_search_places"),
        ("在 Albany 2 公里內找健身房，依距離排序。", "google_maps_search_places"),
        ("How long to walk from Kelley to Valley Library?", "google_maps_directions"),
        ("從 OSU 開車到 Portland，避開收費道路。", "google_maps_directions"),
        ("Transit route to PDX arriving by 7:30 AM.", "google_maps_directions"),
        ("Create a meeting tomorrow 3–4pm with Alice and Bob at Kelley 2001, add a Meet link.",
         "google_calendar_create_event"),
        ("每週一早上 9:00–9:15 standup，附上線上會議連結。", "google_calendar_create_event"),
        ("新增待辦：週五下午 5 點前繳交作業，放進預設清單。", "google_tasks_create_task"),
        ("Add a subtask under “Project Alpha”: Prepare slides.",
         "google_tasks_create_task"),
        ("請用中文提供從 KEC 到 PDX 的開車路線，避開高速公路。", "google_maps_directions"),
    ]

    complex_: List[Tuple[str, List[str]]] = [
        ("Find the highest-rated pizza place nearby and take me there.",
         ["google_maps_search_places", "google_maps_directions"]),
        ("導航到附近最近的加油站",
         ["google_maps_search_places", "google_maps_directions"]),
        ("I need to find a pharmacy that's open now, what's the ETA?",
         ["google_maps_search_places", "google_maps_directions"]),
        ("幫我找一家便宜的餐廳並告訴我怎麼走",
         ["google_maps_search_places", "google_maps_directions"]),
        ("Search for parks within 5km and show me the route to the one with the best rating.",
         ["google_maps_search_places", "google_maps_directions"]),
        ("What's the closest grocery store and how long will it take to drive there?",
         ["google_maps_search_places", "google_maps_directions"]),
        ("Find a good sushi place near campus and navigate me there.",
         ["google_maps_search_places", "google_maps_directions"]),
        ("幫我找現在有開的超市，順便規劃走路過去。",
         ["google_maps_search_places", "google_maps_directions"]),
        ("Schedule project kickoff next Tuesday afternoon and add a to-do to prepare slides.",
         ["google_calendar_create_event", "google_tasks_create_task"]),
        ("明天上午安排 30 分鐘的討論，另外提醒我今晚先寫大綱。",
         ["google_calendar_create_event", "google_tasks_create_task"]),
    ]
    return simple, complex_


# ------------------------------
# Utility
# ------------------------------
def print_config_summary(cfg: RouterConfig):
    print("\n=== CONFIG ===")
    print(f"model={cfg.model}  normalize={cfg.normalize}")
    print(
        f"retrieval: pos_top_m={cfg.retrieval_pos_top_m}  neg_top_l={cfg.retrieval_neg_top_l}  avg_topk={cfg.retrieval_avg_topk}")
    print(
        f"scoring  : λ={cfg.neg_lambda}  t_neg={cfg.neg_t}  β={cfg.meta_beta}")
    print(f"selection: τ={cfg.simple_tau}  Δ={cfg.simple_margin}  δ={cfg.complex_delta}  Kmax={cfg.complex_kmax}  results_topk={cfg.results_topk}")
    print(
        f"exclude_support={cfg.exclude_support}  dedupe_same_domain={cfg.dedupe_same_domain}  per_card_normalize={cfg.per_card_normalize}")


def pretty_result_line(r: Dict[str, Any]) -> str:
    return json.dumps({
        "card": r["card"],
        "score": round(r["score"], 4),
        "max_pos": round(r["max_pos"], 4),
        "avg_topk": round(r["avg_topk"], 4),
        "max_neg": round(r["max_neg"], 4),
        "neg_penalty": round(r["neg_penalty"], 4),
        "meta_bonus": round(r["meta_bonus"], 4),
    }, ensure_ascii=False)


def assert_results_shape(out: Dict[str, Any], topk_expected: int) -> List[str]:
    """確保 results 只有 Top-K 且卡片唯一；回傳卡片清單"""
    results = out.get("results", [])
    cards = [r["card"] for r in results]
    assert len(
        results) <= topk_expected, f"results 超過 Top-K: {len(results)} > {topk_expected}"
    assert len(cards) == len(set(cards)), f"results 內卡片重複: {cards}"
    return cards


def ensure_selected_in_results(out: Dict[str, Any]):
    """新版 QueryRouter 會把 selected 注入 Top-K；這裡做一致性檢查"""
    decision = out.get("decision", {})
    selected = decision.get("selected") or []
    cards = [r["card"] for r in out.get("results", [])]
    for c in selected:
        assert c in cards, f"selected 中的 {c} 未出現在 Top-K results：{cards}"


def dump_prototype_hits(router: QueryRouter, out: Dict[str, Any], topk_show: int, per_card_hits: int):
    print(f"\nTOP {topk_show} SCORES:")
    for r in out["results"][:topk_show]:
        print("  " + pretty_result_line(r))

    print(f"\nPROTOTYPE HITS PER CARD (top {per_card_hits})")
    for r in out["results"][:topk_show]:
        card = r["card"]
        hits = r.get("hits", [])[:per_card_hits]
        if not hits:
            continue
        print(f"  [Card] {card}")
        for hid, sim in hits:
            item = router.pos_id2item.get(int(hid), {})
            field = item.get("field", "?")
            lang = item.get("lang", "?")
            text_preview = item.get(
                "text_preview", "") or item.get("text", "")[:80]
            print(
                f"    • sim={sim:.4f}  field={field:<4}  lang={lang:<2}  text='{text_preview}'")


def run_and_log(router: QueryRouter, cfg: RouterConfig, query: str, expected, idx: int, group: str) -> bool:
    out = router.route(query)
    decision = out.get("decision", {})
    mode = decision.get("mode")
    selected = decision.get("selected", [])

    print("\n" + "=" * 90)
    print(f"[{group.upper()} #{idx}]  {query}")
    print(f"DECISION: {json.dumps(decision, ensure_ascii=False)}")

    # 形狀、包含性檢查
    cards = assert_results_shape(out, cfg.results_topk)
    ensure_selected_in_results(out)

    # 判定 PASS/FAIL
    passed = False
    if group == "simple":
        passed = (mode == "simple" and selected and selected[0] == expected)
    else:
        exp_set = set(expected)
        sel_set = set(selected)
        passed = (mode == "complex" and len(exp_set & sel_set) > 0)

    print(
        f"RESULT  : {'PASS' if passed else 'FAIL'}  expected={expected}  selected={selected}  topk_cards={cards}")

    # dump_prototype_hits(router, out, cfg.results_topk,
    #                     cfg.logging_topk_per_card)
    return passed


# ------------------------------
# Runner
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Batch run routing tests for QueryRouter (Top-K candidates).")
    ap.add_argument("--config", default="embeddings/config.json",
                    help="Path to embeddings/config.json")
    ap.add_argument("--group", default="all",
                    choices=["all", "simple", "complex"], help="Run which group")
    args = ap.parse_args()

    cfg = RouterConfig.from_json(args.config)
    router = QueryRouter(cfg)
    print_config_summary(cfg)

    simple, complex_ = get_test_queries()

    total = 0
    passed = 0

    if args.group in ("all", "simple"):
        for i, (q, exp) in enumerate(simple, start=1):
            ok = run_and_log(router, cfg, q, exp, i, "simple")
            total += 1
            passed += int(ok)

    if args.group in ("all", "complex"):
        for i, (q, exp_list) in enumerate(complex_, start=1):
            ok = run_and_log(router, cfg, q, exp_list, i, "complex")
            total += 1
            passed += int(ok)

    print("\n" + "=" * 90)
    print(f"[SUMMARY] group={args.group}  passed={passed}/{total}")


if __name__ == "__main__":
    main()
