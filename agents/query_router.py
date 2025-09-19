from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable, Optional
import numpy as np
import faiss
from openai import OpenAI
from config import OPENAI_API_KEY

# ---------- IO helpers ----------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = vec.astype("float32")
    n = np.linalg.norm(v) + 1e-12
    return v / n

# ---------- Config & state ----------


@dataclass
class RouterConfig:
    model: str
    normalize: bool

    retrieval_pos_top_m: int
    retrieval_neg_top_l: int
    retrieval_avg_topk: int

    pos_max_w: float
    pos_avg_w: float

    neg_lambda: float
    neg_t: float

    meta_beta: float
    field_weights: Dict[str, float]

    exclude_support: bool
    dedupe_same_domain: bool

    simple_tau: float
    simple_margin: float
    complex_delta: float
    complex_kmax: int

    paths: Dict[str, str]
    logging_topk_per_card: int
    emit_prototypes: bool

    per_card_normalize: bool
    results_topk: int

    @staticmethod
    def from_json(path: str) -> "RouterConfig":
        cfg = json.load(open(path, "r", encoding="utf-8"))

        scoring = cfg.get("scoring", {})
        selection = cfg.get("selection", {})
        retrieval = cfg.get("retrieval", {})
        paths = cfg.get("paths", {})
        logging = cfg.get("logging", {})

        per_card_normalize = cfg.get("per_card_normalize", False)
        results_topk = selection.get("results_topk", 3)

        return RouterConfig(
            model=cfg.get("model") or paths.get(
                "model") or "text-embedding-3-large",
            normalize=cfg.get("normalize", True),

            retrieval_pos_top_m=retrieval["pos_top_m"],
            retrieval_neg_top_l=retrieval["neg_top_l"],
            retrieval_avg_topk=retrieval["avg_topk"],

            pos_max_w=scoring["pos"]["max_weight"],
            pos_avg_w=scoring["pos"]["avg_topk_weight"],

            neg_lambda=scoring["neg"]["lambda"],
            neg_t=scoring["neg"]["t_neg"],

            meta_beta=scoring["meta"]["beta"],
            field_weights=scoring.get("field_weights", {}),

            exclude_support=selection.get("exclude_support", False),
            dedupe_same_domain=selection.get("dedupe_same_domain", False),

            simple_tau=selection["simple"]["tau"],
            simple_margin=selection["simple"]["margin"],
            complex_delta=selection["complex"]["delta_window"],
            complex_kmax=selection["complex"]["k_max"],

            paths=paths,
            logging_topk_per_card=logging.get("show_topk_per_card", 3),
            emit_prototypes=logging.get("emit_prototypes", True),

            per_card_normalize=per_card_normalize,
            results_topk=results_topk,
        )

# ---------- Router ----------


class QueryRouter:
    def __init__(self, config: RouterConfig):
        self.cfg = config
        # FAISS
        self.pos_index = faiss.read_index(self.cfg.paths["pos_index"])
        self.neg_index = faiss.read_index(self.cfg.paths["neg_index"])
        # items
        self.pos_items = read_jsonl(self.cfg.paths["pos_items"])
        self.neg_items = read_jsonl(self.cfg.paths["neg_items"])
        # card meta
        self.cards_meta = json.load(
            open(self.cfg.paths["cards_meta"], "r", encoding="utf-8"))

        # id -> item
        self.pos_id2item = {row["id"]: row for row in self.pos_items}
        self.neg_id2item = {row["id"]: row for row in self.neg_items}

        # support cards set
        self.support_cards = {
            k for k, v in self.cards_meta.items() if v.get("is_support", False)}

        # optional per-card score normalization
        self.card_norms: Dict[str, Dict[str, float]] = {}
        norms_path = self.cfg.paths.get("card_norms")
        if norms_path and os.path.exists(norms_path):
            try:
                self.card_norms = json.load(
                    open(norms_path, "r", encoding="utf-8"))
            except Exception:
                self.card_norms = {}

        self.client = OpenAI()

    # --- Embedding
    def embed(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.cfg.model, input=text)
        v = np.array(resp.data[0].embedding, dtype="float32")
        return l2_normalize(v) if self.cfg.normalize else v

    # --- FAISS search
    def _search(self, index, vec: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        vec2 = vec.reshape(1, -1).astype("float32")
        # inner product; cosine if L2-normalized
        sims, ids = index.search(vec2, topk)
        return ids[0], sims[0]

    # --- Aggregate positives per card
    def _pos_score_by_card(self, pos_hits: List[Tuple[int, float]]) -> Dict[str, Dict[str, Any]]:
        by_card: Dict[str, List[float]] = {}
        keep_hits: Dict[str, List[Tuple[int, float]]] = {}
        for hid, sim in pos_hits:
            item = self.pos_id2item.get(int(hid))
            if not item:
                continue
            card = item["card"]
            if self.cfg.exclude_support and card in self.support_cards:
                continue
            by_card.setdefault(card, []).append(float(sim))
            keep_hits.setdefault(card, []).append((hid, float(sim)))

        res: Dict[str, Dict[str, Any]] = {}
        K = self.cfg.retrieval_avg_topk
        for card, sims in by_card.items():
            sims_sorted = sorted(sims, reverse=True)
            max_pos = sims_sorted[0]
            avg_topk = sum(sims_sorted[:K]) / min(K, len(sims_sorted))
            pos_score = self.cfg.pos_max_w * max_pos + self.cfg.pos_avg_w * avg_topk
            res[card] = {
                "max_pos": max_pos,
                "avg_topk": avg_topk,
                "pos_score": pos_score,
                "hits": keep_hits[card],
            }
        return res

    # --- NEG penalty (thresholded)
    def _neg_penalty_by_card(self, qvec: np.ndarray, candidate_cards: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        ids, sims = self._search(self.neg_index, qvec,
                                 self.cfg.retrieval_neg_top_l)
        res = {c: {"max_neg": 0.0, "penalty": 0.0} for c in candidate_cards}
        for hid, sim in zip(ids, sims):
            item = self.neg_id2item.get(int(hid))
            if not item:
                continue
            card = item["card"]
            if card not in res:
                continue
            s = float(sim)
            if s > res[card]["max_neg"]:
                res[card]["max_neg"] = s
        lam, t = self.cfg.neg_lambda, self.cfg.neg_t
        for card, d in res.items():
            d["penalty"] = lam * max(0.0, d["max_neg"] - t)
        return res

    # --- Meta bonus (neutral stub)
    def _meta_bonus(self, query: str, card: str) -> float:
        return 0.0

    # --- Optional per-card normalization (z-score)
    def _maybe_normalize_scores(self, results: List[Dict[str, Any]]) -> None:
        if not self.cfg.per_card_normalize or not self.card_norms:
            return
        for r in results:
            card = r["card"]
            stats = self.card_norms.get(card)
            if not stats:
                continue
            mu = float(stats.get("mean", 0.0))
            sigma = float(stats.get("std", 1.0)) or 1.0
            r["norm_score"] = (r["score"] - mu) / sigma
        for r in results:
            if "norm_score" in r:
                r["score"] = r["norm_score"]

    # --- Public API
    def route(self, query: str) -> Dict[str, Any]:
        qvec = self.embed(query)

        # 1) POS retrieval
        pos_ids, pos_sims = self._search(
            self.pos_index, qvec, self.cfg.retrieval_pos_top_m)
        pos_hits = [(int(i), float(s))
                    for i, s in zip(pos_ids, pos_sims) if i != -1]
        pos_by_card = self._pos_score_by_card(pos_hits)
        candidate_cards = list(pos_by_card.keys())

        # 2) NEG for candidates
        neg_by_card = self._neg_penalty_by_card(qvec, candidate_cards)

        # 3) Score per card
        results_full: List[Dict[str, Any]] = []
        for card in candidate_cards:
            pos_d = pos_by_card[card]
            neg_d = neg_by_card.get(card, {"max_neg": 0.0, "penalty": 0.0})
            meta_b = self._meta_bonus(query, card)
            score = pos_d["pos_score"] - neg_d["penalty"] + \
                self.cfg.meta_beta * meta_b
            results_full.append({
                "card": card,
                "score": float(score),
                "max_pos": float(pos_d["max_pos"]),
                "avg_topk": float(pos_d["avg_topk"]),
                "pos_score": float(pos_d["pos_score"]),
                "max_neg": float(neg_d["max_neg"]),
                "neg_penalty": float(neg_d["penalty"]),
                "meta_bonus": float(meta_b),
                "hits": pos_d["hits"][: self.cfg.logging_topk_per_card],
            })

        # 4) Normalize (optional) and sort
        self._maybe_normalize_scores(results_full)
        results_full.sort(key=lambda x: x["score"], reverse=True)

        # 5) Decide: simple vs complex
        decision = self._decide_simple_complex(results_full)

        # 6) Build Top-K (distinct cards). Ensure selected cards included first.
        k = max(1, int(self.cfg.results_topk))
        by_card = {r["card"]: r for r in results_full}
        selected_cards: List[str] = decision.get("selected", [])

        final_results: List[Dict[str, Any]] = []
        seen: set = set()
        for c in selected_cards:
            if c in by_card and c not in seen:
                final_results.append(by_card[c])
                seen.add(c)
        for r in results_full:
            if len(final_results) >= k:
                break
            if r["card"] in seen:
                continue
            final_results.append(r)
            seen.add(r["card"])
        results_topk = final_results[:k]

        return {
            "query": query,
            "results": results_topk,
            "results_all": results_full,
            "decision": decision,
        }

    def _decide_simple_complex(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"mode": "none", "selected": []}

        top1 = results[0]
        top2 = results[1] if len(results) > 1 else None
        tau = self.cfg.simple_tau
        margin = self.cfg.simple_margin

        # SIMPLE if top1 strong and clear margin
        if top1["score"] >= tau and (top2 is None or (top1["score"] - top2["score"] >= margin)):
            return {"mode": "simple", "selected": [top1["card"]], "top1": top1}

        # Otherwise COMPLEX: take a window around top1
        window = self.cfg.complex_delta
        kmax = self.cfg.complex_kmax
        cutoff = top1["score"] - window
        selected = [r["card"] for r in results if r["score"] >= cutoff][:kmax]

        if self.cfg.dedupe_same_domain and len(selected) > 1:
            meta = self.cards_meta
            kept: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for r in results:
                if r["card"] not in selected:
                    continue
                key = (meta.get(r["card"], {}).get("domain", ""),
                       meta.get(r["card"], {}).get("capability", ""))
                if key not in kept or r["score"] > kept[key]["score"]:
                    kept[key] = r
            selected = [r["card"] for r in sorted(
                kept.values(), key=lambda x: x["score"], reverse=True)][:kmax]

        return {"mode": "complex", "selected": selected, "topK": results[:kmax]}
