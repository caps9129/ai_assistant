#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build global embeddings for ToolCards into two FAISS indices (pos/neg), with L2 normalization
and rich logs so you can verify what went in.

Outputs (under --outdir, default ./embeddings_global):
  - pos.index.faiss                # FAISS index (cosine via IP on L2-normalized vectors)
  - pos.items.jsonl                # sidecar: id → {item_id, card, field, lang, ...}
  - neg.index.faiss
  - neg.items.jsonl
  - meta.cards.json                # per-card static metadata (provider/domain/capability/...)
  - build_info.json                # model, dim, time, counts, params

Design notes:
- We exclude `aliases` / `display_name` from embedding to reduce cross-card noise.
- Positives contain: one_line_desc(en/zh), positive_queries(en/zh), args_hint signal (en/zh),
  and a compact tag line (provider domain capability modalities).
- Negatives contain: negative_queries(en/zh).
- Each embedded item gets a stable item_id: "<card>|<field>|<lang>|<seq>" where field ∈ {desc,pos,neg,args,tags}.
- We L2-normalize *all* vectors (both index and queries later). With IndexFlatIP this yields cosine similarity.
"""

import os
from config import OPENAI_API_KEY
import glob
import json
import argparse
import datetime
import re
import hashlib
from typing import Any, Dict, List, Tuple, Iterable
from collections import Counter, defaultdict
import yaml
import numpy as np
import faiss
from openai import OpenAI


# -------------------- helpers --------------------

_EN_ZH_KEY_RE = re.compile(r'^(\s*-?\s*)(en|zh)\s*:\s*(.+?)\s*$')


def _needs_quotes(val: str) -> bool:
    """
    判斷此值是否『建議/需要』加引號：
    - 未以引號開頭
    - 並且符合下列任一條件：
      a) 含有 ': '（冒號+空白）或 ' #'（註解）
      b) 以 { 或 [ 開頭（易被當成 mapping/list）
      c) 含有非轉義的雙引號不等狀況可保持，用外層引號包裹即可
    """
    if not val:
        return False
    starts_quoted = val.startswith('"') or val.startswith("'")
    if starts_quoted:
        return False
    if ': ' in val or ' #' in val:
        return True
    if val.lstrip().startswith('{') or val.lstrip().startswith('['):
        return True
    return False


def _quote(val: str) -> str:
    """用雙引號包起來，並把內部的雙引號轉義。"""
    v = val
    v = v.replace('"', r'\"')
    return f"\"{v}\""


def autofix_yaml_text(raw: str, file_label: str = "") -> Tuple[str, List[str]]:
    """
    只處理【單行】的 en:/zh: 文字值：
    - 若值含 ': ' 或 ' #' 或以 { / [ 開頭，且未加引號 → 自動補上雙引號
    - 回傳 (fixed_text, logs)
    """
    logs = []
    out_lines = []
    for idx, line in enumerate(raw.splitlines(), start=1):
        m = _EN_ZH_KEY_RE.match(line)
        if m:
            prefix, lang, val = m.groups()
            if _needs_quotes(val):
                fixed = f"{prefix}{lang}: {_quote(val)}"
                logs.append(
                    f"[AUTOFIX] {file_label}:{idx}: {line.strip()}  ->  {fixed.strip()}")
                out_lines.append(fixed)
                continue
        out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if raw.endswith("\n") else ""), logs


def load_yaml_with_autofix(path: str, verbose: bool = True):
    """
    讀檔 → 自動補引號 → safe_load。
    若補過，引出詳細 log；若仍解析失敗，丟出原始錯誤讓你定位。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    fixed, logs = autofix_yaml_text(raw, file_label=os.path.basename(path))
    if logs and verbose:
        print("\n".join(logs))
    try:
        return yaml.safe_load(fixed)
    except yaml.YAMLError as e:
        # 解析仍失敗：印出數行上下文方便查
        caret = f"[YAML ERROR] while loading {path}: {e}"
        # 只示範前 80 行，可自行調整或列印 e.problem_mark
        snippet = "\n".join(fixed.splitlines()[:80])
        print(caret)
        raise


def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\ufeff", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def sha1(s: str) -> str:
    return "sha1:" + hashlib.sha1(s.encode("utf-8")).hexdigest()


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def iter_lang_items(mixed_list: List[Any], default_lang="en") -> Iterable[Tuple[str, str]]:
    """
    Accepts YAML lists like:
      - en: "foo"
      - zh: "bar"
    Or plain strings:
      - "plain string"  # treated as default_lang
    Yields (lang, text)
    """
    for item in as_list(mixed_list):
        if isinstance(item, dict):
            for lang, txt in item.items():
                if isinstance(txt, str) and txt.strip():
                    yield (lang.lower(), norm_text(txt))
        elif isinstance(item, str):
            yield (default_lang, norm_text(item))


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    # mat: (n, d)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -------------------- flatten logic --------------------


# provider/domain/capability/modalities, duplicated as en/zh for balance
INCLUDE_TAG_LINE = True


def make_arg_signal(card: Dict[str, Any]) -> Dict[str, str]:
    """Turn args_hint into concise human-readable signals in EN/ZH."""
    hints = as_list(card.get("args_hint", []))
    if not hints:
        return {}
    en = "Args: " + ", ".join(hints) + "."
    zh = "參數提示：" + "、".join(hints) + "。"
    return {"en": en, "zh": zh}


def flatten_card(card: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Returns (pos_items, neg_items, card_meta)
    Each item: {item_id, card, field, lang, text, weight, is_support, version}
    """
    name = card.get("name", "unknown_card")
    is_support = bool(card.get("is_support", False))
    version = str(card.get("version", ""))
    provider = card.get("provider", "")
    domain = card.get("domain", "")
    capability = card.get("capability", "")
    modalities = card.get("modalities", [])

    pos_items, neg_items = [], []

    # 1) one_line_desc (dict with en/zh)
    desc = card.get("one_line_desc", {})
    if isinstance(desc, dict):
        for lang, txt in desc.items():
            if isinstance(txt, str) and txt.strip():
                pos_items.append({
                    "item_id": f"{name}|desc|{lang}|0000",
                    "card": name, "field": "desc", "lang": lang,
                    "text": norm_text(txt), "weight": 1.0,
                    "is_support": is_support, "version": version
                })

    # 2) args_hint -> signal line (en/zh)
    arg_sig = make_arg_signal(card)
    for lang, txt in arg_sig.items():
        pos_items.append({
            "item_id": f"{name}|args|{lang}|0000",
            "card": name, "field": "args", "lang": lang,
            "text": norm_text(txt), "weight": 1.0,
            "is_support": is_support, "version": version
        })

    # 3) positive_queries
    seq_pos = 0
    for lang, txt in iter_lang_items(card.get("positive_queries", [])):
        pos_items.append({
            "item_id": f"{name}|pos|{lang}|{seq_pos:04d}",
            "card": name, "field": "pos", "lang": lang,
            "text": txt, "weight": 1.0,
            "is_support": is_support, "version": version
        })
        seq_pos += 1

    # 4) negative_queries
    seq_neg = 0
    for lang, txt in iter_lang_items(card.get("negative_queries", [])):
        neg_items.append({
            "item_id": f"{name}|neg|{lang}|{seq_neg:04d}",
            "card": name, "field": "neg", "lang": lang,
            "text": txt, "weight": 1.0,
            "is_support": is_support, "version": version
        })
        seq_neg += 1

    # 5) tags (provider/domain/capability/modalities)
    if INCLUDE_TAG_LINE:
        tag = f"{provider} {domain} {capability} {','.join(modalities)}".strip(
        )
        if tag:
            for lang in ("en", "zh"):
                pos_items.append({
                    "item_id": f"{name}|tags|{lang}|0000",
                    "card": name, "field": "tags", "lang": lang,
                    "text": tag, "weight": 1.0,
                    "is_support": is_support, "version": version
                })

    # De-duplicate within this card on (field, lang, text)
    seen = set()

    def dedup(rows):
        out = []
        for r in rows:
            key = (r["field"], r["lang"], r["text"])
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out

    card_meta = {
        "name": name,
        "version": version,
        "provider": provider,
        "domain": domain,
        "capability": capability,
        "modalities": modalities,
        "is_support": is_support
    }
    return dedup(pos_items), dedup(neg_items), card_meta

# -------------------- embedding --------------------


def embed_texts(texts: List[str], model: str, batch_size: int, log_prefix: str) -> np.ndarray:

    client = OpenAI()

    vecs = []
    total = len(texts)
    dim = None
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_vecs = [np.array(d.embedding, dtype="float32")
                      for d in resp.data]
        if dim is None:
            dim = batch_vecs[0].shape[0]
            print(f"[EMB] {log_prefix}: embedding dim = {dim}")
        vecs.append(np.vstack(batch_vecs))
        print(f"[EMB] {log_prefix}: {min(i+batch_size, total)}/{total} done")

    mat = np.vstack(vecs) if vecs else np.zeros(
        (0, dim or 1536), dtype="float32")
    # L2 normalize
    mat = l2_normalize(mat)
    print(f"[EMB] {log_prefix}: L2 normalization applied to all vectors.")
    return mat

# -------------------- FAISS build --------------------


def build_faiss_ip(vectors: np.ndarray, ids: np.ndarray):
    # IndexFlatIP + unit vectors = cosine similarity
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    if ids is not None:
        index = faiss.IndexIDMap2(index)  # store user-provided ids
        index.add_with_ids(vectors, ids)
    else:
        index.add(vectors)
    return index

# -------------------- main --------------------


def main():
    ap = argparse.ArgumentParser(
        description="Build global FAISS indices (pos/neg) from ToolCard YAMLs with L2-normalized embeddings.")
    ap.add_argument("--cards-glob", default="tool_cards/*.y*ml",
                    help="Glob for YAML cards")
    ap.add_argument("--outdir", default="embeddings_global",
                    help="Output directory")
    ap.add_argument("--model", default="text-embedding-3-large",
                    help="OpenAI embedding model")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Embedding batch size")
    ap.add_argument("--cards-version", default="unknown",
                    help="Cards version or git hash")
    args = ap.parse_args()

    print("=== CONFIG ===")
    print(f"cards_glob     : {args.cards_glob}")
    print(f"outdir         : {args.outdir}")
    print(f"model          : {args.model}")
    print(f"batch_size     : {args.batch_size}")
    print(f"cards_version  : {args.cards_version}")
    print("==============")

    paths = sorted(glob.glob(args.cards_glob))
    if not paths:
        raise SystemExit(f"No YAML files matched: {args.cards_glob}")
    print(f"[INFO] Found {len(paths)} ToolCards:")
    for p in paths:
        print(f"  - {os.path.basename(p)}")

    all_pos, all_neg = [], []
    cards_meta = {}
    per_card_stats = {}

    # Flatten
    for p in paths:
        card = yaml.safe_load(open(p, "r", encoding="utf-8"))
        # card = load_yaml_with_autofix(p, verbose=True)
        name = card.get("name", os.path.basename(p))
        pos_items, neg_items, card_meta = flatten_card(card)
        cards_meta[name] = card_meta

        cards_meta[name] = card_meta  # 仍記錄
        if card_meta["is_support"]:
            print(f"[SKIP] {name} is_support=true → not embedding this card.")
        else:
            all_pos.extend(pos_items)
            all_neg.extend(neg_items)
            # stats
        cp = Counter([r["lang"] for r in pos_items if r["field"] == "pos"])
        cn = Counter([r["lang"] for r in neg_items if r["field"] == "neg"])
        per_card_stats[name] = {
            "is_support": card_meta["is_support"],
            "desc_cnt": sum(1 for r in pos_items if r["field"] == "desc"),
            "args_cnt": sum(1 for r in pos_items if r["field"] == "args"),
            "tags_cnt": sum(1 for r in pos_items if r["field"] == "tags"),
            "pos_cnt": sum(1 for r in pos_items if r["field"] == "pos"),
            "neg_cnt": sum(1 for r in neg_items if r["field"] == "neg"),
            "pos_en": cp.get("en", 0), "pos_zh": cp.get("zh", 0),
            "neg_en": cn.get("en", 0), "neg_zh": cn.get("zh", 0),
        }

    # Logs: flatten summary
    print("\n=== FLATTEN SUMMARY ===")
    print(f"Total pos items: {len(all_pos)} | Total neg items: {len(all_neg)}")
    for name, s in per_card_stats.items():
        tag = "support" if s["is_support"] else "user-facing"
        print(f"- {name} [{tag}]  pos={s['pos_cnt']} (en={s['pos_en']}, zh={s['pos_zh']})  "
              f"neg={s['neg_cnt']} (en={s['neg_en']}, zh={s['neg_zh']})  "
              f"desc={s['desc_cnt']} args={s['args_cnt']} tags={s['tags_cnt']}")

    # Prepare texts & ids
    def build_sidecar_rows(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        texts, ids, items = [], [], []
        next_id = 0
        for r in rows:
            # stable hash could be used as ID; here we just assign int ids in order
            texts.append(r["text"])
            ids.append(next_id)
            items.append({
                "id": next_id,
                "item_id": r["item_id"],
                "card": r["card"],
                "field": r["field"],   # desc | pos | neg | args | tags
                "lang": r["lang"],
                "weight": r["weight"],
                "is_support": r["is_support"],
                "version": r["version"],
                "text_hash": sha1(r["text"]),
                "text_preview": (r["text"][:160] + ("..." if len(r["text"]) > 160 else "")),
            })
            next_id += 1
        return texts, ids, items

    pos_texts, pos_ids, pos_items = build_sidecar_rows(all_pos)
    neg_texts, neg_ids, neg_items = build_sidecar_rows(all_neg)

    # Embed + normalize
    print("\n=== EMBEDDING POSITIVES ===")
    pos_vecs = embed_texts(pos_texts, model=args.model,
                           batch_size=args.batch_size, log_prefix="POS")
    print("\n=== EMBEDDING NEGATIVES ===")
    neg_vecs = embed_texts(neg_texts, model=args.model,
                           batch_size=args.batch_size, log_prefix="NEG")

    # Build FAISS (IndexFlatIP + IDMap2)
    print("\n=== BUILD FAISS ===")
    pos_ids_np = np.array(pos_ids, dtype="int64")
    neg_ids_np = np.array(neg_ids, dtype="int64")
    pos_index = build_faiss_ip(pos_vecs, pos_ids_np)
    neg_index = build_faiss_ip(neg_vecs, neg_ids_np)
    print(f"[FAISS] pos: added {pos_vecs.shape[0]} vectors, dim={pos_vecs.shape[1]}, type={type(pos_index.index) if hasattr(pos_index,'index') else type(pos_index)}")
    print(f"[FAISS] neg: added {neg_vecs.shape[0]} vectors, dim={neg_vecs.shape[1]}, type={type(neg_index.index) if hasattr(neg_index,'index') else type(neg_index)}")

    # Persist
    outdir = args.outdir
    ensure_dir(outdir)
    faiss.write_index(pos_index, os.path.join(outdir, "pos.index.faiss"))
    faiss.write_index(neg_index, os.path.join(outdir, "neg.index.faiss"))
    write_jsonl(os.path.join(outdir, "pos.items.jsonl"), pos_items)
    write_jsonl(os.path.join(outdir, "neg.items.jsonl"), neg_items)
    save_json(os.path.join(outdir, "meta.cards.json"), cards_meta)

    # Build info
    build_info = {
        "model": args.model,
        "built_at": now_iso(),
        "cards_version": args.cards_version,
        "cards_count": len(paths),
        "pos_items": len(pos_items),
        "neg_items": len(neg_items),
        "embedding_dim": int(pos_vecs.shape[1]) if pos_vecs.size else None,
        "normalize": True,
        # cosine via IP on unit vectors
        "index_type": "IndexIDMap2(IndexFlatIP)",
    }
    save_json(os.path.join(outdir, "build_info.json"), build_info)

    # Sanity sample logs
    def sample_items(items, label, k=3):
        print(f"\n[SAMPLE] {label}:")
        for row in items[:k]:
            print(
                f"  id={row['id']} | {row['item_id']} | {row['text_preview']}")

    sample_items(pos_items, "first few POS")
    sample_items(neg_items, "first few NEG")

    print("\n=== DONE ===")
    print(f"Saved to: {os.path.abspath(outdir)}")
    print("Artifacts:")
    for fname in ["pos.index.faiss", "pos.items.jsonl", "neg.index.faiss", "neg.items.jsonl", "meta.cards.json", "build_info.json"]:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
