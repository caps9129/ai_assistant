# FuseDecision Policies — Concise Guide

This guide summarizes how `NodeFuseDecision` fuses **MainRouter needs** and **QueryRouter (QR) candidates** into a final tool list.

## Inputs
- **route**: `SIMPLE_TOOL | COMPLEX_TOOL | GENERAL_CHAT | EXIT`
- **needs**: Dict or List of tool names (truthy entries mean “needed”)
- **qr_candidates**: List of `{tool: str, score: float, ...}`

## Core Policies (config fields)
- **max_tools**: Upper bound of tools returned.
- **require_user_facing**: Keep only tools that pass `is_user_facing(tool)` (if provided).
- **min_qr_score**: Drop QR candidates with score < threshold.
- **adopt_qr_when_needs_empty**: If no needs, allow using QR-only suggestions.
- **order_policy**: How to merge needs and QR:
  - `needs_first` (default): `[needs...] + [qr_by_score...]`
  - `qr_first`: `[qr_by_score...] + [needs...]`
  - `merge_by_score`: Interleave by score with **needs treated as +∞**.
- **prefer_exact_needs**: After merging, keep original **needs order** first, then fill with others.
- **collapse_duplicates**: De-duplicate while preserving first occurrence.
- **simple_max_primary**: If `route=SIMPLE_TOOL`, hard cap returned tools to this value (usually 1).
- **complex_min_primary**: If `route=COMPLEX_TOOL`, try to ensure at least this many tools (bounded by `max_tools`).
- **allowed_capabilities**: Optional allowlist; drop tools not in it early.

## Sanitization & Filtering
1) **Sanitize needs**: accept dict (`{name: bool}`) or list (`[name...]`); produce list of tool names.  
2) **Sanitize QR**: accept `qr_candidates` or legacy `topk`.  
3) **Score filter**: keep only QR items with `score >= min_qr_score`.  
4) **Allowlist/User-facing filter**: drop tools not allowed or not user-facing.  
5) **Best-per-tool**: if a tool appears multiple times, keep the **highest score**.

## Merge Algorithm (high level)
1) Build **qr_order** by descending score.  
2) Merge by `order_policy` (see above).  
3) If `needs` is empty **and** `adopt_qr_when_needs_empty=True`, accept QR-only list.  
4) If `collapse_duplicates=True`, remove duplicates (stable).  
5) If `prefer_exact_needs=True`, reorder to put **needs** first (stable), then others.  
6) Soft-cap to `max_tools`.  
7) Enforce **route cardinality**:
   - `SIMPLE_TOOL` → cap to `simple_max_primary` (often 1).
   - `COMPLEX_TOOL` → keep up to `max_tools`, try to have ≥ `complex_min_primary` when available.
   - `GENERAL_CHAT | EXIT` → return empty list.

## Quick Examples
- **Simple, one tool**: needs=`['google_maps_directions']`, route=`SIMPLE_TOOL` → returns `['google_maps_directions']` (capped to 1).  
- **Complex, combine two**: needs=`['google_maps_search_places','google_maps_directions']`, QR suggests the same + extras; with allowlist and `max_tools=3`, expect `['google_maps_search_places','google_maps_directions', (maybe) 'google_maps_get_place_details']`.  
- **No needs, use QR**: needs=`[]`, good `qr_candidates`, and `adopt_qr_when_needs_empty=True` → returns top-K QR tools after filters.

## Defaults (typical)
- `max_tools=3`, `require_user_facing=True`, `min_qr_score≈0.30–0.35`,  
  `order_policy='needs_first'`, `prefer_exact_needs=True`, `collapse_duplicates=True`,  
  `simple_max_primary=1`, `complex_min_primary=2`.

---

**Tip**: For production, document your chosen config (thresholds, allowlist, user-facing function) and add unit tests for each policy branch (e.g., empty needs, low-scored QR, duplicate tools, cardinality enforcement).
