# agents/simple_planner_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from agent_config import AGENT_REGISTRY
from agents.base import BaseAgentConfig, OpenAIChatAgentBase


# 可擴充：primary -> [supporting tools]
# 這裡僅內建 directions 需要 geocode 的對應關係；未來有新支援工具時在此擴充或由 registry 注入。
_PRIMARY_SUPPORT_RULES: Dict[str, List[Dict[str, str]]] = {
    "google_maps_directions": [
        {
            "name": "google_maps_geocode_address",
            "desc": "Supporting tool to disambiguate place strings into addresses for directions."
        }
    ]
}


class SimplePlannerAgent(OpenAIChatAgentBase):
    """
    SIMPLE 支援規劃器：
      - 僅在 routing 判定為 SIMPLE 且 cand_tools 指向單一步主工具時使用。
      - 從 cand_tools 中選 EXACT 一個 primary。
      - 視情況加入 0~1 個 supporting（僅當該 supporting 也在 cand_tools；若未提供則由本 Agent 自動補入常見支援工具）。
      - 產出:
          * 無 support → type="SIMPLE_TOOL"
          * 有 support → type="COMPLEX_TOOL"
      - 若資訊不足，回單句 clarification。
    """

    def __init__(self):
        cfg = AGENT_REGISTRY.get_config("SimplePlannerAgent")
        if not cfg:
            raise ValueError("Config for 'SimplePlannerAgent' not found.")

        base_cfg = BaseAgentConfig(
            id="SimplePlannerAgent",
            prompt_name=cfg.prompt_name,         # e.g. "simple_planner"
            model=cfg.model,
            temperature=getattr(cfg, "temperature", 0.0),
            max_tokens=getattr(cfg, "max_tokens", None),
            history_limit_pairs=getattr(cfg, "history_limit_pairs", 0),
            max_chars=getattr(cfg, "max_chars", 12000),
            output_mode="json",
        )
        super().__init__(base_cfg)
        print("✅ SimplePlannerAgent ready.")

    # ---------- Public API ----------
    def plan(
        self,
        user_text: str,
        cand_tools: List[Any],
        memory=None,
    ) -> Dict[str, Any]:
        # 1) 規範化候選工具
        norm_cand = self._normalize_cand_tools(cand_tools)
        # 2) 依 primary 自動補上常見支援工具（若尚未包含）
        norm_cand = self._auto_inject_supporting_tools(norm_cand)

        # 3) 呼叫 LLM（payload 直接給 dict；Base 會序列化成 JSON 字串）
        payload = {
            "user_text": user_text,
            "cand_tools": norm_cand
        }
        raw = self.process_request(payload, memory)

        # 4) 解析/正規化輸出
        if isinstance(raw, str):
            # 文字兜底：視為 clarification
            return {
                "type": "SIMPLE_TOOL",
                "steps": [],
                "clarification": raw.strip() or "Could you clarify your exact destination or target?",
                "confidence": 0.0,
                "raw": {"fallback": raw},
            }

        out = self._normalize_output(raw, norm_cand)
        out["raw"] = raw
        return out

    # ---------- Helpers ----------
    def _normalize_cand_tools(self, cand_tools: List[Any]) -> List[Dict[str, str]]:
        """接受 list[str] 或 list[dict]，統一回傳含至少 'name' 的 dict 清單。"""
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

    def _auto_inject_supporting_tools(self, cand_tools: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        若 cand_tools 中包含某些 primary，且尚未帶入對應支援工具，則自動補入（以便 Prompt 規則允許使用）。
        目前：directions -> geocode_address
        """
        names = {c["name"] for c in cand_tools if "name" in c}
        # 檢查是否有 directions（或其他在規則中列出的 primary）
        for primary_name, support_list in _PRIMARY_SUPPORT_RULES.items():
            if primary_name in names:
                # 將規則內的支援工具逐一補入（若尚未存在）
                for sup in support_list:
                    sup_name = sup.get("name")
                    if sup_name and sup_name not in names:
                        cand_tools.append(
                            {"name": sup_name, "desc": sup.get("desc", "")})
                        names.add(sup_name)
        return cand_tools

    def _normalize_output(self, raw: Any, cand_tools: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        - 僅保留 cand_tools 允許的 primary（最多 1 個）與 supporting（最多 1 個，且必須也在 cand_tools）。
        - 有 support → type="COMPLEX_TOOL"；無 support 且有 1 個 primary → type="SIMPLE_TOOL"。
        - 夾斷多餘步驟，並夾斷非允許工具。
        - 置信度限制在 [0.0, 1.0]。
        - 空結果時，提供簡短 clarification。
        """
        result = {
            "type": "SIMPLE_TOOL",      # 預設：無支援 → SIMPLE
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

        primary_kept: Optional[Dict[str, Any]] = None
        support_kept: Optional[Dict[str, Any]] = None

        if isinstance(steps_in, list):
            for s in steps_in:
                if not isinstance(s, dict):
                    continue
                tname = s.get("tool")
                use = s.get("use")
                if not isinstance(tname, str) or not isinstance(use, str):
                    continue
                tname = tname.strip()
                use_l = use.strip().lower()

                # 只保留在 cand_tools 內的工具
                if tname not in cand_set:
                    continue

                if use_l == "primary":
                    if primary_kept is None:
                        primary_kept = s
                elif use_l == "support":
                    if support_kept is None:
                        support_kept = s
                # 其他 use 值忽略

        # 組裝步驟（支援若存在，建議排在 primary 前或後視模型輸出；我們維持模型順序，若都存在、且支援目標為 p1，盡量置前）
        steps_out: List[Dict[str, Any]] = []
        has_support = support_kept is not None
        if has_support:
            # 嘗試確保 target_key 指向 primary key（若 primary 沒 key，就補 p1 並改 support 的 target_key）
            if primary_kept is None:
                # 沒 primary → 無效計畫；改為要求澄清
                support_kept = None
                has_support = False
            else:
                # 正規化 primary key
                pkey = primary_kept.get("key") or "p1"
                primary_kept["key"] = pkey
                # 正規化 support target_key
                if "target_key" not in support_kept or not isinstance(support_kept.get("target_key"), str):
                    support_kept["target_key"] = pkey
                steps_out.append(support_kept)

        if primary_kept is not None:
            if "key" not in primary_kept or not isinstance(primary_kept.get("key"), str):
                primary_kept["key"] = "p1"
            steps_out.append(primary_kept)

        # 型別與置信度
        enforced_type = "COMPLEX_TOOL" if has_support else (
            "SIMPLE_TOOL" if primary_kept else "SIMPLE_TOOL")
        if typ not in ("SIMPLE_TOOL", "COMPLEX_TOOL"):
            typ = enforced_type
        else:
            if typ == "SIMPLE_TOOL" and has_support:
                typ = "COMPLEX_TOOL"
            elif typ == "COMPLEX_TOOL" and not has_support and primary_kept:
                typ = "SIMPLE_TOOL"

        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        conf_f = 0.0 if conf_f < 0 else 1.0 if conf_f > 1.0 else conf_f

        # Clarification 處理
        clar_s = None
        if isinstance(clar, str) and clar.strip():
            clar_s = clar.strip()

        if not steps_out and clar_s is None:
            clar_s = "Which exact place should I navigate to? Please provide a name or address."

        result.update({
            "type": typ,
            "steps": steps_out,
            "clarification": clar_s,
            "confidence": conf_f,
        })
        return result


if __name__ == '__main__':
    # Minimal local test (non-streaming)
    import json
    from memory.manager import MemoryManager

    agent = SimplePlannerAgent()
    memory = MemoryManager()

    user_text = "What's the phone number for Din Tai Fung?"
    cand_tools = [
        {
            "name": "google_maps_get_place_details",
            "desc": "Fetch details (hours, phone, website, address, rating) for a single identified place by name or place_id."
        }
    ]

    user_text = "Take me there right now."
    cand_tools = [
        {
            "name": "google_maps_directions",
            "desc": "Route from origin (default current location) to destination."
        },
        {
            "name": "google_maps_geocode_address",
            "desc": "Supporting tool to disambiguate place strings into addresses for directions."
        }
    ]

    user_text = "Show 24/7 pharmacies around Shinjuku Station."
    cand_tools = [
        {
            "name": "google_maps_search_places",
            "desc": "Search for nearby places centered on the user's current location (cafes, gyms, supermarkets, etc.) with filters."
        }
    ]

    payload = {
        "user_text": user_text,
        "cand_tools": cand_tools
    }
    raw = agent.process_request(payload, memory)
    out = agent._normalize_output(raw, agent._auto_inject_supporting_tools(
        agent._normalize_cand_tools(cand_tools)))
    out["raw"] = raw
    print(json.dumps(out, ensure_ascii=False, indent=2))
