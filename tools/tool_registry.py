from typing import List, Optional
from tools.google_calendar import google_calendar_create_event, google_tasks_create_task
from tools.google_maps import google_maps_search_places, google_maps_directions, google_maps_get_place_details

ALL_TOOLS = [
    google_calendar_create_event,
    google_tasks_create_task,
    google_maps_search_places,
    google_maps_directions,
    google_maps_get_place_details
]


def select_tools(*, provider: str, domain: Optional[str] = None, complexity: str = "simple"):
    out = []
    for t in ALL_TOOLS:
        md = getattr(t, "metadata", {})
        if md.get("provider") != provider:
            continue
        if domain and md.get("domain") != domain:
            continue
        # simple 任務也允許撿到 complexity="simple" 的工具；complex 任務可以全撿
        if complexity == "simple" and md.get("complexity") not in ("simple", None):
            continue
        out.append(t)
    return out
