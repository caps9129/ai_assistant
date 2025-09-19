"""
Google Maps tools for the unified LangChain agent.

This module exposes four functions—``find_nearby_places``,
``get_directions``, ``get_place_details`` and ``geocode_address``—which
wrap Google Maps API operations.  Each function is decorated with
``@tool`` from ``langchain.tools`` so they can be registered with a
LangChain agent.  They can also be invoked directly like ordinary
Python functions.

The implementations here mirror the behaviour of the original
``GoogleMapsTool`` class in the repository, including filtering and
sorting logic.  If the Google Maps client library is unavailable or
the API key is missing, the functions return a structured error
dictionary instead of raising exceptions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any

from langchain.tools import tool
import googlemaps
from config import Maps_API_KEY
from utils.location import LocationManager  # type: ignore


def _init_gmaps_client() -> Optional[Any]:
    """Initialise and return a Google Maps client or None if unavailable."""
    if googlemaps is None or not Maps_API_KEY:
        return None
    try:
        return googlemaps.Client(key=Maps_API_KEY)
    except Exception:
        return None


def _get_location_manager() -> Optional[Any]:
    """Initialise and return a LocationManager or None if unavailable."""
    if LocationManager is None:
        return None
    try:
        return LocationManager()
    except Exception:
        return None


@tool
def google_maps_search_places(
    query: str,
    min_rating: float = 0.0,
    min_total_ratings: int = 0,
    max_price_level: Optional[int] = None,
    open_now: bool = False,
    rank_by: str = "prominence",
    radius_km: float = 5.0,
    max_results: int = 5,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """Search for nearby points of interest around the user's location.

    This function wraps the Google Maps Places API and applies optional
    filters, such as minimum rating, minimum number of reviews and
    maximum price level.  Results are sorted by rating and number of
    reviews, then truncated to ``max_results`` entries.
    """
    gmaps = _init_gmaps_client()
    loc_manager = _get_location_manager()
    if gmaps is None or loc_manager is None:
        return [{"error": "Google Maps client or location manager is unavailable."}]

    loc_info = loc_manager.get_location_info() if hasattr(
        loc_manager, "get_location_info") else None
    if not loc_info or 'lat' not in loc_info or 'lon' not in loc_info:
        return [{"error": "Could not determine current location."}]

    current_location = (loc_info['lat'], loc_info['lon'])

    # Construct parameters for the Places API.
    params: Dict[str, Any] = {
        "location": current_location,
        "keyword": query,
        "rank_by": rank_by,
        "language": language,
    }
    if open_now:
        params['open_now'] = True
    if max_price_level is not None and 1 <= max_price_level <= 4:
        params['max_price'] = max_price_level
    # When ranking by distance, the API forbids radius; otherwise, include radius.
    if rank_by != "distance":
        params['radius'] = radius_km * 1000

    try:
        places_result = gmaps.places_nearby(**params)
    except Exception as e:
        return [{"error": f"API Error: {e}"}]

    results = places_result.get('results', [])
    filtered: List[Dict[str, Any]] = []
    for place in results:
        rating = place.get('rating', 0.0)
        total_ratings = place.get('user_ratings_total', 0)
        if rating >= min_rating and total_ratings >= min_total_ratings:
            filtered.append({
                "name": place.get('name'),
                "rating": rating,
                "total_ratings": total_ratings,
                "price_level": place.get('price_level', 'N/A'),
                "address": place.get('vicinity'),
                "is_open_now": place.get('opening_hours', {}).get('open_now', 'Unknown'),
            })

    sorted_places = sorted(
        filtered,
        key=lambda p: (p.get('rating', 0), p.get('total_ratings', 0)),
        reverse=True
    )
    return sorted_places[:max_results]


@tool
def google_maps_directions(
    destination: str,
    origin: Optional[str] = None,
    mode: str = "driving",
    language: str = "en",
) -> Dict[str, Any]:
    """Provide navigation directions from ``origin`` to ``destination``.

    If ``origin`` is ``None``, the current user location is used.  The
    response includes start and end addresses, distance, travel time with
    traffic and the route summary.  When the API call fails, a
    dictionary with an ``error`` key is returned.
    """
    gmaps = _init_gmaps_client()
    loc_manager = _get_location_manager()
    if gmaps is None or loc_manager is None:
        return {"error": "Google Maps client or location manager is unavailable."}

    if not origin:
        loc_info = loc_manager.get_location_info() if hasattr(
            loc_manager, "get_location_info") else None
        if not loc_info or 'lat' not in loc_info or 'lon' not in loc_info:
            return {"error": "Could not determine current location for origin."}
        origin = f"{loc_info['lat']},{loc_info['lon']}"

    try:
        directions_result = gmaps.directions(
            origin,
            destination,
            mode=mode,
            departure_time=datetime.now(),
            language=language,
        )
    except Exception as e:
        return {"error": f"API Error: {e}"}

    if not directions_result:
        return {"error": f"Could not find directions to {destination}."}

    leg = directions_result[0]['legs'][0]
    return {
        "origin": leg['start_address'],
        "destination": leg['end_address'],
        "distance": leg['distance']['text'],
        "duration_with_traffic": leg['duration']['text'],
        "summary": directions_result[0]['summary'],
    }


@tool
def google_maps_get_place_details(
    query: str,
    fields: List[str],
    language: str = "en",
) -> Dict[str, Any]:
    """Retrieve specific details about a place identified by a query string.

    The function first uses ``find_place`` to resolve the place ID, then
    calls ``place`` to fetch the requested fields.  If a place cannot
    be found or an error occurs, a dictionary with an ``error`` key is
    returned.
    """
    gmaps = _init_gmaps_client()
    if gmaps is None:
        return {"error": "Google Maps client is unavailable."}

    try:
        find_result = gmaps.find_place(query, 'textquery', fields=[
                                       'place_id', 'name'], language=language)
        candidates = find_result.get('candidates', [])
        if not candidates:
            return {"error": f"Could not find a place matching '{query}'."}
        place_id = candidates[0]['place_id']
        place_name = candidates[0]['name']
        details_result = gmaps.place(
            place_id, fields=fields, language=language)
        result = details_result.get('result', {})
        result['name'] = place_name
        return result
    except Exception as e:
        return {"error": f"API Error: {e}"}


@tool
def google_maps_geocode_address(
    address: str,
    language: str = "en",
) -> Dict[str, Any]:
    """Convert an address or landmark name into geographic coordinates."""
    gmaps = _init_gmaps_client()
    if gmaps is None:
        return {"error": "Google Maps client is unavailable."}

    try:
        geocode_result = gmaps.geocode(address, language=language)
        if not geocode_result:
            return {"error": f"Could not geocode '{address}'."}
        return {
            "formatted_address": geocode_result[0]['formatted_address'],
            "location": geocode_result[0]['geometry']['location'],
        }
    except Exception as e:
        return {"error": f"API Error: {e}"}
