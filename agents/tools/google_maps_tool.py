import googlemaps
from datetime import datetime

# Import the API key from your central config
from config import Maps_API_KEY
# Import the location manager to get the user's current location
from utils.location import LocationManager


# This list of dictionaries should be a class attribute or defined
# in the __init__ method of your GoogleMapsHandler class.

Maps_TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "find_nearby_places",
            "description": "Finds points of interest. IMPORTANT: It automatically searches near the user's current location unless a different city or area is mentioned in the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The type of place to search for, e.g., 'coffee', 'cheap restaurants', 'sushi'."
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "The minimum required star rating for the places, from 1.0 to 5.0."
                    },
                    "min_total_ratings": {
                        "type": "integer",
                        "description": "The minimum number of user reviews a place must have to be included."
                    },
                    "max_price_level": {
                        "type": "integer",
                        "description": "The maximum price level, from 1 (cheap) to 4 (expensive). e.g., for 'cheap food', use 1 or 2."
                    },
                    "open_now": {
                        "type": "boolean",
                        "description": "Set to true to only search for places that are currently open."
                    },
                    "rank_by": {
                        "type": "string",
                        "enum": ["prominence", "distance"],
                        "description": "How to rank results. 'prominence' is a mix of popularity and relevance. 'distance' ranks by closest first."
                    },
                    "language": {
                        "type": "string",
                        "description": "IETF BCP 47 language tag specifying the desired language for the API response (e.g. \"zh-TW\", \"en-US\")."
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_directions",
            # --- MODIFIED DESCRIPTION ---
            "description": "Provides navigation directions. The origin defaults to the user's current location if not explicitly specified by the user.",
            # --- END MODIFICATION ---
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The user's desired destination address or landmark."
                    },
                    "origin": {  # Origin is now clearly optional
                        "type": "string",
                        "description": "The starting point for the directions. Only use if the user explicitly mentions a starting point other than their current location."
                    },
                    "mode": {  # No change needed here
                        "type": "string",
                        "enum": ["driving", "walking", "bicycling", "transit"],
                        "description": "The mode of travel. Defaults to driving."
                    },
                    "language": {
                        "type": "string",
                        "description": "IETF BCP 47 language tag specifying the desired language for the API response (e.g. \"zh-TW\", \"en-US\")."
                    }
                },
                "required": ["destination"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_place_details",
            "description": "Retrieves specific details about a known place, such as its operating hours or phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The name of the specific place to get details for, e.g., 'downtown Safeway' or 'Corvallis library'."
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of specific details to fetch. Supported values are: 'opening_hours' for questions about when a place is open or closes, 'business_status' to check if it's operational, and 'formatted_phone_number' for contact info."
                    },
                    "language": {
                        "type": "string",
                        "description": "IETF BCP 47 language tag specifying the desired language for the API response (e.g. \"zh-TW\", \"en-US\")."
                    }
                },
                "required": ["query", "fields"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "geocode_address",
            "description": "Finds the specific address or geographic coordinates for a landmark or place name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "The name of the place or the address to geocode, e.g., 'Oregon State University'."
                    },
                    "language": {
                        "type": "string",
                        "description": "IETF BCP 47 language tag specifying the desired language for the API response (e.g. \"zh-TW\", \"en-US\")."
                    }
                },
                "required": ["address"],
            },
        }
    }
]


class GoogleMapsTool:
    def __init__(self):
        """
        Initializes the Google Maps client and other necessary utilities.
        """
        if not Maps_API_KEY:
            raise ValueError(
                "Maps_API_KEY is not set in your config/.env file.")

        self.gmaps = googlemaps.Client(key=Maps_API_KEY)
        self.location_manager = LocationManager()
        print("✅ Google Maps Tool initialized.")

    def find_nearby_places(
        self,
        query: str,
        min_rating: float = 0.0,
        min_total_ratings: int = 0,
        max_price_level: int = None,
        open_now: bool = False,
        rank_by: str = "prominence",
        radius_km: float = 5.0,
        max_results: int = 5,
        language: str = "en"

    ) -> list[dict]:
        """
        Finds nearby places with optional filtering and ranking.
        """
        print(f"TOOL: Searching for '{query}' with rank_by='{rank_by}'...")

        location_info = self.location_manager.get_location_info()
        if not location_info:
            return [{"error": "Could not determine current location."}]
        current_location = (location_info['lat'], location_info['lon'])

        # --- MODIFIED: Conditionally add radius based on rank_by ---
        places_params = {
            "location": current_location,
            "keyword": query,
            "rank_by": rank_by,
            "language": language
        }
        if open_now:
            places_params['open_now'] = True
        if max_price_level is not None and 1 <= max_price_level <= 4:
            places_params['max_price'] = max_price_level

        if rank_by != "distance":
            places_params['radius'] = radius_km * 1000
        # --- END MODIFICATION ---

        places_result = self.gmaps.places_nearby(**places_params)

        # ... (The rest of the function for filtering and sorting remains the same)
        results = places_result.get('results', [])

        filtered_places = []
        for place in results:

            if (place.get('rating', 0) >= min_rating and
                    place.get('user_ratings_total', 0) >= min_total_ratings):
                filtered_places.append({
                    "name": place.get('name'),
                    "rating": place.get('rating', 0),
                    "total_ratings": place.get('user_ratings_total', 0),
                    "price_level": place.get('price_level', 'N/A'),
                    "address": place.get('vicinity'),
                    "is_open_now": place.get('opening_hours', {}).get('open_now', 'Unknown')
                })

        sorted_places = sorted(
            filtered_places,
            key=lambda p: (p.get('rating', 0), p.get('total_ratings', 0)),
            reverse=True
        )

        return sorted_places[:max_results]

    def get_directions(
        self,
        destination: str,
        origin: str = None,  # If None, use current location
        mode: str = "driving",
        language: str = "en"
    ) -> dict:
        """
        [Tool for Q4, Q5, Q7]
        Provides directions, including travel time with traffic.
        """
        print(f"TOOL: Getting directions to '{destination}'...")

        if not origin:
            location_info = self.location_manager.get_location_info()
            if not location_info:
                return {"error": "Could not determine current location for origin."}
            origin = f"{location_info['lat']},{location_info['lon']}"

        try:
            directions_result = self.gmaps.directions(
                origin,
                destination,
                mode=mode,
                departure_time=datetime.now(),  # Use current time for traffic estimation
                language=language
            )

            if not directions_result:
                return {"error": f"Could not find directions to {destination}."}

            leg = directions_result[0]['legs'][0]
            return {
                "origin": leg['start_address'],
                "destination": leg['end_address'],
                "distance": leg['distance']['text'],
                "duration_with_traffic": leg['duration']['text'],
                "summary": directions_result[0]['summary']  # e.g., "I-5 N"
            }
        except Exception as e:
            return {"error": str(e)}

    def get_place_details(
        self,
        query: str,
        fields: list[str],
        language: str = "en"
    ) -> dict:
        """
        [Tool for Q6, Q10]
        Finds a specific place and retrieves detailed information.
        """
        print(f"TOOL: Getting details for '{query}'...")
        try:
            # First, find the place to get its Place ID
            find_place_result = self.gmaps.find_place(
                query,
                'textquery',
                fields=['place_id', 'name'],
                language=language
            )
            candidates = find_place_result.get('candidates', [])
            if not candidates:
                return {"error": f"Could not find a place matching '{query}'."}

            place_id = candidates[0]['place_id']
            place_name = candidates[0]['name']

            # Now, get the specific details for that Place ID
            details_result = self.gmaps.place(
                place_id, fields=fields, language='zh-TW')
            result = details_result.get('result', {})
            result['name'] = place_name  # Add the name for context
            return result
        except Exception as e:
            return {"error": str(e)}

    def geocode_address(
        self,
        address: str,
        language: str = "en"
    ) -> dict:
        """
        [Tool for Q8]
        Converts an address or landmark into a formatted address and coordinates.
        """
        print(f"TOOL: Geocoding '{address}'...")
        try:
            geocode_result = self.gmaps.geocode(address, language=language)
            if not geocode_result:
                return {"error": f"Could not geocode '{address}'."}

            return {
                "formatted_address": geocode_result[0]['formatted_address'],
                # {lat, lng}
                "location": geocode_result[0]['geometry']['location']
            }
        except Exception as e:
            return {"error": str(e)}


# --- Standalone Test Block ---
if __name__ == '__main__':
    # To run this test: python -m agents.tools.Maps_tool
    import os
    import sys
    import json
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    print("\n" + "="*50)
    print("Running tests for GoogleMapsTool...")
    print("="*50 + "\n")

    tool = GoogleMapsTool()

    # Test 1: Find nearby places (Q1, Q2, Q3, Q9)
    print("--- Testing find_nearby_places ---")
    places = tool.find_nearby_places(
        query="coffee shop", min_rating=4.0, max_results=3)
    print(json.dumps(places, indent=2, ensure_ascii=False))

    # Test 2: Get directions (Q4, Q5, Q7)
    print("\n--- Testing get_directions ---")
    directions = tool.get_directions(destination="Portland, OR")
    print(json.dumps(directions, indent=2, ensure_ascii=False))

    # Test 3: Get place details (Q6, Q10)
    print("\n--- Testing get_place_details ---")
    details = tool.get_place_details(query="Starbucks near downtown Corvallis", fields=[
                                     "name", "formatted_phone_number", "opening_hours"])
    print(json.dumps(details, indent=2, ensure_ascii=False))

    # Test 4: Geocode address (Q8)
    print("\n--- Testing geocode_address ---")
    location = tool.geocode_address("Oregon State University")
    print(json.dumps(location, indent=2, ensure_ascii=False))
