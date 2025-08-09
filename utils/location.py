# utils/location.py

import requests
from datetime import datetime
from zoneinfo import ZoneInfo


class LocationManager:
    """
    Determines the user's location and timezone using a free IP geolocation API.
    """

    def __init__(self):
        # This API is simple and requires no key for moderate use
        self.api_url = "http://ip-api.com/json/"

    def get_location_info(self) -> dict | None:
        """
        Fetches location information based on the user's public IP address.
        """
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()

            if data.get("status") == "success":
                return {
                    "city": data.get("city"),
                    "region": data.get("regionName"),
                    "country": data.get("country"),
                    "lat": data.get("lat"),
                    "lon": data.get("lon"),
                    # e.g., "America/Los_Angeles"
                    "timezone": data.get("timezone")
                }
            else:
                print(
                    f"Geolocation API returned an error: {data.get('message')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error calling Geolocation API: {e}")
            return None

    def get_current_timezone_name(self) -> str | None:
        """
        High-level method to get the IANA timezone name for the current location.
        """
        info = self.get_location_info()
        return info.get("timezone") if info else None


# --- Standalone Test Block ---
if __name__ == '__main__':
    print("--- Testing Free LocationManager ---")
    location_manager = LocationManager()

    timezone_name = location_manager.get_current_timezone_name()

    if timezone_name:
        print(f"Successfully determined timezone: {timezone_name}")

        # Demonstrate using the timezone to get the correct local time
        try:
            tz = ZoneInfo(timezone_name)
            local_time = datetime.now(tz)
            print(
                f"Current local time is: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        except Exception as e:
            print(f"Could not process timezone: {e}")
    else:
        print("Failed to determine timezone.")
