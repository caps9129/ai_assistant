from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from googleapiclient.discovery import build
from .google_auth import get_credentials


class GoogleCalendarTool:
    def __init__(self):
        """
        Initializes the Google Calendar API service.
        """
        try:
            creds = get_credentials()
            self.service = build("calendar", "v3", credentials=creds)
            # We need the user's timezone to create correct timestamps
            self.user_timezone = ZoneInfo(
                "America/Los_Angeles")  # For Corvallis, Oregon
            print("✅ Google Calendar Tool initialized.")
        except Exception as e:
            print(f"❌ Error initializing Google Calendar Tool: {e}")
            self.service = None

    def create_event(self, params: dict) -> str:
        """
        Creates a new event in the user's primary calendar.

        Args:
            params (dict): A dictionary from the LLM with keys like
                           'title', 'year', 'month', 'day', 'time'.
        """
        if not self.service:
            return "Error: Google Calendar service is not available."

        title = params.get("title", "Untitled Event")
        now = datetime.now(self.user_timezone)

        # Use extracted parameters, falling back to today's date
        year = params.get("year") or now.year
        month = params.get("month") or now.month
        day = params.get("day") or now.day

        if params.get("time"):
            # If time is specified, create a 1-hour event
            try:
                hour, minute, second = map(int, params["time"].split(':'))
                start_time = now.replace(
                    year=year, month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
                end_time = start_time + timedelta(hours=1)

                start_iso = start_time.isoformat()
                end_iso = end_time.isoformat()

                event_body = {
                    'summary': title,
                    'start': {'dateTime': start_iso, 'timeZone': str(self.user_timezone)},
                    'end': {'dateTime': end_iso, 'timeZone': str(self.user_timezone)},
                }
            except ValueError:
                return f"Sorry, I couldn't understand the time '{params['time']}'."
        else:
            # If no time is specified, create an all-day event
            start_date = f"{year:04d}-{month:02d}-{day:02d}"
            event_body = {
                'summary': title,
                'start': {'date': start_date},
                'end': {'date': start_date},
            }

        try:
            print(f"--> Creating Google Calendar event: {event_body}")
            self.service.events().insert(calendarId='primary', body=event_body).execute()
            return {"status": "success", "title": title}
        except Exception as e:
            print(f"❌ Error creating calendar event: {e}")
            # --- MODIFIED: Return a structured error dictionary ---
            return {"status": "error", "message": f"API Error: {str(e)}"}
