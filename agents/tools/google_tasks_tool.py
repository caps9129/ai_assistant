from datetime import datetime, time
from zoneinfo import ZoneInfo
from googleapiclient.discovery import build
from .google_auth import get_credentials


class GoogleTasksTool:
    def __init__(self):
        # ... (init method is the same)
        try:
            creds = get_credentials()
            self.service = build("tasks", "v1", credentials=creds)
            self.user_timezone = ZoneInfo("America/Los_Angeles")
            print("✅ Google Tasks Tool initialized.")
        except Exception as e:
            print(f"❌ Error initializing Google Tasks Tool: {e}")
            self.service = None

    def create_task(self, params: dict) -> str:
        """
        Creates a new task, correctly handling due date and time.
        """
        if not self.service:
            return "Error: Google Tasks service is not available."

        title = params.get("title", "Untitled Task")
        task_body = {'title': title}

        # --- MODIFIED: Improved date and time handling ---
        year = params.get("year")
        month = params.get("month")
        day = params.get("day")
        time_str = params.get("time")  # e.g., "16:00:00"

        if year and month and day:
            try:
                if time_str:
                    # If time is provided, create a specific, timezone-aware datetime
                    task_time = time.fromisoformat(time_str)
                    due_datetime_local = datetime(
                        year, month, day,
                        hour=task_time.hour,
                        minute=task_time.minute,
                        second=task_time.second,
                        tzinfo=self.user_timezone
                    )
                    # Convert to the required UTC RFC3339 timestamp
                    task_body['due'] = due_datetime_local.astimezone(
                        ZoneInfo("UTC")).strftime('%Y-%m-%dT%H:%M:%S.000Z')
                else:
                    # --- FIX for ALL-DAY TASK ---
                    # If no time is provided, create an "all-day" task.
                    # The standard is to represent this as midnight UTC on the due date.
                    # This avoids the confusing "end of day" timezone rollover.
                    due_date_utc = f"{year:04d}-{month:02d}-{day:02d}T00:00:00.000Z"
                    task_body['due'] = due_date_utc

            except ValueError as e:
                print(f"⚠️  Could not parse date/time for task: {e}")
        # --- END MODIFICATION ---

        try:
            print(f"--> Creating Google Task: {task_body}")
            self.service.tasks().insert(tasklist='@default', body=task_body).execute()
            return {"status": "success", "title": title}
        except Exception as e:
            print(f"❌ Error creating task: {e}")
            return {"status": "error", "message": f"API Error: {str(e)}"}
