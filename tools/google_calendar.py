"""
Google Calendar and Tasks tools for the unified LangChain agent.

This module exposes two functions, ``create_calendar_event`` and
``create_task``, that encapsulate the logic for interacting with
Google Calendar and Google Tasks.  Both functions are decorated with
``@tool`` from ``langchain.tools`` so that they can be registered as
callable tools in a LangChain agent.  They can also be called
directly like ordinary Python functions without going through an
agent.

Each function accepts structured arguments rather than a single
``params`` dictionary.  This makes the intent of each parameter
explicit and allows LangChain to infer the schema automatically.  The
original class‑based implementations in ``GoogleCalendarTool`` and
``GoogleTasksTool`` have been converted into functional form here,
retaining the same behaviour and error handling.

Note: These functions expect that a valid Google API credential can
be obtained via ``get_credentials``.  When running on an environment
without Google API access, they will return a dictionary indicating
failure.  Replace the body of each function with mocks or stubs as
needed during development or testing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, time as time_class
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any
from langchain.tools import tool
from googleapiclient.discovery import build  # type: ignore
from utils.google_auth import get_credentials  # type: ignore


@tool
def google_calendar_create_event(
    title: str,
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    time: Optional[str] = None,
    location: str = "",
) -> Dict[str, Any]:
    """Create an event in the user's primary Google Calendar.

    Args:
        title: The event title.  If not provided, defaults to "Untitled Event".
        year: The year component of the event date.  If ``None``, uses
            today's year in the user's timezone.
        month: The month component of the event date.  If ``None``, uses
            today's month in the user's timezone.
        day: The day component of the event date.  If ``None``, uses
            today's day in the user's timezone.
        time: An optional HH:MM string representing the start time of
            the event (24‑hour clock).  If provided, the event will be
            one hour long.  If omitted, an all‑day event will be created.
        location: The event location.  Currently unused in the API request.

    Returns:
        A dictionary with at least a ``status`` field.  ``status`` is
        ``"success"`` if the event was created successfully.  When
        successful, the dictionary may also include a ``title`` field.
        If an error occurs, returns ``{"status": "error", "message": "..."}``.
    """
    # Guard against missing dependencies.
    if get_credentials is None or build is None:
        return {"status": "error", "message": "Google API libraries are not available."}

    # Initialize the service.
    try:
        creds = get_credentials()
        service = build("calendar", "v3", credentials=creds)
    except Exception as e:
        return {"status": "error", "message": f"Error initializing Google Calendar service: {e}"}

    # Determine the user's timezone (adjust for your app as needed).
    user_timezone = ZoneInfo("America/Los_Angeles")
    now = datetime.now(user_timezone)

    # Use provided date components or fall back to today's date.
    year = year or now.year
    month = month or now.month
    day = day or now.day
    title = title or "Untitled Event"

    # Construct the event body.
    if time:
        try:
            time_parts = time.split(":")
            hour = int(time_parts[0])
            minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            second = int(time_parts[2]) if len(time_parts) > 2 else 0
            start_time = now.replace(
                year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=0
            )
            end_time = start_time + timedelta(hours=1)
            event_body = {
                "summary": title,
                "start": {"dateTime": start_time.isoformat(), "timeZone": str(user_timezone)},
                "end": {"dateTime": end_time.isoformat(), "timeZone": str(user_timezone)},
            }
        except Exception:
            return {"status": "error", "message": f"Could not parse time '{time}'."}
    else:
        start_date = f"{year:04d}-{month:02d}-{day:02d}"
        event_body = {
            "summary": title,
            "start": {"date": start_date},
            "end": {"date": start_date},
        }

    # Call the API.
    try:
        service.events().insert(calendarId="primary", body=event_body).execute()
        return {"status": "success", "title": title}
    except Exception as e:
        return {"status": "error", "message": f"API Error: {e}"}


@tool
def google_tasks_create_task(
    title: str,
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    time: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a task in the user's default Google Tasks list.

    Args:
        title: The task title.  Defaults to "Untitled Task" if blank.
        year/month/day: Due date components.  Any missing component means no due date.
        time: Optional HH:MM string for a specific due time; if omitted, an all‑day task is created.

    Returns:
        ``{"status": "success", "title": title}`` on success, or
        ``{"status": "error", "message": "..."}`` on failure.
    """
    if get_credentials is None or build is None:
        return {"status": "error", "message": "Google API libraries are not available."}

    try:
        creds = get_credentials()
        service = build("tasks", "v1", credentials=creds)
    except Exception as e:
        return {"status": "error", "message": f"Error initializing Google Tasks service: {e}"}

    user_timezone = ZoneInfo("America/Los_Angeles")
    title = title or "Untitled Task"
    task_body: Dict[str, Any] = {"title": title}

    if year and month and day:
        try:
            if time:
                t = time_class.fromisoformat(time)
                due_datetime_local = datetime(
                    year, month, day, t.hour, t.minute, t.second if len(
                        time.split(":")) > 2 else 0,
                    tzinfo=user_timezone
                )
                task_body["due"] = due_datetime_local.astimezone(ZoneInfo("UTC")).strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z"
                )
            else:
                task_body["due"] = f"{year:04d}-{month:02d}-{day:02d}T00:00:00.000Z"
        except Exception as e:
            return {"status": "error", "message": f"Could not parse date/time: {e}"}

    try:
        service.tasks().insert(tasklist="@default", body=task_body).execute()
        return {"status": "success", "title": title}
    except Exception as e:
        return {"status": "error", "message": f"API Error: {e}"}
