"""
Timezone helpers for CrisisIntel.
All timestamps in the database are stored in UTC.
Templates display them in IST (Asia/Kolkata).
"""
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def now_ist():
    """Current time in IST (aware)."""
    return datetime.now(IST)


def utc_now():
    """Current time in UTC (aware). Used for DB writes."""
    return datetime.now(timezone.utc)


def to_ist(value, fmt="%d-%b-%Y %I:%M %p"):
    """
    Convert a datetime or ISO string to an IST-formatted string.
    Accepts:
      - datetime (naive → assumed UTC)
      - ISO string like '2024-06-10 09:30:00'
      - None
    Returns a formatted IST string or '—' on failure.
    """
    if value is None or value == "":
        return "—"

    try:
        if isinstance(value, str):
            # SQLite format: "2024-06-10 09:30:00"
            dt = datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
        elif isinstance(value, datetime):
            dt = value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            return str(value)

        return dt.astimezone(IST).strftime(fmt)

    except Exception:
        return str(value)


def to_ist_short(value):
    """Short date only — DD-MMM-YYYY."""
    return to_ist(value, fmt="%d-%b-%Y")


def to_ist_time_only(value):
    """Time only — 12-hour format with AM/PM."""
    return to_ist(value, fmt="%I:%M %p")