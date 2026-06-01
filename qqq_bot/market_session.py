from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
UTC = timezone.utc
REGULAR_START = time(9, 30)
REGULAR_END = time(16, 0)


def parse_ts(ts: Any) -> datetime:
    """Parse a bar timestamp into timezone-aware UTC datetime."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        text = str(ts)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        # TraderNet/cache bars in this project are UTC by convention.
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_ny(ts: Any) -> datetime:
    return parse_ts(ts).astimezone(NY_TZ)


def is_regular_session(ts: Any, *, include_close: bool = True) -> bool:
    """US equity regular session check: Mon-Fri, 09:30-16:00 New York time.

    This intentionally does not model exchange holidays because the live feed will normally
    have no bars on closed days. It is enough for intraday signal gating.
    """
    ny = to_ny(ts)
    if ny.weekday() >= 5:
        return False
    t = ny.time().replace(tzinfo=None)
    if include_close:
        return REGULAR_START <= t <= REGULAR_END
    return REGULAR_START <= t < REGULAR_END


def is_friday(ts: Any) -> bool:
    return to_ny(ts).weekday() == 4


def is_thursday(ts: Any) -> bool:
    return to_ny(ts).weekday() == 3


def ny_date(ts: Any):
    return to_ny(ts).date()


def session_label(ts: Any) -> str:
    return "regular" if is_regular_session(ts) else "extended"
