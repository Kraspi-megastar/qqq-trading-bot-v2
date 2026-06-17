from __future__ import annotations

from datetime import datetime, timezone, timedelta


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def floor_time(dt: datetime, timeframe_minutes: int) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    epoch = int(dt.timestamp())
    step = timeframe_minutes * 60
    floored = epoch - (epoch % step)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def dt_to_tradernet_str(dt: datetime) -> str:
    # TraderNet expects dd.MM.yyyy hh:mm (per getHloc description):contentReference[oaicite:5]{index=5}
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%d.%m.%Y %H:%M")


def safe_float(x: str | float | int | None) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None
