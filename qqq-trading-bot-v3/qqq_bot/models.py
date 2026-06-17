from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Bar:
    ts: datetime  # UTC, bar open time aligned to timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    synthetic: bool = False  # filled for non-trading gaps


@dataclass
class Quote:
    symbol: str
    ltp: float | None
    ltt: datetime | None  # UTC
