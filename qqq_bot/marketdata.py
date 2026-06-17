from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

from .models import Bar


@dataclass
class BarBuilder:
    timeframe_min: int

    _current_start: Optional[datetime] = None
    _o: Optional[float] = None
    _h: float = 0.0
    _l: float = 0.0
    _c: Optional[float] = None
    _v: float = 0.0

    def _floor_time(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
        minutes = (ts.minute // self.timeframe_min) * self.timeframe_min
        return ts.replace(second=0, microsecond=0, minute=minutes)

    def update(self, ts: datetime, price: float, volume: float = 0.0) -> Optional[Bar]:
        """
        Возвращает готовый бар, если при апдейте тиком мы "перешли" в новый интервал.
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)

        start = self._floor_time(ts)
        if self._current_start is None:
            self._start_new(start, price, volume)
            return None

        if start != self._current_start:
            # close previous bar
            closed = Bar(
                ts=self._current_start,
                o=float(self._o),
                h=float(self._h),
                l=float(self._l),
                c=float(self._c),
                v=float(self._v),
            )
            # start new bar
            self._start_new(start, price, volume)
            return closed

        # update current bar
        self._c = price
        self._h = max(self._h, price)
        self._l = min(self._l, price)
        self._v += volume
        return None

    def force_close(self) -> Optional[Bar]:
        if self._current_start is None or self._o is None or self._c is None:
            return None
        return Bar(
            ts=self._current_start,
            o=float(self._o),
            h=float(self._h),
            l=float(self._l),
            c=float(self._c),
            v=float(self._v),
        )

    def _start_new(self, start: datetime, price: float, volume: float) -> None:
        self._current_start = start
        self._o = price
        self._h = price
        self._l = price
        self._c = price
        self._v = volume
