from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo


@dataclass
class CandleClose:
    end_utc: datetime          # timestamp "конца" свечи в UTC
    close: float
    synth: bool                # True, если свеча синтезирована (заполнили пропуск)


def _floor_to_tf(dt_utc: datetime, tf_min: int, tz_name: str) -> datetime:
    """
    Приводим время к "корзине" TF в заданной TZ (например America/New_York),
    затем возвращаем end time свечи в UTC.
    """
    tz = ZoneInfo(tz_name)
    local = dt_utc.astimezone(tz)
    minute = (local.minute // tf_min) * tf_min
    floored = local.replace(minute=minute, second=0, microsecond=0)
    end_local = floored + timedelta(minutes=tf_min)
    return end_local.astimezone(timezone.utc)


class RealtimeCloseBuilder:
    """
    Строит close по котировкам (ltp).
    ВАЖНО: в случае пропусков может досинтезировать свечи, но с лимитом max_synth_gaps.
    """

    def __init__(self, tf_min: int, bucket_tz: str, max_synth_gaps: int = 24) -> None:
        self.tf_min = tf_min
        self.bucket_tz = bucket_tz
        self.max_synth_gaps = int(max_synth_gaps)

        self._current_end_utc: Optional[datetime] = None
        self._current_close: Optional[float] = None

        self._last_final_end_utc: Optional[datetime] = None
        self._last_final_close: Optional[float] = None

    def seed_from_history(self, last_end_utc: datetime, last_close: float) -> None:
        self._last_final_end_utc = last_end_utc
        self._last_final_close = float(last_close)

    def update(self, now_utc: datetime, ltp: float) -> List[CandleClose]:
        now_utc = now_utc.astimezone(timezone.utc)
        ltp = float(ltp)

        bucket_end = _floor_to_tf(now_utc, self.tf_min, self.bucket_tz)
        produced: List[CandleClose] = []

        if self._current_end_utc is None:
            self._current_end_utc = bucket_end
            self._current_close = ltp
            return produced

        if bucket_end == self._current_end_utc:
            self._current_close = ltp
            return produced

        prev_end = self._current_end_utc
        prev_close = self._current_close if self._current_close is not None else ltp

        produced.append(CandleClose(end_utc=prev_end, close=float(prev_close), synth=False))
        self._last_final_end_utc = prev_end
        self._last_final_close = float(prev_close)

        next_end = prev_end + timedelta(minutes=self.tf_min)

        synth_count = 0
        while next_end < bucket_end:
            if self.max_synth_gaps >= 0 and synth_count >= self.max_synth_gaps:
                # Останавливаемся: слишком большой разрыв (например выходные)
                break
            produced.append(CandleClose(end_utc=next_end, close=float(self._last_final_close), synth=True))
            self._last_final_end_utc = next_end
            next_end = next_end + timedelta(minutes=self.tf_min)
            synth_count += 1

        self._current_end_utc = bucket_end
        self._current_close = ltp

        return produced

    def current_preview_close(self) -> Optional[Tuple[datetime, float]]:
        if self._current_end_utc is None or self._current_close is None:
            return None
        return self._current_end_utc, float(self._current_close)
