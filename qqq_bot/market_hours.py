"""
market_hours.py — окна торговли опционами в основную сессию (RTH, ET).

Правила (значения в минутах настраиваются через OptionConfig):
  RTH: 9:30–16:00 ET, будни.
  Открытие опциона:   разрешено 9:40–15:45 (первые 10 мин и последние 15 мин — нет).
  Закрытие по сигналу: разрешено 9:40–15:45.
  Принудительное закрытие: в 15:45 ET (не держим позицию на ночь). Имеет приоритет
                           над всеми окнами.
На сам QQQ сигналы приходят и в расширенные часы — это контролируется отдельно
(в scheduler через _is_extended_session_open). Здесь только опционные окна.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo


RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


def _local(now_utc: datetime, tz_name: str) -> datetime:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(ZoneInfo(tz_name))


def _minutes_from_open(local_dt: datetime) -> int:
    """Минуты от 9:30 ET (может быть отрицательным до открытия)."""
    t = local_dt.time()
    return (t.hour * 60 + t.minute) - (RTH_OPEN.hour * 60 + RTH_OPEN.minute)


def _minutes_to_close(local_dt: datetime) -> int:
    """Минуты до 16:00 ET (может быть отрицательным после закрытия)."""
    t = local_dt.time()
    return (RTH_CLOSE.hour * 60 + RTH_CLOSE.minute) - (t.hour * 60 + t.minute)


def is_rth(now_utc: datetime, tz_name: str) -> bool:
    """Основная сессия: будни 9:30–16:00 ET."""
    local = _local(now_utc, tz_name)
    if local.weekday() >= 5:
        return False
    t = local.time()
    return RTH_OPEN <= t < RTH_CLOSE


def can_open_option(now_utc: datetime, tz_name: str, open_blackout_min: int, close_blackout_min: int) -> bool:
    """
    Можно ли ОТКРЫВАТЬ опцион сейчас.
    Только RTH, исключая первые open_blackout_min и последние close_blackout_min.
    """
    if not is_rth(now_utc, tz_name):
        return False
    local = _local(now_utc, tz_name)
    if _minutes_from_open(local) < open_blackout_min:
        return False
    if _minutes_to_close(local) <= close_blackout_min:
        return False
    return True


def can_close_option(now_utc: datetime, tz_name: str, open_blackout_min: int, close_blackout_min: int) -> bool:
    """
    Можно ли ЗАКРЫВАТЬ опцион по сигналу сейчас.
    Те же окна, что и для открытия (первые 10 мин и последние 15 мин — нет).
    Принудительное закрытие (should_force_close) проверяется отдельно и имеет приоритет.
    """
    if not is_rth(now_utc, tz_name):
        return False
    local = _local(now_utc, tz_name)
    if _minutes_from_open(local) < open_blackout_min:
        return False
    if _minutes_to_close(local) <= close_blackout_min:
        return False
    return True


def should_force_close(now_utc: datetime, tz_name: str, force_close_min: int) -> bool:
    """
    Пора ли принудительно закрыть позицию в конце дня.
    True когда до закрытия RTH осталось <= force_close_min минут (и рынок ещё открыт).
    """
    if not is_rth(now_utc, tz_name):
        return False
    local = _local(now_utc, tz_name)
    mtc = _minutes_to_close(local)
    return 0 < mtc <= force_close_min


def describe_window(now_utc: datetime, tz_name: str, cfg) -> str:
    """Текстовое описание текущего окна — для диагностики (/status, /optest)."""
    local = _local(now_utc, tz_name)
    if local.weekday() >= 5:
        return "выходной"
    if not is_rth(now_utc, tz_name):
        return "вне RTH (опционы закрыты)"
    if should_force_close(now_utc, tz_name, cfg.force_close_min):
        return "конец дня — принудительное закрытие"
    if can_open_option(now_utc, tz_name, cfg.open_blackout_min, cfg.close_blackout_min):
        return "RTH — открытие/закрытие разрешено"
    mfo = _minutes_from_open(local)
    if mfo < cfg.open_blackout_min:
        return f"первые {cfg.open_blackout_min} мин — опционы заблокированы"
    return f"последние {cfg.close_blackout_min} мин — открытие заблокировано"
