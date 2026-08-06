"""
htf_filter.py — фильтр входов по тренду старшего таймфрейма (1h).

Стратегия #1 контртрендовая: хорошо работает в боковике, теряет в тренде.
Бэктест (45 сделок, 27.07-05.08) показал: если НЕ входить против тренда 1h,
результат улучшается с -$819 до +$383. Этот модуль определяет тренд 1h и
говорит, идёт ли сигнал против него.

Тренд считается по наклону EMA20 за 3 часа на часовых барах, агрегированных
из 5m. Порог настраивается (дефолт 0.6 пункта QQQ).
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def compute_htf_trend(df_5m: pd.DataFrame, ema_period: int = 20,
                      slope_hours: int = 3) -> tuple[float, float]:
    """
    Агрегирует 5m бары в 1h и считает наклон EMA за slope_hours часов.

    Возвращает (slope, last_ema):
      slope > 0 → восходящий тренд 1h
      slope < 0 → нисходящий тренд 1h
      |slope| мал → нет чёткого тренда (боковик)
    last_ema — текущее значение EMA (для справки).

    df_5m должен содержать колонки ts (datetime) и close.
    """
    if df_5m is None or len(df_5m) < 12:
        return 0.0, float("nan")

    d = df_5m.copy()
    if not pd.api.types.is_datetime64_any_dtype(d["ts"]):
        d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    d = d.dropna(subset=["ts"]).sort_values("ts").set_index("ts")

    # агрегируем в 1h
    h = d["close"].resample("1h").last().dropna()
    if len(h) < slope_hours + 2:
        return 0.0, float("nan")

    ema = h.ewm(span=ema_period, adjust=False).mean()
    # наклон за slope_hours часов
    if len(ema) <= slope_hours:
        return 0.0, float(ema.iloc[-1])
    slope = float(ema.iloc[-1] - ema.iloc[-1 - slope_hours])
    return slope, float(ema.iloc[-1])


def is_counter_trend(action: str, slope: float, threshold: float) -> bool:
    """
    Идёт ли сигнал ПРОТИВ тренда 1h.

    action: "BUY" (→CALL, бычья позиция) или "SELL" (→PUT, медвежья).
    slope: наклон EMA20 1h.
    threshold: минимальная |slope| чтобы считать тренд значимым.

    Против тренда:
      BUY когда 1h падает круче -threshold  → контртренд
      SELL когда 1h растёт круче +threshold  → контртренд
    Если |slope| <= threshold — тренда нет, не блокируем (боковик = вотчина #1).
    """
    if abs(slope) <= threshold:
        return False  # боковик — #1 разрешена
    if action == "BUY" and slope < -threshold:
        return True   # покупаем против падения
    if action == "SELL" and slope > threshold:
        return True   # продаём против роста
    return False


def trend_label(slope: float, threshold: float) -> str:
    """Человекочитаемая метка тренда 1h."""
    if abs(slope) <= threshold:
        return "боковик"
    return "восходящий" if slope > 0 else "нисходящий"
