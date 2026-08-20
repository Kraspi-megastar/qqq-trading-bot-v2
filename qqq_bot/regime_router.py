"""
regime_router.py — режимный роутер #1 ↔ #2 по тренду старшего ТФ (1h).

Логика (подтверждена бэктестом на 7 месяцах полного RTH, порог 0.45):
  - БОКОВИК (|наклон EMA20 1h| <= порог) → активна #1 (контртрендовая, её вотчина).
      #1 в боковике за 7 мес: +$3312. В тренде #1 теряет (её глушит фильтр).
  - ТРЕНД (|наклон| > порог) → активна #2 (трендследящая, по тренду).
      #2 в тренде: +$764 винрейт 49%. В боковике #2 теряет (-$921) — глушим.

Роутер решает ТОЛЬКО про новые входы. Открытую позицию не трогает
(она доходит до штатного выхода/стопа/force-close).

Обе стратегии используют ЕДИНЫЙ риск-контур: тот же размер (position_pct,
max_contracts), стоп-лосс, паузу после стопа.
"""
from __future__ import annotations
import pandas as pd
from .htf_filter import compute_htf_trend


def decide_regime(df_5m: pd.DataFrame, threshold: float) -> tuple[str, float]:
    """
    Определяет режим рынка по тренду 1h.

    Возвращает (regime, slope):
      regime = "trend"   → |наклон| > threshold  (активна #2)
      regime = "range"   → |наклон| <= threshold (активна #1)
    slope — наклон EMA20 1h (для сообщений и логов).
    """
    slope, _ = compute_htf_trend(df_5m, ema_period=20, slope_hours=3)
    regime = "trend" if abs(slope) > threshold else "range"
    return regime, slope


def active_strategy(regime: str) -> int:
    """Какая стратегия активна в данном режиме. trend→2, range→1."""
    return 2 if regime == "trend" else 1


def regime_label(regime: str, slope: float) -> str:
    """Человекочитаемое описание для канала."""
    if regime == "trend":
        direction = "восходящий" if slope > 0 else "нисходящий"
        return f"тренд 1h ({direction}, наклон {slope:+.2f}) → активна #2"
    return f"боковик 1h (наклон {slope:+.2f}) → активна #1"
