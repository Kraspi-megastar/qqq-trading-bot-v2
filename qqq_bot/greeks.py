"""
greeks.py — расчёт дельты опциона по Блэку-Шоулзу + подбор страйка по целевой дельте.

Используется как fallback, когда TraderNet не отдаёт греки в опционной цепочке.

Дельта:
  CALL: N(d1)
  PUT:  N(d1) - 1   (по модулю |delta| для удобства)

  d1 = (ln(S/K) + (r + σ²/2)·T) / (σ·√T)

где:
  S — цена базового актива
  K — страйк
  r — безрисковая ставка (по умолчанию ~5% годовых)
  σ — годовая волатильность
  T — время до экспирации в годах
"""
from __future__ import annotations

import math
from datetime import date


def _norm_cdf(x: float) -> float:
    """Кумулятивная функция стандартного нормального распределения."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(
    option_type: str,
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    r: float = 0.05,
) -> float:
    """
    Дельта по Блэку-Шоулзу. Возвращает абсолютное значение |delta| в [0, 1].
    """
    if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        # При экспирации: дельта 1 если ITM, иначе 0
        if option_type == "CALL":
            return 1.0 if spot > strike else 0.0
        else:
            return 1.0 if spot < strike else 0.0

    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * math.sqrt(t_years))

    if option_type == "CALL":
        return _norm_cdf(d1)
    else:
        return abs(_norm_cdf(d1) - 1.0)


def years_to_expiry(expiry: date, today: date) -> float:
    """Время до экспирации в годах (календарных)."""
    days = (expiry - today).days
    return max(days, 0) / 365.0


def strike_for_target_delta(
    option_type: str,
    spot: float,
    target_delta: float,
    t_years: float,
    sigma: float,
    strike_step: float = 1.0,
    r: float = 0.05,
    search_range_pct: float = 0.15,
) -> float:
    """
    Подбирает страйк, при котором |delta| ≈ target_delta.

    Перебирает страйки в диапазоне ±search_range_pct вокруг spot
    с шагом strike_step и выбирает тот, чья дельта ближе всего к целевой.
    """
    lo = spot * (1.0 - search_range_pct)
    hi = spot * (1.0 + search_range_pct)

    # генерируем сетку страйков, кратных strike_step
    first = math.floor(lo / strike_step) * strike_step
    candidates: list[float] = []
    k = first
    while k <= hi:
        if k > 0:
            candidates.append(round(k, 2))
        k += strike_step

    if not candidates:
        return round(spot / strike_step) * strike_step

    best_strike = candidates[0]
    best_diff = float("inf")
    for strike in candidates:
        d = bs_delta(option_type, spot, strike, t_years, sigma, r)
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_strike = strike

    return best_strike


def estimate_sigma_from_atr(atr: float, spot: float, bars_per_year: float = 19656.0) -> float:
    """
    Грубая оценка годовой волатильности из ATR на 5-минутном баре.

    bars_per_year для 5m: 252 торговых дня × 78 баров/день ≈ 19656.
    σ_annual ≈ (ATR/spot) × √(bars_per_year)

    Это очень приблизительно, но даёт разумный порядок величины
    когда нет других данных о волатильности.
    """
    if spot <= 0 or atr <= 0:
        return 0.20  # дефолт 20% если данных нет
    sigma = (atr / spot) * math.sqrt(bars_per_year)
    # ограничиваем разумными рамками
    return max(0.05, min(sigma, 1.50))
