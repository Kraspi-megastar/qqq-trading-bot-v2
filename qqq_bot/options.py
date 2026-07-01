"""
options.py — управление опционными позициями для стратегий #1 и #2.

Логика позиций (обе стратегии двунаправленные):
  FLAT      + BUY  → OPEN CALL
  FLAT      + SELL → OPEN PUT
  CALL open + SELL → CLOSE CALL
  PUT  open + BUY  → CLOSE PUT
  CALL open + BUY  → HOLD CALL
  PUT  open + SELL → HOLD PUT

Выбор страйка (для OPEN):
  Цель — опцион с дельтой target_delta (по умолчанию 0.375, диапазон 0.35–0.40).
  Источник дельты, по приоритету:
    1. TraderNet опционная цепочка (если в ней есть поле delta)
    2. Black-Scholes (волатильность оценивается из ATR)
    3. Аппроксимация по расстоянию от ATM

Валидация существования:
  Перед открытием проверяем что опцион реально торгуется (есть котировка в TraderNet).
  Если ближайшая экспирация попадает на нерабочий день — опциона не будет,
  пробуем следующую пятницу.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional

import aiohttp

from .greeks import (
    bs_delta,
    strike_for_target_delta,
    years_to_expiry,
    estimate_sigma_from_atr,
)


OptionActionType = Literal["OPEN", "CLOSE", "HOLD"]


# ────────────────────────────────────────────────────────────────────────────
# Состояние позиции
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionPosition:
    option_type: str          # "CALL" | "PUT"
    ticker: str               # краткий тикер "QQQ 250718C480"
    tn_ticker: str            # тикер TraderNet "QQQ.18JUL2025.C480"
    strike: float
    expiry: date
    entry_underlying: float
    entry_date: date


@dataclass
class OptionConfig:
    enabled: bool = True
    min_dte: int = 0
    max_dte: int = 4               # максимальный срок до экспирации (дней)
    strike_step: float = 1.0
    underlying_symbol: str = "QQQ.US"
    # Целевая дельта для выбора страйка
    target_delta: float = 0.375
    # Сколько дней-кандидатов перебрать в поисках торгуемой экспирации
    max_expiry_tries: int = 8
    # Безрисковая ставка для Black-Scholes
    risk_free_rate: float = 0.05
    # Если True — НЕ открывать позицию пока TraderNet не подтвердит существование
    # контракта. Если False (по умолчанию) — при неудачной проверке открываем
    # по расчётному тикеру (валидация не должна глушить все сигналы).
    require_validation: bool = False
    # Временные окна (минуты), используются scheduler через market_hours
    open_blackout_min: int = 10
    close_blackout_min: int = 15
    force_close_min: int = 15


@dataclass
class OptionRecommendation:
    action_type: OptionActionType
    option_type: str
    ticker: str
    tn_ticker: str
    strike: float
    expiry: date
    dte: int
    underlying_price: float
    delta: Optional[float]          # дельта выбранного страйка (если известна)
    delta_source: str               # "tradernet" / "black-scholes" / "approx" / "-"
    moneyness: str
    source: str                     # источник тикера: "tradernet" / "calculated" / "position" / "none"
    new_position: Optional[OptionPosition]
    # Флаг: действие не выполнено т.к. рынок опционов закрыт (премаркет/афтермаркет)
    skipped_market_closed: bool = False
    # Флаг: не нашли торгуемый опцион
    skipped_no_contract: bool = False


# ────────────────────────────────────────────────────────────────────────────
# Тикеры и даты
# ────────────────────────────────────────────────────────────────────────────

def _candidate_expiries(today: date, min_dte: int, max_dte: int) -> list[date]:
    """
    Список дат-кандидатов для экспирации: каждый РАБОЧИЙ день (Пн–Пт)
    в окне [min_dte, max_dte] от сегодня, по возрастанию срока.
    QQQ имеет daily-опционы, поэтому перебираем все будни, а не только пятницы.
    Проверка реального существования контракта — выше по стеку (getSecurityInfo).
    """
    out: list[date] = []
    for d in range(min_dte, max_dte + 1):
        cand = today + timedelta(days=d)
        if cand.weekday() < 5:  # 0..4 = Пн..Пт
            out.append(cand)
    return out


def _nearest_business_day(from_date: date, min_dte: int) -> date:
    """Ближайший рабочий день не раньше from_date+min_dte (для заглушек/стабов)."""
    cand = from_date + timedelta(days=max(min_dte, 0))
    while cand.weekday() >= 5:
        cand += timedelta(days=1)
    return cand


def _dte(expiry: date, today: date) -> int:
    return (expiry - today).days


def short_ticker(option_type: str, strike: float, expiry: date) -> str:
    ot = "C" if option_type == "CALL" else "P"
    strike_str = str(int(strike)) if strike == int(strike) else f"{strike:.1f}"
    return f"QQQ {expiry.strftime('%y%m%d')}{ot}{strike_str}"


def tradernet_option_ticker(option_type: str, strike: float, expiry: date) -> str:
    """
    Формат TraderNet: +QQQ.31JUL2026.C732 / +QQQ.31JUL2026.P735
    Префикс '+' обязателен, день месяца с ведущим нулём (strftime %d).
    """
    ot = "C" if option_type == "CALL" else "P"
    date_str = expiry.strftime("%d%b%Y").upper()   # 31JUL2026
    strike_str = str(int(strike)) if strike == int(strike) else f"{strike:.2f}"
    return f"+QQQ.{date_str}.{ot}{strike_str}"


def _moneyness(option_type: str, strike: float, price: float, step: float) -> str:
    if abs(strike - price) <= step * 0.5:
        return "ATM"
    if option_type == "CALL":
        return "OTM" if strike > price else "ITM"
    else:
        return "OTM" if strike < price else "ITM"


# ────────────────────────────────────────────────────────────────────────────
# Решение о действии
# ────────────────────────────────────────────────────────────────────────────

def _resolve_action(
    signal: str,
    position: Optional[OptionPosition],
) -> tuple[OptionActionType, str]:
    if position is None:
        return "OPEN", ("CALL" if signal == "BUY" else "PUT")
    if position.option_type == "CALL":
        return ("CLOSE", "CALL") if signal == "SELL" else ("HOLD", "CALL")
    else:  # PUT
        return ("CLOSE", "PUT") if signal == "BUY" else ("HOLD", "PUT")


# ────────────────────────────────────────────────────────────────────────────
# Выбор страйка по дельте
# ────────────────────────────────────────────────────────────────────────────

async def _pick_strike_by_delta(
    *,
    tn,                              # TraderNetClient | None
    cfg: OptionConfig,
    option_type: str,
    spot: float,
    expiry: date,
    today: date,
    atr: Optional[float],
) -> tuple[float, Optional[float], str]:
    """
    Возвращает (strike, delta, delta_source).

    Приоритет:
      1) TraderNet getSecurityInfo — пробуем страйки вокруг оценки BS и читаем
         РЕАЛЬНУЮ дельту каждого контракта, выбираем ближайший к target_delta.
      2) Black-Scholes (волатильность из ATR).
      3) Аппроксимация по расстоянию от ATM.
    """
    target = cfg.target_delta

    # Предварительная оценка страйка по BS (чтобы знать где искать в цепочке)
    t_years = years_to_expiry(expiry, today)
    sigma = estimate_sigma_from_atr(atr, spot) if (atr and atr > 0) else 0.25
    if t_years > 0:
        bs_strike = strike_for_target_delta(
            option_type, spot, target, t_years, sigma,
            strike_step=cfg.strike_step, r=cfg.risk_free_rate,
        )
    else:
        bs_strike = round(spot / cfg.strike_step) * cfg.strike_step

    # 1) Реальные дельты из TraderNet — пробуем страйки вокруг bs_strike
    if tn is not None:
        best_strike = None
        best_delta = None
        best_diff = float("inf")
        # перебираем страйки в окне ±5 шагов вокруг расчётного
        steps = sorted(range(-5, 6), key=lambda k: abs(k))
        checked = 0
        for k in steps:
            cand = bs_strike + k * cfg.strike_step
            if cand <= 0:
                continue
            tn_tick = tradernet_option_ticker(option_type, cand, expiry)
            try:
                greeks = await asyncio.wait_for(tn.get_option_greeks(tn_tick), timeout=6.0)
            except Exception:
                greeks = None
            if greeks is None or greeks.get("delta") is None:
                continue
            checked += 1
            d = abs(float(greeks["delta"]))
            diff = abs(d - target)
            if diff < best_diff:
                best_diff = diff
                best_strike = cand
                best_delta = d
            # ранний выход: попали в целевой диапазон 0.35–0.40
            if 0.34 <= d <= 0.41:
                return float(cand), d, "tradernet"
            # ограничим число сетевых запросов
            if checked >= 8:
                break
        if best_strike is not None:
            return float(best_strike), best_delta, "tradernet"

    # 2) Black-Scholes
    if t_years > 0 and atr is not None and atr > 0:
        actual_delta = bs_delta(option_type, spot, bs_strike, t_years, sigma, cfg.risk_free_rate)
        return bs_strike, actual_delta, "black-scholes"

    # 3) Аппроксимация
    offset_pct = 0.015
    if option_type == "CALL":
        raw = spot * (1.0 + offset_pct)
    else:
        raw = spot * (1.0 - offset_pct)
    strike = round(raw / cfg.strike_step) * cfg.strike_step
    return float(strike), None, "approx"


# ────────────────────────────────────────────────────────────────────────────
# Поиск торгуемой экспирации
# ────────────────────────────────────────────────────────────────────────────

async def _find_tradable_expiry_and_strike(
    *,
    tn,
    cfg: OptionConfig,
    option_type: str,
    spot: float,
    today: date,
    atr: Optional[float],
) -> Optional[tuple[date, float, Optional[float], str]]:
    """
    Перебирает рабочие дни в окне [min_dte, max_dte], для каждого подбирает
    страйк по дельте и проверяет что опцион реально торгуется (getSecurityInfo).
    Берёт БЛИЖАЙШУЮ доступную экспирацию (короткий срок по стратегии).

    Возвращает (expiry, strike, delta, delta_source) или None если ничего не найдено.
    Если tn is None — не можем проверить, возвращаем первый расчёт.
    """
    candidates = _candidate_expiries(today, cfg.min_dte, cfg.max_dte)
    if not candidates:
        # окно пустое (например все дни выходные) — берём ближайший рабочий день
        candidates = [_nearest_business_day(today, cfg.min_dte)]

    # Запоминаем самый первый расчётный вариант — fallback,
    # если ни один контракт не удалось ПОДТВЕРДИТЬ через TraderNet.
    fallback: Optional[tuple[date, float, Optional[float], str]] = None

    for expiry in candidates:
        strike, delta, delta_source = await _pick_strike_by_delta(
            tn=tn, cfg=cfg, option_type=option_type,
            spot=spot, expiry=expiry, today=today, atr=atr,
        )

        if fallback is None:
            fallback = (expiry, strike, delta, delta_source)

        # Без TraderNet проверить не можем — возвращаем расчёт как есть
        if tn is None:
            return expiry, strike, delta, delta_source

        # Проверяем существование опциона
        tn_tick = tradernet_option_ticker(option_type, strike, expiry)
        try:
            exists = await asyncio.wait_for(tn.option_exists(tn_tick), timeout=8.0)
        except Exception:
            exists = False

        if exists:
            return expiry, strike, delta, delta_source

        # Если страйк не торгуется — пробуем соседние ±1..2 шага на этой же экспирации
        for delta_steps in (1, -1, 2, -2):
            alt_strike = strike + delta_steps * cfg.strike_step
            if alt_strike <= 0:
                continue
            alt_tick = tradernet_option_ticker(option_type, alt_strike, expiry)
            try:
                if await asyncio.wait_for(tn.option_exists(alt_tick), timeout=6.0):
                    t_years = years_to_expiry(expiry, today)
                    alt_delta = None
                    if t_years > 0 and atr and atr > 0:
                        sigma = estimate_sigma_from_atr(atr, spot)
                        alt_delta = bs_delta(option_type, spot, alt_strike, t_years, sigma, cfg.risk_free_rate)
                    return expiry, alt_strike, alt_delta, delta_source
            except Exception:
                continue

    # Ни один контракт не подтверждён через TraderNet.
    # По умолчанию (require_validation=False) НЕ блокируем сигнал — открываем
    # позицию по первому расчётному варианту. Это защищает от ситуации, когда
    # опционный эндпоинт TraderNet просто не отдаёт котировки в ожидаемом формате.
    if not cfg.require_validation and fallback is not None:
        return fallback

    return None


# ────────────────────────────────────────────────────────────────────────────
# Публичный API
# ────────────────────────────────────────────────────────────────────────────

async def get_option_recommendation(
    signal: str,
    underlying_price: float,
    cfg: OptionConfig,
    current_position: Optional[OptionPosition],
    can_open: bool = True,           # окно открытия RTH (9:40–15:45)
    can_close: bool = True,          # окно закрытия RTH (9:40–15:45)
    atr: Optional[float] = None,     # для оценки волатильности
    tn=None,                         # TraderNetClient для проверки/цепочки
    session: Optional[aiohttp.ClientSession] = None,  # legacy
    api_url: str = "",
    sid: Optional[str] = None,
) -> OptionRecommendation:
    """
    Рекомендация с учётом позиции, окон торговли опционами и существования контракта.

    can_open  — можно ли открывать опцион сейчас (RTH минус блэкауты).
    can_close — можно ли закрывать опцион по сигналу сейчас.

    Если действие выпадает на запрещённое окно:
      OPEN  вне окна → HOLD, позиция НЕ открывается (skipped_market_closed=True).
      CLOSE вне окна → CLOSE с пометкой pending (skipped_market_closed=True),
                       позиция НЕ снимается — scheduler пометит её "ожидает закрытия".
    """
    action_type, option_type = _resolve_action(signal, current_position)
    today = datetime.now(tz=timezone.utc).date()

    # ── CLOSE: используем параметры открытой позиции ──────────────────────
    if action_type == "CLOSE" and current_position is not None:
        # Вне окна закрытия — не снимаем позицию, помечаем pending.
        new_pos = current_position if not can_close else None
        return OptionRecommendation(
            action_type="CLOSE",
            option_type=option_type,
            ticker=current_position.ticker,
            tn_ticker=current_position.tn_ticker,
            strike=current_position.strike,
            expiry=current_position.expiry,
            dte=_dte(current_position.expiry, today),
            underlying_price=underlying_price,
            delta=None,
            delta_source="-",
            moneyness=_moneyness(option_type, current_position.strike, underlying_price, cfg.strike_step),
            source="position",
            new_position=new_pos,
            skipped_market_closed=(not can_close),
        )

    # ── HOLD ──────────────────────────────────────────────────────────────
    if action_type == "HOLD":
        if current_position is not None:
            return OptionRecommendation(
                action_type="HOLD",
                option_type=option_type,
                ticker=current_position.ticker,
                tn_ticker=current_position.tn_ticker,
                strike=current_position.strike,
                expiry=current_position.expiry,
                dte=_dte(current_position.expiry, today),
                underlying_price=underlying_price,
                delta=None,
                delta_source="-",
                moneyness=_moneyness(option_type, current_position.strike, underlying_price, cfg.strike_step),
                source="position",
                new_position=current_position,
            )
        else:
            stub_expiry = _nearest_business_day(today, cfg.min_dte)
            return OptionRecommendation(
                action_type="HOLD", option_type="CALL",
                ticker="-", tn_ticker="-", strike=0.0, expiry=stub_expiry,
                dte=_dte(stub_expiry, today), underlying_price=underlying_price,
                delta=None, delta_source="-", moneyness="-", source="none",
                new_position=None,
            )

    # ── OPEN ────────────────────────────────────────────────────────────────
    # Опционы открываем ТОЛЬКО в разрешённое окно RTH.
    if not can_open:
        stub_expiry = _nearest_business_day(today, cfg.min_dte)
        return OptionRecommendation(
            action_type="HOLD",          # действие не выполнено
            option_type=option_type,
            ticker="-", tn_ticker="-", strike=0.0, expiry=stub_expiry,
            dte=_dte(stub_expiry, today), underlying_price=underlying_price,
            delta=None, delta_source="-", moneyness="-", source="none",
            new_position=None,            # позиция НЕ открывается
            skipped_market_closed=True,
        )

    # Ищем торгуемую экспирацию + страйк по целевой дельте
    found = await _find_tradable_expiry_and_strike(
        tn=tn, cfg=cfg, option_type=option_type,
        spot=underlying_price, today=today, atr=atr,
    )

    if found is None:
        # Не нашли торгуемый контракт — не открываем позицию
        stub_expiry = _nearest_business_day(today, cfg.min_dte)
        return OptionRecommendation(
            action_type="HOLD", option_type=option_type,
            ticker="-", tn_ticker="-", strike=0.0, expiry=stub_expiry,
            dte=_dte(stub_expiry, today), underlying_price=underlying_price,
            delta=None, delta_source="-", moneyness="-", source="none",
            new_position=None,
            skipped_no_contract=True,
        )

    expiry, strike, delta, delta_source = found
    dte = _dte(expiry, today)
    moneyness = _moneyness(option_type, strike, underlying_price, cfg.strike_step)
    tn_tick = tradernet_option_ticker(option_type, strike, expiry)
    sh_tick = short_ticker(option_type, strike, expiry)

    new_position = OptionPosition(
        option_type=option_type,
        ticker=sh_tick,
        tn_ticker=tn_tick,
        strike=strike,
        expiry=expiry,
        entry_underlying=underlying_price,
        entry_date=today,
    )

    return OptionRecommendation(
        action_type="OPEN",
        option_type=option_type,
        ticker=sh_tick,
        tn_ticker=tn_tick,
        strike=strike,
        expiry=expiry,
        dte=dte,
        underlying_price=underlying_price,
        delta=delta,
        delta_source=delta_source,
        moneyness=moneyness,
        source="tradernet" if tn is not None else "calculated",
        new_position=new_position,
    )


def build_close_recommendation(
    position: OptionPosition,
    underlying_price: float,
    cfg: OptionConfig,
    reason: str = "close",
) -> OptionRecommendation:
    """
    Строит CLOSE-рекомендацию для существующей позиции напрямую
    (без сигнала). Используется для:
      - принудительного закрытия в конце дня (reason="force_close"),
      - отложенного закрытия при открытии рынка (reason="pending_close").
    Позиция всегда снимается (new_position=None).
    """
    today = datetime.now(tz=timezone.utc).date()
    return OptionRecommendation(
        action_type="CLOSE",
        option_type=position.option_type,
        ticker=position.ticker,
        tn_ticker=position.tn_ticker,
        strike=position.strike,
        expiry=position.expiry,
        dte=_dte(position.expiry, today),
        underlying_price=underlying_price,
        delta=None,
        delta_source=reason,   # храним причину в delta_source для форматтера
        moneyness=_moneyness(position.option_type, position.strike, underlying_price, cfg.strike_step),
        source="position",
        new_position=None,
        skipped_market_closed=False,
    )


# ────────────────────────────────────────────────────────────────────────────
# Форматирование
# ────────────────────────────────────────────────────────────────────────────

def format_option_message(rec: OptionRecommendation) -> str:
    if rec.skipped_market_closed and rec.action_type in ("HOLD", "OPEN"):
        return (
            "⏸ <b>Опцион: пропуск</b>\n"
            "Рынок опционов закрыт (вне основной сессии 9:30–16:00 ET).\n"
            "Сигнал зафиксирован, опцион откроется при подтверждении в RTH."
        )
    if rec.skipped_no_contract:
        return (
            "⚠️ <b>Опцион: контракт не найден</b>\n"
            "Не удалось найти торгуемый опцион с нужной экспирацией/страйком."
        )

    if rec.action_type == "OPEN":
        emoji = "📈" if rec.option_type == "CALL" else "📉"
        header = f"{emoji} <b>Опцион: ОТКРЫТЬ {rec.option_type}</b>"
    elif rec.action_type == "CLOSE":
        if rec.delta_source == "force_close":
            header = f"🔚 <b>Опцион: ЗАКРЫТЬ {rec.option_type}</b> (конец дня, не держим ночь)"
        elif rec.delta_source == "pending_close":
            header = f"🔓 <b>Опцион: ЗАКРЫТЬ {rec.option_type}</b> (отложенное закрытие, рынок открылся)"
        elif rec.delta_source == "conflict_close":
            header = f"🔻 <b>Опцион: ЗАКРЫТЬ {rec.option_type}</b> (конфликт консенсуса, досрочно)"
        else:
            header = f"🔒 <b>Опцион: ЗАКРЫТЬ {rec.option_type}</b>"
        if rec.skipped_market_closed:
            header += "\n⚠️ вне окна закрытия — закроется при открытии RTH"
    else:  # HOLD
        if rec.new_position is not None:
            header = f"⏸ <b>Опцион: ДЕРЖАТЬ {rec.option_type}</b> (уже в позиции)"
        else:
            return "⏸ <b>Опцион: нет действия</b>"

    lines = [
        header,
        f"Тикер: <code>{rec.ticker}</code>",
        f"TraderNet: <code>{rec.tn_ticker}</code>",
        f"Страйк: {rec.strike:.0f}  |  {rec.moneyness}",
    ]
    if rec.delta is not None:
        lines.append(f"Дельта: {rec.delta:.3f}  (источник: {rec.delta_source})")
    lines += [
        f"Экспирация: {rec.expiry.strftime('%d %b %Y')}  ({rec.dte} DTE)",
        f"QQQ сейчас: {rec.underlying_price:.2f}",
    ]
    return "\n".join(lines)
