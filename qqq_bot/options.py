"""
options.py — управление опционными позициями для стратегии #1.

Логика позиций:
  Состояние FLAT (нет открытой позиции):
    BUY  → OPEN CALL
    SELL → OPEN PUT

  Состояние LONG CALL (открыт CALL):
    SELL → CLOSE CALL   ← разворот, не открываем PUT
    BUY  → HOLD         ← уже в позиции, игнорируем

  Состояние LONG PUT (открыт PUT):
    BUY  → CLOSE PUT    ← разворот, не открываем CALL
    SELL → HOLD         ← уже в позиции, игнорируем

После CLOSE позиция сбрасывается в FLAT.
Следующий сигнал того же направления откроет новую позицию.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional

import aiohttp


# ────────────────────────────────────────────────────────────────────────────
# Типы действий над опционом
# ────────────────────────────────────────────────────────────────────────────

OptionActionType = Literal["OPEN", "CLOSE", "HOLD"]


# ────────────────────────────────────────────────────────────────────────────
# Состояние текущей опционной позиции
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionPosition:
    """Открытая опционная позиция. None = позиция закрыта (FLAT)."""
    option_type: str          # "CALL" | "PUT"
    ticker: str               # краткий тикер "QQQ 250718C480"
    tn_ticker: str            # тикер TraderNet "QQQ.18JUL2025.C480" для котировок
    strike: float
    expiry: date
    entry_underlying: float   # цена QQQ на момент открытия
    entry_date: date          # дата открытия

    def describe(self) -> str:
        return (
            f"{self.option_type} {self.ticker} "
            f"страйк={self.strike:.0f} экспирация={self.expiry.strftime('%d %b %Y')} "
            f"открыт по QQQ={self.entry_underlying:.2f}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Конфиг
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionConfig:
    enabled: bool = True
    min_dte: int = 1
    strike_step: float = 1.0
    strike_offset: int = 0        # 0=ATM, 1=1 шаг OTM, -1=1 шаг ITM
    underlying_symbol: str = "QQQ.US"


# ────────────────────────────────────────────────────────────────────────────
# Результат рекомендации
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionRecommendation:
    action_type: OptionActionType   # "OPEN" | "CLOSE" | "HOLD"
    option_type: str                # "CALL" | "PUT"
    ticker: str                     # краткий тикер QQQ 260619C480
    tn_ticker: str                  # тикер TraderNet QQQ.19JUN2026.C480 (для котировок)
    strike: float
    expiry: date
    dte: int
    underlying_price: float
    moneyness: str                  # "ATM" / "OTM" / "ITM"
    source: str                     # "tradernet" / "calculated"
    # позиция после применения этой рекомендации
    new_position: Optional[OptionPosition]


# ────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ────────────────────────────────────────────────────────────────────────────

def _next_expiry(from_date: date, min_dte: int) -> date:
    """Ближайшая пятница >= from_date + min_dte."""
    target = from_date + timedelta(days=min_dte)
    days_to_friday = (4 - target.weekday()) % 7
    return target + timedelta(days=days_to_friday)


def _dte(expiry: date, today: date) -> int:
    return (expiry - today).days


def _build_ticker(option_type: str, strike: float, expiry: date) -> str:
    ot = "C" if option_type == "CALL" else "P"
    strike_str = str(int(strike)) if strike == int(strike) else f"{strike:.1f}"
    return f"QQQ {expiry.strftime('%y%m%d')}{ot}{strike_str}"


def _round_strike(price: float, step: float, offset: int, option_type: str) -> float:
    atm = round(price / step) * step
    if option_type == "CALL":
        return float(atm + offset * step)
    else:
        return float(atm - offset * step)


def _moneyness(option_type: str, strike: float, price: float, step: float) -> str:
    if abs(strike - price) <= step * 0.5:
        return "ATM"
    if option_type == "CALL":
        return "OTM" if strike > price else "ITM"
    else:
        return "OTM" if strike < price else "ITM"


# ────────────────────────────────────────────────────────────────────────────
# Основная логика позиции
# ────────────────────────────────────────────────────────────────────────────

def _resolve_action(
    signal: str,                        # "BUY" | "SELL"
    position: Optional[OptionPosition], # текущая открытая позиция
    can_short: bool = True,             # False для стратегий без шорта (напр. стратегия #2)
) -> tuple[OptionActionType, str]:      # (action_type, option_type)
    """
    Определяет что нужно сделать с опционом.

    can_short=True  (стратегия #1, двунаправленная):
      BUY  при FLAT → OPEN CALL
      SELL при FLAT → OPEN PUT
      SELL при CALL → CLOSE CALL
      BUY  при PUT  → CLOSE PUT
      BUY  при CALL → HOLD CALL
      SELL при PUT  → HOLD PUT

    can_short=False (стратегия #2, только лонг):
      BUY  при FLAT → OPEN CALL
      SELL при FLAT → HOLD (нет позиции для закрытия, шорт не открываем)
      SELL при CALL → CLOSE CALL
      BUY  при CALL → HOLD CALL
    """
    if position is None:
        if signal == "BUY":
            return "OPEN", "CALL"
        else:  # SELL
            if can_short:
                return "OPEN", "PUT"
            else:
                # Стратегия без шорта: SELL при FLAT — просто нет позиции
                return "HOLD", "CALL"  # тип не важен, action=HOLD

    if position.option_type == "CALL":
        if signal == "SELL":
            return "CLOSE", "CALL"
        else:
            return "HOLD", "CALL"

    else:  # position.option_type == "PUT" (только стратегия #1)
        if signal == "BUY":
            return "CLOSE", "PUT"
        else:
            return "HOLD", "PUT"


def tradernet_option_ticker(option_type: str, strike: float, expiry: date) -> str:
    """
    Формат TraderNet для запроса цены опциона: QQQ.17JUN2026.C749
    Используется в get_quote_ltp чтобы получить рыночную цену опциона.
    """
    ot = "C" if option_type == "CALL" else "P"
    date_str = expiry.strftime("%d%b%Y").upper()   # 17JUN2026
    strike_str = str(int(strike)) if strike == int(strike) else f"{strike:.2f}"
    return f"QQQ.{date_str}.{ot}{strike_str}"


# ────────────────────────────────────────────────────────────────────────────
# TraderNet — попытка получить реальный тикер из цепочки
# ────────────────────────────────────────────────────────────────────────────

async def _try_tradernet_option(
    session: aiohttp.ClientSession,
    api_url: str,
    underlying: str,
    option_type: str,
    strike: float,
    expiry: date,
    sid: Optional[str],
    timeout: int = 10,
) -> Optional[str]:
    payload: dict = {
        "cmd": "getOptionChain",
        "params": {"id": underlying, "type": option_type.lower()},
    }
    if sid:
        payload["SID"] = sid

    try:
        to = aiohttp.ClientTimeout(total=float(timeout))
        async with session.post(
            api_url,
            data={"q": json.dumps(payload, ensure_ascii=False)},
            timeout=to,
            headers={"User-Agent": "qqq_trading_bot/2.0"},
            cookies={"SID": sid} if sid else None,
        ) as r:
            if r.status != 200:
                return None
            data = json.loads(await r.text())
    except Exception:
        return None

    contracts = None
    if isinstance(data, list):
        contracts = data
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v:
                contracts = v
                break

    if not contracts:
        return None

    expiry_str = expiry.strftime("%Y-%m-%d")
    for c in contracts:
        if not isinstance(c, dict):
            continue
        c_strike = c.get("strike") or c.get("exercise_price")
        c_expiry = c.get("expiry") or c.get("expiration") or c.get("exp_date")
        c_ticker = c.get("ticker") or c.get("symbol") or c.get("id")
        if c_strike is None or c_expiry is None or c_ticker is None:
            continue
        try:
            if abs(float(c_strike) - strike) < 0.01 and expiry_str in str(c_expiry):
                return str(c_ticker)
        except (TypeError, ValueError):
            continue

    return None


# ────────────────────────────────────────────────────────────────────────────
# Публичный API
# ────────────────────────────────────────────────────────────────────────────

async def get_option_recommendation(
    signal: str,                                    # "BUY" | "SELL"
    underlying_price: float,
    cfg: OptionConfig,
    current_position: Optional[OptionPosition],     # текущая открытая позиция
    can_short: bool = True,                         # False для стратегии #2 (только лонг)
    session: Optional[aiohttp.ClientSession] = None,
    api_url: str = "https://tradernet.ru/api/",
    sid: Optional[str] = None,
) -> OptionRecommendation:
    """
    Возвращает рекомендацию с учётом текущей позиции.

    CLOSE означает закрытие существующей позиции.
    OPEN  означает открытие новой позиции (CALL или PUT в зависимости от can_short).
    HOLD  означает что действие не требуется.

    can_short=False (стратегия #2): SELL при FLAT → HOLD, PUT никогда не открывается.
    """
    action_type, option_type = _resolve_action(signal, current_position, can_short)

    today = datetime.now(tz=timezone.utc).date()

    # При CLOSE используем параметры текущей позиции (не пересчитываем страйк)
    if action_type == "CLOSE" and current_position is not None:
        ticker = current_position.ticker
        strike = current_position.strike
        expiry = current_position.expiry
        dte = _dte(expiry, today)
        moneyness = _moneyness(option_type, strike, underlying_price, cfg.strike_step)
        return OptionRecommendation(
            action_type="CLOSE",
            option_type=option_type,
            ticker=ticker,
            tn_ticker=current_position.tn_ticker,
            strike=strike,
            expiry=expiry,
            dte=dte,
            underlying_price=underlying_price,
            moneyness=moneyness,
            source="position",
            new_position=None,
        )

    # При HOLD — два случая:
    #   1. Есть открытая позиция → возвращаем её без изменений
    #   2. Нет позиции (SELL при FLAT, can_short=False) → возвращаем-заглушку без действия
    if action_type == "HOLD":
        if current_position is not None:
            dte = _dte(current_position.expiry, today)
            moneyness = _moneyness(option_type, current_position.strike, underlying_price, cfg.strike_step)
            return OptionRecommendation(
                action_type="HOLD",
                option_type=option_type,
                ticker=current_position.ticker,
                tn_ticker=current_position.tn_ticker,
                strike=current_position.strike,
                expiry=current_position.expiry,
                dte=dte,
                underlying_price=underlying_price,
                moneyness=moneyness,
                source="position",
                new_position=current_position,
            )
        else:
            # SELL при FLAT для стратегии без шорта — ничего не делать
            expiry_stub = _next_expiry(today, cfg.min_dte)
            return OptionRecommendation(
                action_type="HOLD",
                option_type="CALL",
                ticker="-",
                tn_ticker="-",
                strike=0.0,
                expiry=expiry_stub,
                dte=_dte(expiry_stub, today),
                underlying_price=underlying_price,
                moneyness="-",
                source="none",
                new_position=None,
            )

    # OPEN — рассчитываем новый контракт
    expiry = _next_expiry(today, cfg.min_dte)
    dte = _dte(expiry, today)
    strike = _round_strike(underlying_price, cfg.strike_step, cfg.strike_offset, option_type)
    moneyness = _moneyness(option_type, strike, underlying_price, cfg.strike_step)

    tn_ticker: Optional[str] = None
    if session is not None:
        try:
            tn_ticker = await asyncio.wait_for(
                _try_tradernet_option(session, api_url, cfg.underlying_symbol,
                                      option_type, strike, expiry, sid),
                timeout=8.0,
            )
        except Exception:
            tn_ticker = None

    source = "tradernet" if tn_ticker else "calculated"
    ticker = tn_ticker or _build_ticker(option_type, strike, expiry)

    tn_tick = tradernet_option_ticker(option_type, strike, expiry)

    new_position = OptionPosition(
        option_type=option_type,
        ticker=ticker,
        tn_ticker=tn_tick,
        strike=strike,
        expiry=expiry,
        entry_underlying=underlying_price,
        entry_date=today,
    )

    return OptionRecommendation(
        action_type="OPEN",
        option_type=option_type,
        ticker=ticker,
        tn_ticker=tn_tick,
        strike=strike,
        expiry=expiry,
        dte=dte,
        underlying_price=underlying_price,
        moneyness=moneyness,
        source=source,
        new_position=new_position,
    )


# ────────────────────────────────────────────────────────────────────────────
# Форматирование сообщения
# ────────────────────────────────────────────────────────────────────────────

def format_option_message(rec: OptionRecommendation) -> str:
    """Форматирует блок для Telegram-сообщения (HTML)."""

    if rec.action_type == "OPEN":
        emoji = "📈" if rec.option_type == "CALL" else "📉"
        header = f"{emoji} <b>Опцион: ОТКРЫТЬ {rec.option_type}</b>"
    elif rec.action_type == "CLOSE":
        emoji = "🔒"
        header = f"{emoji} <b>Опцион: ЗАКРЫТЬ {rec.option_type}</b>"
    else:  # HOLD
        emoji = "⏸"
        if rec.new_position is not None:
            header = f"{emoji} <b>Опцион: ДЕРЖАТЬ {rec.option_type}</b> (уже в позиции)"
        else:
            header = f"{emoji} <b>Опцион: нет действия</b> (позиция не открыта, шорт не используется)"

    lines = [
        header,
        f"Тикер: <code>{rec.ticker}</code>",
        f"Страйк: {rec.strike:.0f}  |  {rec.moneyness}",
        f"Экспирация: {rec.expiry.strftime('%d %b %Y')}  ({rec.dte} DTE)",
        f"QQQ сейчас: {rec.underlying_price:.2f}",
    ]

    if rec.action_type != "OPEN":
        lines.append(f"Источник тикера: {rec.source}")

    return "\n".join(lines)
