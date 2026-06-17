"""
handlers.py — Telegram-команды бота.

Изменения v3:
  - Убраны дублирующие _bars_to_df / _add_indicators / _min_bars_for_indicators;
    теперь используется pipeline.py.
  - compute_signal вызывается с state= вместо runtime_state=.
  - Добавлена команда /options — показывает текущий опционный контракт.
  - Исправлен /status: пересчёт сигнала не меняет app.strategy2.
"""
from __future__ import annotations

import html
import re
from datetime import timedelta, timezone
from typing import Any

import pandas as pd
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile

from .scheduler import AppState
from .pipeline import bars_to_df, add_indicators, min_bars_for_indicators
from .signals import compute_signal, Strategy2State
from .utils_time import utc_now

router = Router()


def _fmt_ts_z(dt) -> str:
    if dt is None:
        return "-"
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cooldown_left_seconds(app: AppState) -> int:
    cd = int(app.cfg.cooldown_seconds)
    if cd <= 0 or app.stats.last_signal_sent_at is None:
        return 0
    now = utc_now()
    left = int((app.stats.last_signal_sent_at + timedelta(seconds=cd) - now).total_seconds())
    return max(left, 0)


def _position_from_history(app: AppState) -> str:
    if app.strategy_id == 2:
        return app.strategy2.position
    hist = getattr(app.stats, "signal_history", None) or []
    if not hist:
        return "FLAT"
    last = hist[-1]
    action = last[0] if isinstance(last, (tuple, list)) and len(last) >= 1 else None
    return "LONG" if action == "BUY" else "FLAT"


def _help_text() -> str:
    return (
        "Команды бота:\n\n"
        "/start — запуск и краткая справка\n"
        "/help — список команд\n"
        "/status — подробный статус\n"
        "/stats — краткая статистика\n"
        "/chart — текущий график\n"
        "/last — последняя цена и последний бар\n"
        "/options — текущий опционный контракт (стр. #1)\n"
        "/dump [N] — последние N баров\n"
        "/config — текущие настройки\n"
        "/strategy — показать текущую стратегию\n"
        "/strategy#1 — включить стратегию #1 (BB+EMA+RSI + опционы)\n"
        "/strategy#2 — включить стратегию #2 (MACD+VWAP+RSI+Supertrend)\n"
        "/ping — проверка, что бот жив\n"
    )


# ────────────────────────────────────────────────────────────────────────────
# Команды
# ────────────────────────────────────────────────────────────────────────────

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer("<pre>" + html.escape(_help_text()) + "</pre>")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer("<pre>" + html.escape(_help_text()) + "</pre>")


@router.message(Command("ping"))
async def cmd_ping(message: Message) -> None:
    await message.answer("pong")


@router.message(Command("chart"))
async def cmd_chart(message: Message, app: AppState) -> None:
    chart_path = app.cfg.cache_dir / "chart.png"
    if not chart_path.exists():
        await message.answer("График ещё не построен. Подождите пару циклов.")
        return
    await message.answer_photo(FSInputFile(str(chart_path)))


@router.message(Command("last"))
async def cmd_last(message: Message, app: AppState) -> None:
    bars = app.cache.to_list()
    last_bar = bars[-1] if bars else None
    last_close = float(last_bar.close) if last_bar else None
    last_bar_ts = _fmt_ts_z(last_bar.ts) if last_bar else "-"

    last_price = None
    try:
        last_price = await app.tn.get_quote_ltp(app.cfg.symbol)
    except Exception:
        last_price = last_close

    txt = (
        "Последнее:\n\n"
        f"Symbol: {app.cfg.symbol}\n"
        f"Strategy: #{app.strategy_id}\n"
        f"Последняя цена (quote): {'${:.2f}'.format(last_price) if isinstance(last_price, float) else '-'}\n"
        f"Последний бар (UTC): {last_bar_ts}\n"
        f"Close последнего бара: {'{:.4f}'.format(last_close) if isinstance(last_close, float) else '-'}\n"
        f"Кеш: {len(app.cache)} баров"
    )
    await message.answer("<pre>" + html.escape(txt) + "</pre>")


@router.message(Command("dump"))
async def cmd_dump(message: Message, app: AppState) -> None:
    bars = app.cache.to_list()
    if not bars:
        await message.answer("Cache пуст.")
        return

    n = 30
    parts = (message.text or "").split()
    if len(parts) >= 2:
        try:
            n = max(1, min(500, int(parts[1])))
        except Exception:
            n = 30

    tail = bars[-n:]
    lines = [
        f"{_fmt_ts_z(b.ts)} O={b.open:.4f} H={b.high:.4f} L={b.low:.4f} C={b.close:.4f} V={int(b.volume)}"
        for b in tail
    ]
    await message.answer("<pre>" + html.escape("\n".join(lines)) + "</pre>")


@router.message(Command("config"))
async def cmd_config(message: Message, app: AppState) -> None:
    s = app.cfg.signal
    o = app.cfg.option
    lines = [
        f"Symbol:        {app.cfg.symbol}",
        f"Timeframe:     {app.cfg.timeframe_minutes}m",
        f"Strategy:      #{app.strategy_id}",
        f"Cooldown:      {app.cfg.cooldown_seconds}s",
        "",
        "── Strategy #1 ──",
        f"BB:            period={s.bb_period} std={s.bb_std}",
        f"EMA:           fast={s.ema_fast} slow={s.ema_slow}",
        f"RSI:           period={s.rsi_period} buy<{s.rsi_buy} sell>{s.rsi_sell}",
        f"Near BB tol:   {s.near_bb_tol}",
        f"Bounce lookbk: {s.bounce_lookback}",
        "",
        "── Strategy #2 ──",
        f"MACD:          {s.macd_fast}/{s.macd_slow}/{s.macd_signal}",
        f"VWAP mode:     {s.vwap_price_mode}",
        f"Supertrend:    period={s.supertrend_period} mult={s.supertrend_mult}",
        f"ATR:           period={s.atr_period} stop_mult={s.atr_stop_mult}",
        f"EMA trend:     {s.ema_trend_period}",
        "",
        "── Опционы (стр.#1) ──",
        f"Enabled:       {o.enabled}",
        f"Min DTE:       {o.min_dte}",
        f"Strike step:   {o.strike_step}",
        f"Strike offset: {o.strike_offset} ({'ATM' if o.strike_offset == 0 else 'OTM'})",
        f"Underlying:    {o.underlying_symbol}",
    ]
    await message.answer("<pre>" + html.escape("\n".join(lines)) + "</pre>")


@router.message(Command("options"))
async def cmd_options(message: Message, app: AppState) -> None:
    """Показывает текущую опционную позицию и что будет при следующем сигнале."""
    if app.strategy_id != 1:
        await message.answer(
            "Опционные сигналы доступны только для стратегии #1.\n"
            "Переключитесь командой /strategy#1"
        )
        return

    if not app.cfg.option.enabled:
        await message.answer("Опционные сигналы отключены (OPTION_SIGNALS_ENABLED=0).")
        return

    from .options import get_option_recommendation, format_option_message, OptionConfig as _OC

    bars = app.cache.to_list()
    last_price: float | None = None
    if bars:
        last_price = float(bars[-1].close)

    pos = app.option_position
    pos_lines: list[str] = []
    if pos is None:
        pos_lines.append("Текущая позиция: FLAT (нет открытых опционов)")
    else:
        pos_lines.append(f"Текущая позиция: {pos.option_type} OPEN")
        pos_lines.append(f"  Тикер:      {pos.ticker}")
        pos_lines.append(f"  Страйк:     {pos.strike:.0f}")
        pos_lines.append(f"  Экспирация: {pos.expiry.strftime('%d %b %Y')}")
        pos_lines.append(f"  Открыт при: QQQ={pos.entry_underlying:.2f} ({pos.entry_date})")

    # Показываем что произойдёт при следующем BUY и при следующем SELL
    next_lines: list[str] = []
    if last_price is not None:
        oc = app.cfg.option
        opt_cfg = _OC(
            enabled=oc.enabled, min_dte=oc.min_dte,
            strike_step=oc.strike_step, strike_offset=oc.strike_offset,
            underlying_symbol=oc.underlying_symbol,
        )
        for sig in ("BUY", "SELL"):
            try:
                rec = await get_option_recommendation(
                    signal=sig,
                    underlying_price=last_price,
                    cfg=opt_cfg,
                    current_position=pos,
                    session=app.tn.session,
                    api_url=app.cfg.tradernet_api_url,
                    sid=app.cfg.tradernet_sid,
                )
                arrow = "→"
                next_lines.append(
                    f"При {sig} {arrow} {rec.action_type} {rec.option_type} {rec.ticker}"
                )
            except Exception as e:
                next_lines.append(f"При {sig} → ошибка: {e}")

    sig_ts = getattr(app.stats, "last_signal_ts", None)
    last_sig = getattr(app.stats, "last_signal", None) or "нет"

    txt_parts = [
        f"QQQ сейчас: {last_price:.2f}" if last_price else "QQQ: н/д",
        f"Последний сигнал: {last_sig} @ {_fmt_ts_z(sig_ts)}",
        "",
        *pos_lines,
    ]
    if next_lines:
        txt_parts += ["", "Следующий сигнал:", *next_lines]

    await message.answer("<pre>" + html.escape("\n".join(txt_parts)) + "</pre>")


@router.message(Command("status"))
async def cmd_status(message: Message, app: AppState) -> None:
    bars = app.cache.to_list()
    s = app.cfg.signal

    cd_left = _cooldown_left_seconds(app)
    cd_txt = f"{app.cfg.cooldown_seconds}s" + (f" (ещё {cd_left}s)" if cd_left > 0 else "")

    min_b = min_bars_for_indicators(app.cfg)
    last_bar = bars[-1] if bars else None
    last_bar_ts = _fmt_ts_z(last_bar.ts) if last_bar else "-"
    last_close = float(last_bar.close) if last_bar else None

    last_price = None
    try:
        last_price = await app.tn.get_quote_ltp(app.cfg.symbol)
    except Exception:
        last_price = last_close

    df = pd.DataFrame()
    if bars:
        tail = bars[-max(app.cfg.chart_bars, min_b + 10):]
        df = bars_to_df(tail)
        df = add_indicators(df, app.cfg)

    decision_txt = "Нет данных."
    if len(df) >= min_b and len(df) >= 2:
        df_sig = df.iloc[:-1].copy()
        df_sig = add_indicators(df_sig, app.cfg)
        # Передаём КОПИЮ состояния — не мутируем app.strategy2
        dec = compute_signal(
            df_sig, s, strategy_id=app.strategy_id,
            state=Strategy2State(**app.strategy2.__dict__),
        )
        decision_txt = dec.reason
    elif len(df) > 0:
        decision_txt = f"Недостаточно данных: нужно {min_b}, есть {len(df)}."

    pos_txt = _position_from_history(app)

    hist = getattr(app.stats, "signal_history", None) or []
    tail_hist = hist[-5:]

    def _fmt_sig(item: Any) -> str:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            return str(item)
        a, t = item[0], item[1]
        p = item[2] if len(item) >= 3 else None
        p_txt = f" @ {p:.2f}" if isinstance(p, float) else ""
        return f"{a} @ {_fmt_ts_z(t)}{p_txt}"

    sig_hist_txt = (
        "Сигналы текущей сессии: -"
        if not tail_hist
        else "Сигналы текущей сессии:\n- " + "\n- ".join(_fmt_sig(x) for x in tail_hist)
    )

    last_sig = getattr(app.stats, "last_signal", None) or "-"
    last_sig_ts = _fmt_ts_z(getattr(app.stats, "last_signal_ts", None))
    last_sig_price = getattr(app.stats, "last_signal_price", None)

    txt = (
        "Статус бота:\n\n"
        f"Символ:          {app.cfg.symbol}\n"
        f"TF:              {app.cfg.timeframe_minutes}m\n"
        f"Strategy:        #{app.strategy_id}\n"
        f"SessionID:       {getattr(app.stats, 'session_id', None) or '-'}\n"
        f"Сессия:          {'ОТКРЫТА' if getattr(app.stats, 'session_state', None) == 'OPEN' else 'ЗАКРЫТА'}\n\n"
        f"Cooldown:        {cd_txt}\n"
        f"Кеш:             {len(app.cache)} баров (мин. {min_b})\n\n"
        f"Последняя ошибка: {getattr(app.stats, 'last_error', None) or '-'}\n\n"
        f"Последняя цена:  {'${:.2f}'.format(last_price) if isinstance(last_price, float) else '-'}\n"
        f"Последний бар:   {last_bar_ts}\n"
        f"Последний сигнал:{last_sig} @ {last_sig_ts}\n"
        f"Цена по сигналу: {'{:.2f}'.format(last_sig_price) if isinstance(last_sig_price, float) else '-'}\n"
        f"Позиция:         {pos_txt}\n"
        f"{sig_hist_txt}\n\n"
        f"Причина:\n- {decision_txt}\n"
    )
    await message.answer("<pre>" + html.escape(txt) + "</pre>")


@router.message(Command("stats"))
async def cmd_stats(message: Message, app: AppState) -> None:
    cd_left = _cooldown_left_seconds(app)
    cd_txt = f"{app.cfg.cooldown_seconds}s" + (f" (ещё {cd_left}s)" if cd_left > 0 else "")

    txt = (
        "Статистика:\n\n"
        f"Strategy:   #{app.strategy_id}\n"
        f"SessionID:  {getattr(app.stats, 'session_id', None) or '-'}\n"
        f"Ticks:      {app.stats.ticks}\n"
        f"Bars:       {len(app.cache)} "
        f"(real={getattr(app.stats, 'bars_real', 0)}, synth={getattr(app.stats, 'bars_synth', 0)})\n"
        f"Signals:    BUY={getattr(app.stats, 'signals_buy', 0)}, SELL={getattr(app.stats, 'signals_sell', 0)}\n"
        f"Cooldown:   {cd_txt}, skips={getattr(app.stats, 'cooldown_skips', 0)}\n"
        f"Session:    {getattr(app.stats, 'session_state', None) or '-'}\n"
        f"Now(UTC):   {_fmt_ts_z(getattr(app.stats, 'now_utc', None))}\n"
        f"Last error: {getattr(app.stats, 'last_error', None) or '-'}"
    )
    await message.answer("<pre>" + html.escape(txt) + "</pre>")


@router.message(F.text.startswith("/strategy"))
async def cmd_strategy_any(message: Message, app: AppState) -> None:
    txt = (message.text or "").strip()
    m = re.search(r"^/strategy(?:#|_)?\s*([12])\s*$", txt)
    if m:
        sid = int(m.group(1))
        app.set_strategy(sid)
        await message.answer(f"OK. Активная стратегия: #{sid}")
        return
    m2 = re.search(r"^/strategy\s+([12])\s*$", txt)
    if m2:
        sid = int(m2.group(1))
        app.set_strategy(sid)
        await message.answer(f"OK. Активная стратегия: #{sid}")
        return
    await message.answer(
        "<pre>"
        + html.escape(
            f"Текущая стратегия: #{app.strategy_id}\n"
            "Переключение: /strategy#1 или /strategy#2"
        )
        + "</pre>"
    )
