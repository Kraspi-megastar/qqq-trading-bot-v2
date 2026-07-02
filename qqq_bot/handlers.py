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
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.types import Message, FSInputFile

from .scheduler import AppState
from .pipeline import bars_to_df, add_indicators, min_bars_for_indicators
from .signals import compute_signal, Strategy2State
from .utils_time import utc_now

router = Router()

def _option_window_desc(app) -> str:
    """Описание текущего окна торговли опционами для /status."""
    try:
        from . import market_hours as mh
        from .scheduler import _option_cfg
        from .utils_time import utc_now
        ocfg = _option_cfg(app.cfg)
        desc = mh.describe_window(utc_now(), app.cfg.display_tz, ocfg)
        pend = app.pending_close
        if pend:
            desc += f" | отложенное закрытие: {pend}"
        return desc
    except Exception as e:
        return f"n/a ({e})"




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
        "/options — текущая опционная позиция\n"
        "/trades [N] — последние N закрытых сделок\n"
        "/dayreport — итоги дня по опционам\n"
        "/optest [ticker] — диагностика котировки опциона\n"
        "/consensus — текущий консенсус трёх источников\n"
        "\n<b>Исполнение (боевое):</b>\n"
        "/exec_status — статус исполнения\n"
        "/account — сводка по счёту\n"
        "/orders — активные ордера\n"
        "/halt — стоп-кран (блокировать)\n"
        "/resume — снять стоп-кран\n"
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
        "── Опционы (стр.#1 и #2) ──",
        f"Enabled:       {o.enabled}",
        f"Min DTE:       {o.min_dte}",
        f"Strike step:   {o.strike_step}",
        f"Target delta:  {o.target_delta}",
        f"Max exp tries: {o.max_expiry_tries}",
        f"Underlying:    {o.underlying_symbol}",
    ]
    await message.answer("<pre>" + html.escape("\n".join(lines)) + "</pre>")


@router.message(Command("options"))
async def cmd_options(message: Message, app: AppState) -> None:
    """Показывает текущую опционную позицию и что будет при следующем сигнале."""
    # Опционы работают для стратегии #1 (CALL + PUT) и #2 (только CALL, без шорта)

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
        pos_lines.append(f"  TraderNet:  {pos.tn_ticker}")
        pos_lines.append(f"  Страйк:     {pos.strike:.0f}")
        pos_lines.append(f"  Экспирация: {pos.expiry.strftime('%d %b %Y')}")
        pos_lines.append(f"  Открыт при: QQQ={pos.entry_underlying:.2f} ({pos.entry_date})")

    # Показываем что произойдёт при следующем BUY и при следующем SELL
    next_lines: list[str] = []
    if last_price is not None:
        oc = app.cfg.option
        opt_cfg = _OC(
            enabled=oc.enabled, min_dte=oc.min_dte,
            strike_step=oc.strike_step,
            underlying_symbol=oc.underlying_symbol,
            target_delta=oc.target_delta,
            max_expiry_tries=oc.max_expiry_tries,
            risk_free_rate=oc.risk_free_rate,
        )
        # ATR из последнего бара (для оценки волатильности)
        atr_v = None
        try:
            tail = app.cache.to_list()[-(app.cfg.atr_period_safe if hasattr(app.cfg, 'atr_period_safe') else 50):]
            import pandas as pd
            from .pipeline import bars_to_df, add_indicators
            dfa = add_indicators(bars_to_df(tail), app.cfg)
            if len(dfa) and "atr" in dfa.columns and pd.notna(dfa["atr"].iloc[-1]):
                atr_v = float(dfa["atr"].iloc[-1])
        except Exception:
            atr_v = None

        for sig in ("BUY", "SELL"):
            try:
                rec = await get_option_recommendation(
                    signal=sig,
                    underlying_price=last_price,
                    cfg=opt_cfg,
                    current_position=pos,
                    market_open=True,    # в /options показываем как было бы в RTH
                    atr=atr_v,
                    tn=app.tn,
                )
                arrow = "→"
                d_str = f" Δ={rec.delta:.2f}" if rec.delta is not None else ""
                next_lines.append(
                    f"При {sig} {arrow} {rec.action_type} {rec.option_type} {rec.ticker}{d_str}"
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
        f"Опц. окно:       {_option_window_desc(app)}\n"
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


@router.message(Command("trades"))
async def cmd_trades(message: Message, app: AppState) -> None:
    """Последние закрытые опционные сделки."""
    if app.trade_journal is None:
        await message.answer("Журнал сделок недоступен.")
        return

    parts = (message.text or "").split()
    n = 10
    if len(parts) >= 2:
        try:
            n = max(1, min(50, int(parts[1])))
        except Exception:
            pass

    trades = app.trade_journal.closed_trades(limit=n)
    open_t = app.trade_journal.any_open()

    if not trades and open_t is None:
        await message.answer("Закрытых сделок нет.")
        return

    lines = [f"Последние закрытые опционные сделки QQQ: {len(trades)}", ""]
    for t in trades:
        entry_dt = t.entry_ts_dt()
        date_str = entry_dt.strftime("%Y-%m-%d") if entry_dt else "?"
        entry_str = f"{t.entry_price:.4f}" if t.entry_price is not None else "None"
        exit_str  = f"{t.exit_price:.4f}"  if t.exit_price  is not None else "None"
        lines.append(
            f"{date_str} | +{t.ticker} | {t.option_type} | exit | "
            f"DTE={t.dte_at_entry} | entry={entry_str} exit={exit_str} | P/L={t.pnl_str()}"
        )

    if open_t is not None:
        lines += ["", "Открытая позиция:"]
        entry_str = f"{open_t.entry_price}" if open_t.entry_price is not None else "None"
        spot_str  = f"{open_t.entry_underlying:.2f}"
        lines.append(
            f"+{open_t.ticker} | {open_t.option_type} | entry={entry_str} | "
            f"spot={spot_str} | ts={open_t.entry_ts}"
        )

    await message.answer("<pre>" + html.escape("\n".join(lines)) + "</pre>")


@router.message(Command("dayreport"))
async def cmd_dayreport(message: Message, app: AppState) -> None:
    """Дневной отчёт по опционным сделкам."""
    if app.trade_journal is None:
        await message.answer("Журнал сделок недоступен.")
        return

    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(app.cfg.display_tz)
    today_str = datetime.now(tz=tz).date().isoformat()

    closed = app.trade_journal.closed_trades(session_date=today_str, limit=200)
    open_t = app.trade_journal.any_open()

    total = len(closed)
    with_pnl = [t for t in closed if t.pnl() is not None]
    wins  = [t for t in with_pnl if (t.pnl() or 0) > 0]
    losses= [t for t in with_pnl if (t.pnl() or 0) <= 0]

    total_pnl = sum(t.pnl() for t in with_pnl) if with_pnl else 0.0
    avg_pnl   = total_pnl / len(with_pnl) if with_pnl else None
    win_rate  = f"{len(wins)/len(with_pnl)*100:.0f}%" if with_pnl else "n/a"

    lines = [
        f"Дневной отчёт по опционам QQQ — {today_str}",
        "",
        f"Закрытых сделок: {total} (с P/L: {len(with_pnl)})",
        f"Win/Loss: {len(wins)} / {len(losses)}",
        f"Win rate: {win_rate}",
        f"Итоговый P/L: ${total_pnl:.2f}",
        f"Средний P/L: {'${:.2f}'.format(avg_pnl) if avg_pnl is not None else 'n/a'}",
    ]

    if open_t is not None:
        entry_str = f"{open_t.entry_price}" if open_t.entry_price is not None else "None"
        spot_str  = f"{open_t.entry_underlying:.2f}"
        lines += [
            "",
            "Открытая позиция:",
            f"  +{open_t.ticker} {open_t.option_type} "
            f"entry={entry_str} spot={spot_str} ts={open_t.entry_ts}",
        ]

    if not closed and open_t is None:
        lines.append("")
        lines.append("Сделок за дату нет или они ещё не закрыты.")

    await message.answer("<pre>" + html.escape("\n".join(lines)) + "</pre>")


@router.message(Command("optest"))
async def cmd_optest(message: Message, app: AppState) -> None:
    """Диагностика: показывает сырой ответ TraderNet на запрос котировки опциона.
    Использование: /optest +QQQ.31JUL2026.C732
    Без аргумента — строит тикер ATM CALL на ближайшую пятницу."""
    from .options import tradernet_option_ticker
    from datetime import date, timedelta

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) >= 2 and parts[1].strip():
        ticker = parts[1].strip()
    else:
        bars = app.cache.to_list()
        spot = float(bars[-1].close) if bars else 500.0
        today = date.today()
        base = today + timedelta(days=1)
        friday = base + timedelta(days=(4 - base.weekday()) % 7)
        strike = round(spot)
        ticker = tradernet_option_ticker("CALL", strike, friday)

    raw = await app.tn.get_option_quote_raw(ticker)
    try:
        parsed = await app.tn.get_option_quote(ticker)
        parsed_str = str(parsed)
    except Exception as e:
        parsed_str = f"ошибка парсинга: {e!r}"

    txt = (
        f"Тикер: {ticker}\n\n"
        f"Распарсено: {parsed_str}\n\n"
        f"Сырой ответ:\n{raw}"
    )
    await message.answer("<pre>" + html.escape(txt) + "</pre>")


# ══════════════════════════════════════════════════════════════════════════
# БОЕВОЕ ИСПОЛНЕНИЕ — команды и подтверждение через кнопки
# ══════════════════════════════════════════════════════════════════════════

def build_account_confirm_row(account_id: str, label: str, token: str) -> list:
    """Ряд кнопок подтверждения для конкретного счёта."""
    return [[
        InlineKeyboardButton(text=f"✅ {label}", callback_data=f"exec_ok:{account_id}:{token}"),
        InlineKeyboardButton(text=f"❌ {label}", callback_data=f"exec_no:{account_id}:{token}"),
    ]]


def build_confirm_keyboard(token: str, account_id: str = "ffa", label: str = "FFA") -> InlineKeyboardMarkup:
    """Совместимость: клавиатура на один счёт."""
    return InlineKeyboardMarkup(inline_keyboard=build_account_confirm_row(account_id, label, token))


def _find_broker(app: AppState, account_id: str):
    """Находит Broker по account_id."""
    for b in getattr(app, "brokers", []) or []:
        if getattr(b.cfg, "account_id", None) == account_id:
            return b
    return None


@router.callback_query(F.data.startswith("exec_ok:"))
async def cb_exec_confirm(cq: CallbackQuery, app: AppState) -> None:
    """Пользователь подтвердил ордер на конкретном счёте."""
    parts = cq.data.split(":")
    # формат: exec_ok:{account_id}:{token}
    if len(parts) >= 3:
        account_id, token = parts[1], parts[2]
    else:
        account_id, token = "ffa", parts[1]  # старый формат

    broker = _find_broker(app, account_id)
    if broker is None:
        await cq.answer("Счёт не найден", show_alert=True)
        return
    po = broker.get_pending(token)
    if po is None:
        await cq.answer("Ордер не найден или истёк", show_alert=True)
        return
    res = broker.confirm_pending(token)
    if res.ok:
        txt = f"✅ <b>[{broker.cfg.label}] Ордер исполнен</b>\n{res.reason}"
        if res.order_id:
            txt += f"\norder_id: {res.order_id}"
        await cq.answer(f"{broker.cfg.label}: исполнено")
    else:
        txt = f"⚠️ <b>[{broker.cfg.label}] Не исполнено</b>\n{res.reason}"
        await cq.answer("Ошибка", show_alert=True)
    try:
        await cq.message.edit_text(cq.message.html_text + f"\n\n{txt}", parse_mode="HTML")
    except Exception:
        await cq.message.answer(txt)


@router.callback_query(F.data.startswith("exec_no:"))
async def cb_exec_reject(cq: CallbackQuery, app: AppState) -> None:
    """Пользователь отклонил ордер на конкретном счёте."""
    parts = cq.data.split(":")
    if len(parts) >= 3:
        account_id, token = parts[1], parts[2]
    else:
        account_id, token = "ffa", parts[1]
    broker = _find_broker(app, account_id)
    if broker is not None:
        broker.reject_pending(token)
    label = broker.cfg.label if broker else account_id
    await cq.answer(f"{label}: отклонено")
    try:
        await cq.message.edit_text(
            cq.message.html_text + f"\n\n❌ <b>[{label}] Отклонено</b>", parse_mode="HTML")
    except Exception:
        pass


@router.message(Command("exec_status"))
async def cmd_exec_status(message: Message, app: AppState) -> None:
    """Статус боевого исполнения по всем счетам."""
    brokers = getattr(app, "brokers", []) or []
    if not brokers:
        await message.answer("Брокеры не инициализированы (EXEC_ACCOUNTS/ключи не заданы).")
        return
    lines = ["<b>Боевое исполнение</b>"]
    for b in brokers:
        ex = b.cfg
        lines.append("")
        lines.append(f"<b>[{ex.label}]</b> ({ex.account_id})")
        lines.append(f"  Включено: {'да' if ex.enabled else 'нет'} | режим: {ex.mode}")
        lines.append(f"  Клиент: {'загружен' if b.available else 'недоступен'}"
                     + (" | ⛔ HALT" if b.halted else ""))
        if b.load_error:
            lines.append(f"  Ошибка: {html.escape(str(b.load_error))}")
        lines.append(f"  Размер: {ex.position_pct}% (потолок {ex.max_position_pct}%), "
                     f"макс {ex.max_contracts} контр.")
        lines.append(f"  Ночь при DTE≥{ex.hold_overnight_min_dte}, 0DTE-блок ≤{ex.block_new_position_if_dte_lte}")
    await message.answer("\n".join(lines))


@router.message(Command("account"))
async def cmd_account(message: Message, app: AppState) -> None:
    """Сводка по всем счетам (только чтение)."""
    brokers = getattr(app, "brokers", []) or []
    avail = [b for b in brokers if b.available]
    if not avail:
        await message.answer("Нет доступных брокеров (EXEC_ENABLED=0 или нет ключей).")
        return
    lines = ["<b>Счета</b>"]
    for b in avail:
        ex = b.cfg
        lines.append("")
        lines.append(f"<b>[{ex.label}]</b>")
        pp = b.purchasing_power()
        if pp is not None:
            budget = pp * min(ex.position_pct, ex.max_position_pct) / 100.0
            lines.append(f"  Свободно USD: ${pp:,.2f}")
            lines.append(f"  Бюджет на сделку ({ex.position_pct}%): ${budget:,.2f}")
        else:
            lines.append("  Свободный остаток: не удалось определить")
        try:
            positions = b.get_positions()
            if positions:
                lines.append(f"  Позиции: {len(positions)}")
                for p in positions[:10]:
                    lines.append(f"    {p.get('i','?')} | q={p.get('q')} | mkt={p.get('mkt_price', p.get('market_value','?'))}")
        except Exception:
            pass
    await message.answer("\n".join(lines))


@router.message(Command("orders"))
async def cmd_orders(message: Message, app: AppState) -> None:
    """Активные ордера по всем счетам (stat==10)."""
    brokers = [b for b in (getattr(app, "brokers", []) or []) if b.available]
    if not brokers:
        await message.answer("Нет доступных брокеров.")
        return
    lines = []
    for b in brokers:
        orders = b.get_active_orders()
        lines.append(f"<b>[{b.cfg.label}]</b> активных: {len(orders)}")
        for o in orders[:15]:
            oper = "BUY" if o.get("oper") == 1 else "SELL" if o.get("oper") == 2 else f"op{o.get('oper')}"
            lines.append(f"  {o.get('order_id')} | {o.get('instr','?')} | {oper} | q={o.get('q')} @ {o.get('p')}")
    await message.answer("<pre>" + html.escape("\n".join(lines)) + "</pre>")


@router.message(Command("positions"))
async def cmd_positions(message: Message, app: AppState) -> None:
    """Открытые позиции по всем счетам (только чтение)."""
    brokers = [b for b in (getattr(app, "brokers", []) or []) if b.available]
    if not brokers:
        await message.answer("Нет доступных брокеров.")
        return
    lines = []
    for b in brokers:
        positions = b.get_positions()
        lines.append(f"<b>[{b.cfg.label}]</b> позиций: {len(positions)}")
        for p in positions[:15]:
            lines.append(f"  {p.get('i','?')} | q={p.get('q')} | mkt={p.get('mkt_price', p.get('market_value','?'))} | P/L={p.get('profit_close','?')}")
    await message.answer("<pre>" + html.escape("\n".join(lines)) + "</pre>")


@router.message(Command("halt"))
async def cmd_halt(message: Message, app: AppState) -> None:
    """Стоп-кран. /halt — все счета; /halt tfos — конкретный счёт."""
    brokers = getattr(app, "brokers", []) or []
    if not brokers:
        await message.answer("Брокеры не инициализированы.")
        return
    arg = (message.text or "").split(maxsplit=1)
    target = arg[1].strip().lower() if len(arg) > 1 else None
    halted = []
    for b in brokers:
        if target is None or b.cfg.account_id == target:
            b.halt()
            halted.append(b.cfg.label)
    if halted:
        await message.answer(f"⛔ <b>СТОП-КРАН</b> для: {', '.join(halted)}\nИсполнение заблокировано. /resume для снятия.")
    else:
        await message.answer(f"Счёт '{target}' не найден.")


@router.message(Command("resume"))
async def cmd_resume(message: Message, app: AppState) -> None:
    """Снятие стоп-крана. /resume — все; /resume tfos — конкретный."""
    brokers = getattr(app, "brokers", []) or []
    if not brokers:
        await message.answer("Брокеры не инициализированы.")
        return
    arg = (message.text or "").split(maxsplit=1)
    target = arg[1].strip().lower() if len(arg) > 1 else None
    resumed = []
    for b in brokers:
        if target is None or b.cfg.account_id == target:
            b.resume()
            resumed.append(b.cfg.label)
    if resumed:
        await message.answer(f"✅ Стоп-кран снят для: {', '.join(resumed)}")
    else:
        await message.answer(f"Счёт '{target}' не найден.")


@router.message(Command("consensus"))
async def cmd_consensus(message: Message, app: AppState) -> None:
    """Показывает текущий консенсус трёх источников."""
    if not getattr(app.cfg, "consensus", None) or not app.cfg.consensus.enabled:
        await message.answer("Консенсус выключен (CONSENSUS_ENABLED=0).")
        return
    try:
        from .consensus import compute_consensus, format_consensus
        res = compute_consensus(app.consensus_state, app.cfg.consensus)
        txt = format_consensus(res)
        ml_status = "загружен" if app.ml_service is not None else "не загружен (#1+#2)"
        txt += f"\n\nML: {ml_status}"
        txt += f"\nОкно согласия: {app.cfg.consensus.agree_window_bars} баров"
        await message.answer(txt)
    except Exception as e:
        await message.answer(f"Ошибка консенсуса: {html.escape(str(e))}")


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
