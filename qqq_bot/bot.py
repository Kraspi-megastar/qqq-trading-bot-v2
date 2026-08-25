"""
bot.py — точка входа.

Опционная рекомендация (rec) вычисляется в scheduler.polling_loop ДО отправки,
и передаётся сюда готовой. Здесь только:
  - запись сделки в журнал (с реальной ценой опциона из TraderNet),
  - применение новой позиции,
  - форматирование и отправка сообщения.

Сообщение приходит сюда ТОЛЬКО когда позиция реально меняется (OPEN/CLOSE) —
HOLD и пропуски из-за закрытого рынка отсекаются в scheduler.
"""
from __future__ import annotations

import asyncio
import html
from datetime import datetime, timezone

import aiohttp
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import FSInputFile

from .config import load_config
from .cache import BarCache, Stats
from .tradernet import TraderNetClient
from .scheduler import AppState, bootstrap_history, polling_loop
from .handlers import router
from .signals import SignalDecision
from .options import OptionRecommendation, format_option_message
from .trades import TradeJournal


def _fmt_ts_z(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt is not None else "-"


async def _fetch_option_price(tn: TraderNetClient, tn_ticker: str) -> float | None:
    try:
        q = await asyncio.wait_for(tn.get_option_quote(tn_ticker), timeout=8.0)
        if q is None:
            return None
        # предпочитаем mid(bid,ask), иначе ltp
        bid, ask, ltp = q.get("bid"), q.get("ask"), q.get("ltp")
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return round((bid + ask) / 2, 4)
        return ltp
    except Exception:
        return None


async def _send_signal_to_channel(
    bot: Bot,
    app: AppState,
    decision: SignalDecision,
    chart_path: str,
    df_sig,
    rec: OptionRecommendation | None,
    session: aiohttp.ClientSession,
    extra_text: str | None = None,
    consensus_res=None,
) -> None:
    import pandas as pd

    s = app.cfg.signal
    direction = "🟢 BUY" if decision.action == "BUY" else "🔴 SELL"

    last = df_sig.iloc[-1] if df_sig is not None and len(df_sig) > 0 else None

    def _fv(col: str) -> float | None:
        if last is None or col not in last:
            return None
        v = last[col]
        return float(v) if pd.notna(v) else None

    close  = _fv("close")
    ts     = last["ts"] if last is not None else None
    rsi_v  = _fv("rsi")
    ema_f  = _fv("ema_fast")
    ema_sl = _fv("ema_slow")
    bb_l   = _fv("bb_lower")
    bb_m   = _fv("bb_mid")
    bb_u   = _fv("bb_upper")

    lines = [
        f"<b>{html.escape(direction)}</b>",
        f"<b>{html.escape(app.cfg.symbol)}</b>  TF={app.cfg.timeframe_minutes}m  STR=#{app.strategy_id}",
        html.escape(f"BarTS={_fmt_ts_z(ts)}"),
        html.escape(f"Close={close:.4f}" if isinstance(close, float) else "Close=n/a"),
        html.escape(f"RSI({s.rsi_period})={rsi_v:.2f}" if isinstance(rsi_v, float) else f"RSI({s.rsi_period})=n/a"),
        html.escape(
            f"EMA{s.ema_fast}={ema_f:.2f} | EMA{s.ema_slow}={ema_sl:.2f}"
            if isinstance(ema_f, float) and isinstance(ema_sl, float) else "EMA=n/a"
        ),
        html.escape(
            f"BB({s.bb_period},{s.bb_std}): L={bb_l:.2f} M={bb_m:.2f} U={bb_u:.2f}"
            if isinstance(bb_l, float) else "BB=n/a"
        ),
    ]

    if app.strategy_id == 2 and app.strategy2.atr_stop is not None:
        lines.append(html.escape(f"ATR-Stop={app.strategy2.atr_stop:.2f}"))

    lines += ["", f"<i>{html.escape(decision.reason)}</i>"]

    # ── Опционная часть ────────────────────────────────────────────────────
    _pnl_str_for_public = None
    if rec is not None:
        now_utc = datetime.now(tz=timezone.utc)
        session_date = getattr(app.stats, "session_id", None) or now_utc.date().isoformat()

        if app.trade_journal is not None:
            if rec.action_type == "OPEN":
                opt_price = await _fetch_option_price(app.tn, rec.tn_ticker)
                app.trade_journal.open_trade(
                    session_date=session_date,
                    option_type=rec.option_type,
                    ticker=rec.tn_ticker,
                    strike=rec.strike,
                    expiry=rec.expiry,
                    dte_at_entry=rec.dte,
                    entry_price=opt_price,
                    entry_underlying=rec.underlying_price,
                    entry_ts=now_utc,
                )
            elif rec.action_type == "CLOSE" and app.option_position is not None:
                opt_price = await _fetch_option_price(app.tn, rec.tn_ticker)
                closed = app.trade_journal.close_trade(
                    ticker=rec.tn_ticker,
                    exit_price=opt_price,
                    exit_underlying=rec.underlying_price,
                    exit_ts=now_utc,
                )
                if closed is not None:
                    _pnl_str_for_public = closed.pnl_str()
                    lines.append(html.escape(f"P&L сделки: {_pnl_str_for_public}"))

        # Применяем новую позицию
        # Если позиция ЗАКРЫЛАСЬ (new_position=None при CLOSE) — разрешаем следующему
        # сигналу того же направления поставить новый треугольник (повторный вход).
        if rec.action_type == "CLOSE" and rec.new_position is None:
            app._allow_reentry_mark = True
        app.option_position = rec.new_position

        lines += ["", format_option_message(rec)]

    if extra_text:
        lines += ["", extra_text]

    # ── Боевое исполнение (semi-auto): предложить ордер с кнопкой ─────────────
    exec_keyboard = None
    if (app.brokers and rec is not None and rec.action_type in ("OPEN", "CLOSE")):
        try:
            price = await _fetch_option_price(app.tn, rec.tn_ticker)
            if price and price > 0:
                keyboard_rows = []
                any_offer = False
                for br in app.brokers:
                    if not br.available or br.cfg.mode != "semi_auto":
                        continue
                    label = br.cfg.label
                    if rec.action_type == "OPEN":
                        acct_val = br.purchasing_power() or 0.0
                        # Размер берём из настроек чата (runtime), а не из статичного .env.
                        # Настройка пользователя приоритетна; жёсткий потолок брокера
                        # (max_position_pct) применяем только если он ВЫШЕ нуля и реально
                        # ниже запрошенного — тогда сообщаем об этом явно.
                        rs = getattr(app, "settings", None)
                        cap_note = ""
                        if rs is not None:
                            want_pct = float(rs.position_pct)
                            hard_cap = float(getattr(br.cfg, "max_position_pct", 100.0) or 100.0)
                            eff_pct = min(want_pct, hard_cap)
                            if eff_pct < want_pct:
                                cap_note = (f" (запрошено {want_pct:.0f}%, "
                                            f"ограничено потолком {hard_cap:.0f}%)")
                            budget = acct_val * (eff_pct / 100.0)
                            per_contract = max(price * 100, 1e-9)
                            contracts = int(budget // per_contract)
                            contracts = max(0, min(contracts, int(rs.max_contracts)))
                        else:
                            contracts = br.calc_contracts(price, acct_val)
                        side = "BUY"
                    else:  # CLOSE
                        # Закрытие = продать то, что БЫЛО КУПЛЕНО при открытии.
                        # И CALL, и PUT открываются через BUY, поэтому закрытие
                        # ВСЕГДА SELL. Количество берём из реальной позиции у брокера
                        # (не хардкодим), чтобы закрыть ровно столько, сколько открыто.
                        side = "SELL"
                        contracts = 0
                        try:
                            for p in br.get_positions():
                                if str(p.get("i", "")) == rec.tn_ticker:
                                    q = int(abs(int(p.get("q", 0))))
                                    contracts = q
                                    break
                        except Exception:
                            contracts = 0
                        # если у брокера позиции нет — закрывать нечего
                        if contracts < 1:
                            lines += ["", f"🎯 <b>{label}</b>: закрывать нечего "
                                          f"(позиции {rec.tn_ticker} нет у брокера)"]
                            continue

                    if contracts >= 1:
                        # Стоп-лосс и принудительное закрытие в конце дня — РЫНОЧНЫМ
                        # приказом (быстрое исполнение, лимитка может зависнуть пока
                        # рынок уходит). Обычные закрытия/открытия — лимитные.
                        use_market = (rec.action_type == "CLOSE"
                                      and getattr(rec, "delta_source", "") in ("stop_loss", "force_close"))
                        po = br.create_pending(
                            tn_ticker=rec.tn_ticker, side=side, contracts=contracts,
                            limit_price=price, dte=rec.dte, is_open=(rec.action_type == "OPEN"),
                            market=use_market,
                        )
                        # callback data: exec_ok:{account_id}:{token}
                        from .handlers import build_account_confirm_row
                        keyboard_rows += build_account_confirm_row(br.cfg.account_id, label, po.token)
                        lines += ["", f"🎯 <b>{label}</b>: {po.human}"]
                        any_offer = True
                    else:
                        # Подробная причина: сколько денег, почём контракт, какой %
                        detail = ""
                        if rec.action_type == "OPEN":
                            per_c = price * 100
                            detail = (f"\nБаланс ${acct_val:.0f}, контракт ~${per_c:.0f}"
                                      f"{cap_note}")
                        lines += ["", f"🎯 <b>{label}</b>: размер = 0 контрактов "
                                      f"(бюджета не хватает на 1 контракт){detail}"]

                if any_offer:
                    from aiogram.types import InlineKeyboardMarkup
                    exec_keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_rows)
                    to = app.brokers[0].cfg.confirm_timeout_sec // 60
                    lines += ["", f"⏳ подтверди в течение {to} мин"]
        except Exception as e:
            lines += ["", html.escape(f"[Исполнение: ошибка подготовки — {e}]")]

    caption = "\n".join(lines)
    await bot.send_photo(
        chat_id=app.cfg.telegram_channel_id,
        photo=FSInputFile(chart_path),
        caption=caption,
        parse_mode=ParseMode.HTML,
        reply_markup=exec_keyboard,
    )

    # ── Второй канал: только сигнал, без исполнения ──────────────────────────
    # Компактный формат для тестировщиков/подписчиков. Ошибка отправки сюда
    # НЕ должна ломать основной канал, поэтому всё в try/except.
    pub_id = getattr(app.cfg, "telegram_public_channel_id", 0)
    if pub_id:
        try:
            pub_caption = _build_public_caption(app, decision, rec, consensus_res,
                                                pnl_str=_pnl_str_for_public)
            await bot.send_photo(
                chat_id=pub_id,
                photo=FSInputFile(chart_path),
                caption=pub_caption,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            app.stats.last_error = f"public_channel: {repr(e)}"


def _build_public_caption(app: AppState, decision: SignalDecision,
                          rec: OptionRecommendation | None, consensus_res=None,
                          pnl_str: str | None = None) -> str:
    """
    Компактное сообщение для публичного канала:
    направление + опцион (с тикером DasTrader) + причина закрытия + P&L + консенсус.
    Без индикаторов, без исполнения, без служебных деталей.
    """
    from .options import format_option_message_public

    if decision.action == "BUY":
        head = "🟢 BUY"
    elif decision.action == "SELL":
        head = "🔴 SELL"
    else:
        head = f"⏸ {decision.action}"

    parts = [head]

    if rec is not None:
        base_symbol = str(getattr(app.cfg, "symbol", "QQQ")).split(".")[0].lower()
        parts.append(format_option_message_public(rec, symbol=base_symbol))

        # Причина закрытия — человекочитаемо (для подписчиков)
        if rec.action_type == "CLOSE":
            reason = getattr(rec, "delta_source", "")
            reason_txt = {
                "stop_loss": "🛑 Закрытие по стоп-лоссу",
                "force_close": "🔚 Закрытие в конце дня (позицию не переносим через ночь)",
                "pending_close": "Закрытие (отложенное)",
            }.get(reason, None)
            if reason_txt:
                parts.append(reason_txt)
            # P&L результат сделки
            if pnl_str:
                parts.append(f"Результат: {pnl_str}")

    if consensus_res is not None:
        try:
            from .consensus import format_consensus_public
            parts += ["", format_consensus_public(consensus_res)]
        except Exception:
            pass

    return "\n".join(parts)


async def _setup_command_menus(bot, app) -> None:
    """
    Задаёт меню команд по областям видимости (scopes).

    Личка владельца — полный набор (включая настройки и исполнение).
    Твои каналы — только информационные команды (для тестировщиков/подписчиков).
    По умолчанию — минимальный набор.
    """
    from aiogram.types import (
        BotCommand,
        BotCommandScopeDefault,
        BotCommandScopeChat,
    )
    from .handlers import OWNER_USER_ID

    def _cmds(pairs):
        return [BotCommand(command=c, description=d) for c, d in pairs]

    # Полное меню (личка владельца)
    full = _cmds([
        ("help", "Список команд"),
        ("status", "Состояние бота и сессия"),
        ("chart", "График с сигналами"),
        ("last", "Последний сигнал"),
        ("options", "Текущая опционная позиция"),
        ("dayreport", "Дневной отчёт"),
        ("trades", "Последние сделки"),
        ("settings", "Настройки (стоп, размер)"),
        ("set_stop", "Стоп-лосс: /set_stop 30"),
        ("set_pct", "Размер позиции: /set_pct 5"),
        ("set_contracts", "Макс контрактов: /set_contracts 2"),
        ("set_htf_filter", "Фильтр тренда 1h: /set_htf_filter on"),
        ("set_htf_slope", "Порог тренда: /set_htf_slope 0.45"),
        ("set_cooldown", "Пауза после стопа: /set_cooldown 15"),
        ("account", "Балансы по счетам"),
        ("positions", "Открытые позиции"),
        ("orders", "Активные ордера"),
        ("exec_status", "Статус исполнения"),
        ("halt", "СТОП-КРАН"),
        ("resume", "Снять стоп-кран"),
        ("stats", "Статистика"),
        ("ping", "Проверка отклика"),
    ])

    # Информационное меню (каналы — тестировщики/подписчики)
    info = _cmds([
        ("help", "Список команд"),
        ("status", "Состояние бота"),
        ("chart", "График с сигналами"),
        ("last", "Последний сигнал"),
        ("options", "Текущая позиция"),
        ("trades", "Последние сделки"),
        ("dayreport", "Дневной отчёт"),
        ("stats", "Статистика"),
    ])

    # Минимум по умолчанию
    default = _cmds([
        ("help", "Список команд"),
        ("status", "Состояние бота"),
    ])

    # 1) дефолт везде
    await bot.set_my_commands(default, scope=BotCommandScopeDefault())

    # 2) полное меню — в личке владельца
    try:
        await bot.set_my_commands(full, scope=BotCommandScopeChat(chat_id=OWNER_USER_ID))
    except Exception as e:
        app.stats.last_error = f"menu owner: {repr(e)}"

    # 3) информационное меню — в твоих каналах (по chat_id)
    for ch in (getattr(app.cfg, "telegram_channel_id", 0),
               getattr(app.cfg, "telegram_public_channel_id", 0)):
        if ch:
            try:
                await bot.set_my_commands(info, scope=BotCommandScopeChat(chat_id=ch))
            except Exception as e:
                app.stats.last_error = f"menu chat {ch}: {repr(e)}"


async def _amain() -> None:
    cfg = load_config()

    bot = Bot(token=cfg.telegram_bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)

    cache = BarCache(timeframe_minutes=cfg.timeframe_minutes, maxlen=max(cfg.chart_bars * 5, 2000))
    stats = Stats()

    async with aiohttp.ClientSession() as session:
        tn = TraderNetClient(
            api_url=cfg.tradernet_api_url,
            quotes_url=cfg.tradernet_quotes_url,
            session=session,
            sid=cfg.tradernet_sid,
            timeout_seconds=cfg.tradernet_timeout_seconds,
        )
        app = AppState(cfg=cfg, tn=tn, cache=cache, stats=stats)
        app.strategy_id = cfg.strategy_id
        app.trade_journal = TradeJournal(cfg.cache_dir)

        # Настройки из чата (стоп-лосс, размер позиции) — переживают рестарт.
        # Дефолты берём из первого exec-аккаунта (если есть), иначе стандартные.
        from .runtime_settings import RuntimeSettings, load_settings
        _def = RuntimeSettings()
        if cfg.executions:
            _ex0 = cfg.executions[0]
            _def = RuntimeSettings(
                stop_loss_pct=0.0,
                position_pct=_ex0.position_pct,
                max_contracts=_ex0.max_contracts,
            )
        app.settings = load_settings(cfg.cache_dir, _def)

        # ML-сервис удалён при чистке (advisory-only, не влиял на торговлю).

        # Брокеры по счетам (ffa, tfos, ...). По умолчанию всё выключено.
        # Изолировано: ошибка инициализации не влияет на сигнальную работу.
        app.brokers = []
        try:
            from .broker import Broker, ExecutionConfig as _EC
            for ex in cfg.executions:
                bcfg = _EC(
                    enabled=ex.enabled, account_id=ex.account_id, label=ex.label,
                    mode=ex.mode, public_key=ex.public_key, private_key=ex.private_key,
                    position_pct=ex.position_pct, max_position_pct=ex.max_position_pct,
                    max_contracts=ex.max_contracts, max_orders_per_day=ex.max_orders_per_day,
                    max_notional_per_trade=ex.max_notional_per_trade,
                    hold_overnight_min_dte=ex.hold_overnight_min_dte,
                    block_new_position_if_dte_lte=ex.block_new_position_if_dte_lte,
                    require_reconcile=ex.require_reconcile,
                    confirm_timeout_sec=ex.confirm_timeout_sec,
                    underlying_symbol=ex.underlying_symbol,
                )
                br = Broker(bcfg)
                app.brokers.append(br)
                if ex.enabled and br.load_error:
                    app.stats.last_error = f"Broker[{ex.account_id}] load: {br.load_error}"
            # совместимость: app.broker = первый включённый
            app.broker = next((b for b in app.brokers if b.available), 
                              app.brokers[0] if app.brokers else None)
        except Exception as e:
            app.stats.last_error = f"Broker init: {repr(e)}"
            app.brokers = []
            app.broker = None

        dp["app"] = app

        asyncio.create_task(bootstrap_history(app))

        async def sender(decision: SignalDecision, chart_path: str, df_sig, rec=None,
                         extra_text=None, consensus_res=None) -> None:
            await _send_signal_to_channel(bot, app, decision, chart_path, df_sig, rec,
                                          session, extra_text, consensus_res)

        # RTH-открытие: сводка настроек ТОЛЬКО в закрытый (основной) канал
        async def on_rth_open():
            try:
                from .runtime_settings import format_settings
                if getattr(app, "settings", None) is None:
                    return
                txt = "🔔 <b>Старт торговой сессии</b>\n\n" + format_settings(app.settings)
                await bot.send_message(chat_id=app.cfg.telegram_channel_id, text=txt,
                                       parse_mode=ParseMode.HTML)
            except Exception as e:
                app.stats.last_error = f"on_rth_open: {repr(e)}"

        # RTH-закрытие: авто-dayreport в ОБА канала (полный/публичный)
        async def on_rth_close():
            try:
                from .trades import build_day_report, build_day_report_public
                # основной канал — полный отчёт
                full = build_day_report(app)
                await bot.send_message(chat_id=app.cfg.telegram_channel_id, text=full,
                                       parse_mode=ParseMode.HTML)
                # публичный канал — версия «на 1 контракт» ($ и %)
                pub_id = getattr(app.cfg, "telegram_public_channel_id", 0)
                if pub_id:
                    pub = build_day_report_public(app)
                    await bot.send_message(chat_id=pub_id, text=pub, parse_mode=ParseMode.HTML)
                # Бумажная #2 — итог дня (только в закрытый канал)
                if getattr(app, "paper_s2_state", None) is not None and \
                        getattr(app.settings, "paper_s2_on", False):
                    try:
                        from .paper_s2 import format_paper_day_summary, process_paper_signal
                        pst = app.paper_s2_state
                        # если к концу дня осталась открытая виртуальная позиция —
                        # закрываем её по последней цене ДО подсчёта итога, иначе
                        # сделка не попадёт в статистику дня.
                        if pst.position is not None:
                            last_price = None
                            try:
                                if len(app.cache) > 0:
                                    last_price = float(app.cache.to_list()[-1].get("close"))
                            except Exception:
                                last_price = pst.position.get("entry_underlying")
                            if last_price:
                                process_paper_signal(pst, "HOLD", last_price,
                                                     "", can_open=False, force_close=True)
                        await bot.send_message(chat_id=app.cfg.telegram_channel_id,
                                               text=format_paper_day_summary(pst),
                                               parse_mode=ParseMode.HTML)
                    except Exception:
                        pass
            except Exception as e:
                app.stats.last_error = f"on_rth_close: {repr(e)}"

        app.on_rth_open_cb = on_rth_open
        app.on_rth_close_cb = on_rth_close

        # Уведомление в закрытый канал (для фильтра тренда 1h, бумажной #2 и т.п.)
        async def notify_closed(text: str):
            try:
                await bot.send_message(chat_id=app.cfg.telegram_channel_id, text=text,
                                       parse_mode=ParseMode.HTML)
            except Exception as e:
                app.stats.last_error = f"notify_closed: {repr(e)}"
        app.notify_closed_cb = notify_closed

        asyncio.create_task(polling_loop(app, sender))

        # Устанавливаем меню команд по областям (scopes):
        #  - полное — в личке у владельца
        #  - краткое информационное — в твоих каналах
        #  - минимальное — по умолчанию везде
        try:
            await _setup_command_menus(bot, app)
        except Exception as e:
            app.stats.last_error = f"setup_menus: {repr(e)}"

        await dp.start_polling(bot)


def main() -> None:
    asyncio.run(_amain())
