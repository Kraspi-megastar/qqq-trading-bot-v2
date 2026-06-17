"""
bot.py — точка входа.

Опционные сделки:
  - При OPEN: запрашиваем цену опциона через TraderNet, открываем TradeRecord.
  - При CLOSE: запрашиваем цену, закрываем TradeRecord, считаем P&L.
  - При HOLD: ничего не делаем с журналом.
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
from .options import get_option_recommendation, format_option_message, OptionConfig as _OC
from .trades import TradeJournal


def _fmt_ts_z(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt is not None else "-"


async def _fetch_option_price(tn: TraderNetClient, tn_ticker: str) -> float | None:
    """Запрашивает рыночную цену опциона из TraderNet. None при ошибке."""
    try:
        price = await asyncio.wait_for(tn.get_quote_ltp(tn_ticker), timeout=8.0)
        return float(price) if price else None
    except Exception:
        return None


async def _send_signal_to_channel(
    bot: Bot,
    app: AppState,
    decision: SignalDecision,
    chart_path: str,
    df_sig,
    session: aiohttp.ClientSession,
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

    close    = _fv("close")
    ts       = last["ts"] if last is not None else None
    rsi_v    = _fv("rsi")
    ema_f    = _fv("ema_fast")
    ema_sl   = _fv("ema_slow")
    bb_l     = _fv("bb_lower")
    bb_m     = _fv("bb_mid")
    bb_u     = _fv("bb_upper")

    lines = [
        f"<b>{html.escape(direction)}</b>",
        f"<b>{html.escape(app.cfg.symbol)}</b>  TF={app.cfg.timeframe_minutes}m  STR=#{app.strategy_id}",
        html.escape(f"BarTS={_fmt_ts_z(ts)}"),
        html.escape(f"Close={close:.4f}" if isinstance(close, float) else "Close=n/a"),
        html.escape(
            f"RSI({s.rsi_period})={rsi_v:.2f}"
            if isinstance(rsi_v, float) else f"RSI({s.rsi_period})=n/a"
        ),
        html.escape(
            f"EMA{s.ema_fast}={ema_f:.2f} | EMA{s.ema_slow}={ema_sl:.2f}"
            if isinstance(ema_f, float) and isinstance(ema_sl, float) else "EMA=n/a"
        ),
        html.escape(
            f"BB({s.bb_period},{s.bb_std}): L={bb_l:.2f} M={bb_m:.2f} U={bb_u:.2f}"
            if isinstance(bb_l, float) else "BB=n/a"
        ),
    ]

    if app.strategy_id == 2:
        atr_stop = app.strategy2.atr_stop
        if atr_stop is not None:
            lines.append(html.escape(f"ATR-Stop={atr_stop:.2f}"))

    lines += ["", f"<i>{html.escape(decision.reason)}</i>"]

    # ── Опционный блок ────────────────────────────────────────────────────
    if app.cfg.option.enabled and isinstance(close, float):
        try:
            oc = app.cfg.option
            opt_cfg = _OC(
                enabled=oc.enabled,
                min_dte=oc.min_dte,
                strike_step=oc.strike_step,
                strike_offset=oc.strike_offset,
                underlying_symbol=oc.underlying_symbol,
            )
            rec = await get_option_recommendation(
                signal=decision.action,
                underlying_price=close,
                cfg=opt_cfg,
                current_position=app.option_position,
                can_short=True,
                session=session,
                api_url=app.cfg.tradernet_api_url,
                sid=app.cfg.tradernet_sid,
            )

            now_utc = datetime.now(tz=timezone.utc)
            session_date = getattr(app.stats, "session_id", None) or now_utc.date().isoformat()

            # ── Запись сделки в журнал ─────────────────────────────────────
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
                        entry_underlying=close,
                        entry_ts=now_utc,
                    )
                elif rec.action_type == "CLOSE" and app.option_position is not None:
                    opt_price = await _fetch_option_price(app.tn, rec.tn_ticker)
                    closed = app.trade_journal.close_trade(
                        ticker=rec.tn_ticker,
                        exit_price=opt_price,
                        exit_underlying=close,
                        exit_ts=now_utc,
                    )
                    # Добавляем P&L в сообщение
                    if closed is not None:
                        pnl_str = closed.pnl_str()
                        lines.append(html.escape(f"P&L сделки: {pnl_str}"))

            # Применяем новую позицию
            app.option_position = rec.new_position

            if not (rec.action_type == "HOLD" and rec.new_position is None):
                lines += ["", format_option_message(rec)]

        except Exception as e:
            lines += ["", html.escape(f"[Опцион: ошибка — {e}]")]

    caption = "\n".join(lines)

    await bot.send_photo(
        chat_id=app.cfg.telegram_channel_id,
        photo=FSInputFile(chart_path),
        caption=caption,
        parse_mode=ParseMode.HTML,
    )


async def _amain() -> None:
    cfg = load_config()

    bot = Bot(
        token=cfg.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)

    cache = BarCache(
        timeframe_minutes=cfg.timeframe_minutes,
        maxlen=max(cfg.chart_bars * 5, 2000),
    )
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

        dp["app"] = app

        asyncio.create_task(bootstrap_history(app))

        async def sender(decision: SignalDecision, chart_path: str, df_sig) -> None:
            await _send_signal_to_channel(bot, app, decision, chart_path, df_sig, session)

        asyncio.create_task(polling_loop(app, sender))

        await dp.start_polling(bot)


def main() -> None:
    asyncio.run(_amain())
