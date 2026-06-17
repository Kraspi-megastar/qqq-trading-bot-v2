"""
bot.py — точка входа: создаёт бота, запускает polling_loop и обработчики.

Ключевая логика опционных сигналов (стратегия #1):
  - Хранится app.option_position (OptionPosition | None).
  - get_option_recommendation() получает текущую позицию и возвращает
    рекомендацию с action_type: OPEN / CLOSE / HOLD.
  - После отправки сигнала app.option_position обновляется через rec.new_position.
  - Состояние персистируется в state.json.
"""
from __future__ import annotations

import asyncio
import html

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
from .options import (
    get_option_recommendation,
    format_option_message,
    OptionConfig as _OC,
)


def _fmt_ts_z(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt is not None else "-"


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

    # ATR-стоп для стратегии #2
    if app.strategy_id == 2:
        atr_stop = app.strategy2.atr_stop
        if atr_stop is not None:
            lines.append(html.escape(f"ATR-Stop={atr_stop:.2f}"))

    lines += ["", f"<i>{html.escape(decision.reason)}</i>"]

    # ── Опционный блок (только стратегия #1) ──────────────────────────────
    if app.strategy_id == 1 and app.cfg.option.enabled and isinstance(close, float):
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
                session=session,
                api_url=app.cfg.tradernet_api_url,
                sid=app.cfg.tradernet_sid,
            )

            # Применяем новую позицию
            app.option_position = rec.new_position

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

        dp["app"] = app

        asyncio.create_task(bootstrap_history(app))

        async def sender(decision: SignalDecision, chart_path: str, df_sig) -> None:
            await _send_signal_to_channel(bot, app, decision, chart_path, df_sig, session)

        asyncio.create_task(polling_loop(app, sender))

        await dp.start_polling(bot)


def main() -> None:
    asyncio.run(_amain())
