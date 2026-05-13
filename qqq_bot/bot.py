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


def _fmt_ts_z(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt is not None else "-"


async def _send_signal_to_channel(bot: Bot, app: AppState, decision: SignalDecision, chart_path: str, df_sig) -> None:
    s = app.cfg.signal
    direction = "🟢 BUY" if decision.action == "BUY" else "🔴 SELL"

    last = df_sig.iloc[-1] if df_sig is not None and len(df_sig) > 0 else None

    close = float(last["close"]) if last is not None else None
    ts = last["ts"] if last is not None else None
    rsi_v = float(last["rsi"]) if last is not None and "rsi" in last else None
    ema_f = float(last["ema_fast"]) if last is not None and "ema_fast" in last else None
    ema_sl = float(last["ema_slow"]) if last is not None and "ema_slow" in last else None
    bb_l = float(last["bb_lower"]) if last is not None and "bb_lower" in last else None
    bb_m = float(last["bb_mid"]) if last is not None and "bb_mid" in last else None
    bb_u = float(last["bb_upper"]) if last is not None and "bb_upper" in last else None

    lines = [
        f"<b>{html.escape(direction)}</b>",
        f"<b>{html.escape(app.cfg.symbol)}</b> TF={app.cfg.timeframe_minutes}m",
        html.escape(f"BarTS={_fmt_ts_z(ts)}"),
        html.escape(f"Close={close:.4f}" if isinstance(close, (int, float)) else "Close=n/a"),
        html.escape(f"RSI({s.rsi_period})={rsi_v:.2f} (BUY<{s.rsi_buy:.1f}, SELL>{s.rsi_sell:.1f})") if isinstance(
            rsi_v, (int, float)) else html.escape(f"RSI({s.rsi_period})=n/a"),
        html.escape(f"EMA{s.ema_fast}={ema_f:.2f} | EMA{s.ema_slow}={ema_sl:.2f}")
        if isinstance(ema_f, (int, float)) and isinstance(ema_sl, (int, float))
        else html.escape("EMA=n/a"),

        html.escape(f"BB({s.bb_period},{s.bb_std}): L={bb_l:.2f} M={bb_m:.2f} U={bb_u:.2f}")
        if isinstance(bb_l, (int, float)) and isinstance(bb_m, (int, float)) and isinstance(bb_u, (int, float))
        else html.escape("BB=n/a"),
        "",
        f"<i>{html.escape(decision.reason)}</i>",
    ]

    caption = "\n".join(lines)

    await bot.send_photo(
        chat_id=app.cfg.telegram_channel_id,
        photo=FSInputFile(chart_path),
        caption=caption,
    )


async def _amain() -> None:
    cfg = load_config()

    bot = Bot(
        token=cfg.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)

    cache = BarCache(timeframe_minutes=cfg.timeframe_minutes, maxlen=max(cfg.chart_bars * 5, 2000))
    stats = Stats()

    # restore persisted signal state (last signals/position)
    try:
        sp = state_file_path(cfg.cache_dir, cfg.symbol, cfg.timeframe_minutes)
        stats.load_state(sp)
    except Exception:
        pass

    async with aiohttp.ClientSession() as session:
        tn = TraderNetClient(api_url=cfg.tradernet_api_url, quotes_url=cfg.tradernet_quotes_url, session=session)
        app = AppState(cfg=cfg, tn=tn, cache=cache, stats=stats)
        app.strategy_id = cfg.strategy_id

        dp["app"] = app

        # Bootstrap истории может занимать время/падать по таймаутам.
        # Запускаем в фоне, чтобы бот сразу отвечал на команды.
        asyncio.create_task(bootstrap_history(app))

        async def sender(decision: SignalDecision, chart_path: str, df_sig) -> None:
            await _send_signal_to_channel(bot, app, decision, chart_path, df_sig)

        asyncio.create_task(polling_loop(app, sender))

        await dp.start_polling(bot)


def main() -> None:
    asyncio.run(_amain())
