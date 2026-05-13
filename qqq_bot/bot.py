from __future__ import annotations

import asyncio
import html

import aiohttp
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile

from .cache import BarCache, Stats
from .config import load_config
from .handlers import router
from .scheduler import AppState, bootstrap_history, polling_loop
from .signals import SignalDecision
from .tradernet import TraderNetClient

try:
    from .cache import state_file_path
except Exception:
    state_file_path = None

try:
    from .ml.service import MLTradingService
except Exception:
    MLTradingService = None


def _fmt_ts_z(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt is not None else "-"


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


async def _send_signal_to_channel(
    bot: Bot,
    app: AppState,
    decision: SignalDecision,
    chart_path: str,
    df_sig,
) -> None:
    s = app.cfg.signal
    direction = "🟢 BUY" if decision.action == "BUY" else "🔴 SELL"

    last = df_sig.iloc[-1] if df_sig is not None and len(df_sig) > 0 else None

    close = _safe_float(last["close"]) if last is not None and "close" in last else None
    ts = last["ts"] if last is not None and "ts" in last else None

    rsi_v = _safe_float(last["rsi"]) if last is not None and "rsi" in last else None

    ema_f = _safe_float(last["ema_fast"]) if last is not None and "ema_fast" in last else None
    ema_sl = _safe_float(last["ema_slow"]) if last is not None and "ema_slow" in last else None

    bb_l = _safe_float(last["bb_lower"]) if last is not None and "bb_lower" in last else None
    bb_m = _safe_float(last["bb_mid"]) if last is not None and "bb_mid" in last else None
    bb_u = _safe_float(last["bb_upper"]) if last is not None and "bb_upper" in last else None

    lines = [
        f"<b>{html.escape(direction)}</b>",
        f"<b>{html.escape(app.cfg.symbol)}</b> TF={app.cfg.timeframe_minutes}m",
        html.escape(f"BarTS={_fmt_ts_z(ts)}"),
        html.escape(f"Close={close:.4f}" if isinstance(close, (int, float)) else "Close=n/a"),

        html.escape(
            f"RSI({s.rsi_period})={rsi_v:.2f} "
            f"(BUY<{s.rsi_buy:.1f}, SELL>{s.rsi_sell:.1f})"
        )
        if isinstance(rsi_v, (int, float))
        else html.escape(f"RSI({s.rsi_period})=n/a"),

        html.escape(f"EMA{s.ema_fast}={ema_f:.2f} | EMA{s.ema_slow}={ema_sl:.2f}")
        if isinstance(ema_f, (int, float)) and isinstance(ema_sl, (int, float))
        else html.escape("EMA=n/a"),

        html.escape(f"BB({s.bb_period},{s.bb_std}): L={bb_l:.2f} M={bb_m:.2f} U={bb_u:.2f}")
        if isinstance(bb_l, (int, float)) and isinstance(bb_m, (int, float)) and isinstance(bb_u, (int, float))
        else html.escape("BB=n/a"),

        "",
        f"<i>{html.escape(decision.reason)}</i>",
    ]

    ml = getattr(app, "last_ml_decision", None)
    if ml is not None:
        long_prob = getattr(ml, "long_prob", None)
        short_prob = getattr(ml, "short_prob", None)

        long_txt = f"{long_prob:.2f}" if isinstance(long_prob, (int, float)) else "n/a"
        short_txt = f"{short_prob:.2f}" if isinstance(short_prob, (int, float)) else "n/a"

        lines.extend(
            [
                "",
                html.escape(
                    f"ML mode={getattr(ml, 'mode', 'n/a')} | "
                    f"final={getattr(ml, 'final_signal', 'n/a')} | "
                    f"long={long_txt} | short={short_txt}"
                ),
                html.escape(f"ML reason={getattr(ml, 'reason', 'n/a')}"),
            ]
        )

    caption = "\n".join(lines)

    await bot.send_photo(
        chat_id=app.cfg.telegram_channel_id,
        photo=FSInputFile(chart_path),
        caption=caption,
    )


def _load_ml_service():
    if MLTradingService is None:
        return None

    try:
        return MLTradingService.from_env()
    except Exception:
        return None


async def _amain() -> None:
    cfg = load_config()
    ml_service = _load_ml_service()

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

    # Restore persisted signal state if this project version supports it.
    if state_file_path is not None:
        try:
            sp = state_file_path(cfg.cache_dir, cfg.symbol, cfg.timeframe_minutes)
            stats.load_state(sp)
        except Exception:
            pass

    async with aiohttp.ClientSession() as session:
        tn = TraderNetClient(
            api_url=cfg.tradernet_api_url,
            quotes_url=cfg.tradernet_quotes_url,
            session=session,
        )

        app = AppState(cfg=cfg, tn=tn, cache=cache, stats=stats)
        app.strategy_id = cfg.strategy_id

        # ML advisory/filter layer. If ML is not installed or not configured,
        # bot continues to work normally without blocking base signals.
        app.ml_service = ml_service
        app.last_ml_decision = None

        dp["app"] = app

        # Bootstrap history can take time or fail on provider timeout.
        # Run it in the background so bot commands become available immediately.
        asyncio.create_task(bootstrap_history(app))

        async def sender(decision: SignalDecision, chart_path: str, df_sig) -> None:
            await _send_signal_to_channel(bot, app, decision, chart_path, df_sig)

        asyncio.create_task(polling_loop(app, sender))

        await dp.start_polling(bot)


def main() -> None:
    asyncio.run(_amain())