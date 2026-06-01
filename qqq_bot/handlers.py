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
from .cache import cache_file_path
from .indicators import ema, rsi, bollinger, macd, vwap, atr, supertrend
from .signals import compute_signal

router = Router()


def _fmt_ts_z(dt) -> str:
    if dt is None:
        return "-"
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bars_to_df(bars):
    df = pd.DataFrame(
        [{
            "ts": b.ts,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        } for b in bars]
    )
    if not df.empty:
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def _add_indicators(df: pd.DataFrame, app: AppState) -> pd.DataFrame:
    if df.empty:
        return df
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    cfg = app.cfg.signal

    df["ema_fast"] = ema(close, cfg.ema_fast)
    df["ema_slow"] = ema(close, cfg.ema_slow)
    df["rsi"] = rsi(close, cfg.rsi_period)

    mid, upper, lower = bollinger(close, cfg.bb_period, cfg.bb_std)
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower

    # extras for strategy2 (чтобы /status показывал нормальный reason)
    df["ema_trend"] = ema(close, getattr(cfg, "ema_trend_period", 200))
    m_line, m_sig, m_hist = macd(
        close,
        fast=getattr(cfg, "macd_fast", 12),
        slow=getattr(cfg, "macd_slow", 26),
        signal=getattr(cfg, "macd_signal", 9),
    )
    df["macd"] = m_line
    df["macd_signal"] = m_sig
    df["macd_hist"] = m_hist
    df["vwap"] = vwap(df, tz_name=app.cfg.display_tz, reset_daily=True, price_mode=getattr(cfg, "vwap_price_mode", "typical"))
    df["atr"] = atr(df, period=getattr(cfg, "atr_period", 14))
    st_line, st_dir = supertrend(df, period=getattr(cfg, "supertrend_period", 10), multiplier=getattr(cfg, "supertrend_mult", 3.0))
    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir
    vol = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
    df["vol_ma"] = vol.rolling(int(getattr(cfg, "vol_ma_period", 20))).mean()

    return df


def _min_bars_for_indicators(app: AppState) -> int:
    s = app.cfg.signal
    need = max(
        int(getattr(s, "bb_period", 20)),
        int(getattr(s, "ema_slow", 21)),
        int(getattr(s, "rsi_period", 14)),
        int(getattr(s, "ema_trend_period", 200)),
        int(getattr(s, "atr_period", 14)),
        int(getattr(s, "supertrend_period", 10)),
        int(getattr(s, "macd_slow", 26)),
    )
    return need + 5


def _cooldown_left_seconds(app: AppState) -> int:
    cd = int(app.cfg.cooldown_seconds)
    if cd <= 0 or app.stats.last_signal_sent_at is None:
        return 0
    from .utils_time import utc_now
    now = utc_now()
    left = int((app.stats.last_signal_sent_at + timedelta(seconds=cd) - now).total_seconds())
    return max(left, 0)


def _help_text() -> str:
    return (
        "Команды бота:\n\n"
        "/start — запуск и краткая справка\n"
        "/help — список команд\n"
        "/status — подробный статус\n"
        "/stats — краткая статистика\n"
        "/chart — текущий график\n"
        "/last — последняя цена и последний бар\n"
        "/dump [N] — последние N баров\n"
        "/config — текущие настройки\n"
        "/strategy — показать текущую стратегию\n"
        "/strategy#1 — включить стратегию #1 (BB+EMA+RSI)\n"
        "/strategy#2 — включить стратегию #2 (MACD+VWAP+RSI+Supertrend)\n"
        "/ping — проверка, что бот жив\n"
        "/reset — очистить кеш (и файл кеша)\n"
    )


def _position_from_history(app: AppState) -> str:
    if app.strategy_id == 2:
        return str(getattr(app.strategy2, "position", "FLAT") or "FLAT").upper()

    hist = getattr(app.stats, "signal_history", None) or []
    if not hist:
        last = getattr(app.stats, "last_signal", None)
        if last == "BUY":
            return "LONG"
        if last == "SELL":
            return "SHORT"
        return "FLAT"

    action = hist[-1][0]
    return "LONG" if action == "BUY" else "SHORT" if action == "SELL" else "FLAT"


@router.message(Command("start"))
async def cmd_start(message: Message, app: AppState) -> None:
    txt = (
        "Бот запущен.\n\n"
        f"Символ: {app.cfg.symbol}\n"
        f"TF: {app.cfg.timeframe_minutes}m\n"
        f"Strategy: #{app.strategy_id}\n\n"
        "Для справки: /help"
    )
    await message.answer("<pre>" + html.escape(txt) + "</pre>")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer("<pre>" + html.escape(_help_text()) + "</pre>")


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

    # просто показать текущую
    await message.answer(
        "<pre>" + html.escape(
            f"Текущая стратегия: #{app.strategy_id}\n"
            "Переключение: /strategy#1 или /strategy#2"
        ) + "</pre>"
    )


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

    price_txt = f"${last_price:.2f}" if isinstance(last_price, (int, float)) else "-"
    close_txt = f"{last_close:.4f}" if isinstance(last_close, (int, float)) else "-"

    txt = (
        "Последнее:\n\n"
        f"Symbol: {app.cfg.symbol}\n"
        f"Strategy: #{app.strategy_id}\n"
        f"Последняя цена (quote): {price_txt}\n"
        f"Последний бар (UTC): {last_bar_ts}\n"
        f"Close последнего бара: {close_txt}\n"
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
    lines = []
    for b in tail:
        lines.append(f"{_fmt_ts_z(b.ts)} O={b.open:.4f} H={b.high:.4f} L={b.low:.4f} C={b.close:.4f} V={int(b.volume)}")

    await message.answer("<pre>" + html.escape("\n".join(lines)) + "</pre>")


@router.message(Command("status"))
async def cmd_status(message: Message, app: AppState) -> None:
    bars = app.cache.to_list()
    s = app.cfg.signal

    cd_left = _cooldown_left_seconds(app)
    cd_txt = f"{app.cfg.cooldown_seconds}s" + (f" (ещё {cd_left}s)" if cd_left > 0 else "")
    cd_active = "да" if cd_left > 0 else "нет"

    min_bars = _min_bars_for_indicators(app)

    last_bar = bars[-1] if bars else None
    last_bar_ts = _fmt_ts_z(last_bar.ts) if last_bar else "-"
    last_close = float(last_bar.close) if last_bar else None

    last_price = None
    try:
        last_price = await app.tn.get_quote_ltp(app.cfg.symbol)
    except Exception:
        last_price = last_close

    price_txt = f"${last_price:.2f}" if isinstance(last_price, (int, float)) else "-"

    df = _bars_to_df(bars[-max(app.cfg.chart_bars, min_bars + 10):]) if bars else pd.DataFrame()
    df = _add_indicators(df, app)

    last_err = app.stats.last_error or "-"
    last_sig = getattr(app.stats, "last_signal", None) or "-"
    last_sig_ts = _fmt_ts_z(getattr(app.stats, "last_signal_ts", None)) if getattr(app.stats, "last_signal_ts", None) else "-"
    last_sig_price = getattr(app.stats, "last_signal_price", None)
    last_sig_price_txt = f"{last_sig_price:.2f}" if isinstance(last_sig_price, (int, float)) else "-"
    last_sig_type = getattr(app.stats, "last_signal_type", None) or "-"

    decision_txt = "Нет данных."
    if len(df) >= min_bars and len(df) >= 2:
        df_sig = df.iloc[:-1].copy()
        df_sig = _add_indicators(df_sig, app)
        dec = compute_signal(df_sig, s, strategy_id=app.strategy_id, runtime_state=None, current_position=getattr(app.strategy2, "position", "FLAT"))
        decision_txt = dec.reason
    else:
        decision_txt = f"Недостаточно данных для индикаторов: нужно {min_bars}, есть {len(df)}."

    pos_txt = _position_from_history(app)

    hist = getattr(app.stats, "signal_history", None) or []
    tail = hist[-5:] if len(hist) > 5 else hist

    def _fmt_sig_item(item) -> str:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            return str(item)
        a = item[0]
        t = item[1]
        p = item[2] if len(item) >= 3 else None
        p_txt = f" @ {p:.2f}" if isinstance(p, (int, float)) else ""
        sig_type = item[3] if len(item) >= 4 and item[3] else None
        mode = item[4] if len(item) >= 5 and item[4] else None
        label = f"{a}/{sig_type}" if sig_type else str(a)
        if mode:
            label += f"/{mode}"
        return f"{label} @ {_fmt_ts_z(t)}{p_txt}"

    tail = hist[-5:] if len(hist) > 5 else hist
    sig_hist_txt = "Сигналы текущей сессии: -" if not tail else "Сигналы текущей сессии:\n- " + "\n- ".join(
        [_fmt_sig_item(x) for x in tail]
    )

    sid = getattr(app.stats, "session_id", None) or "-"

    txt = (
        "Статус бота:\n\n"
        f"Символ: {app.cfg.symbol}\n"
        f"TF баров: {app.cfg.timeframe_minutes}m\n"
        f"Strategy: #{app.strategy_id}\n"
        f"SessionID(NY date): {sid}\n"
        f"Сессия: {'ОТКРЫТА' if (getattr(app.stats, 'session_state', None) == 'OPEN') else 'ЗАКРЫТА'}\n\n"
        f"Cooldown: {cd_txt} (активен: {cd_active})\n"
        f"Кеш: {len(app.cache)} точек (минимум для индикаторов: {min_bars})\n\n"
        f"Последняя ошибка: {last_err}\n\n"
        f"Последняя цена: {price_txt}\n"
        f"Время последнего бара (UTC): {last_bar_ts}\n"
        f"Последний сигнал: {last_sig} / {last_sig_type} @ {last_sig_ts}\n"
        f"Цена открытия (по сигналу): {last_sig_price_txt}\n"
        f"Позиция: {pos_txt}\n"
        f"{sig_hist_txt}\n\n"
        "Причина/состояние:\n"
        f"- {decision_txt}\n"
    )

    await message.answer("<pre>" + html.escape(txt) + "</pre>")


@router.message(Command("stats"))
async def cmd_stats(message: Message, app: AppState) -> None:
    cd_left = _cooldown_left_seconds(app)
    cd_txt = f"{app.cfg.cooldown_seconds}s" + (f" (ещё {cd_left}s)" if cd_left > 0 else "")

    txt = (
        "Статистика:\n\n"
        f"Strategy: #{app.strategy_id}\n"
        f"SessionID(NY date): {getattr(app.stats,'session_id',None) or '-'}\n"
        f"Ticks: {app.stats.ticks}\n"
        f"Bars: {len(app.cache)} (real={getattr(app.stats,'bars_real',0)}, synth={getattr(app.stats,'bars_synth',0)})\n"
        f"Signals: BUY={getattr(app.stats,'signals_buy',0)}, SELL={getattr(app.stats,'signals_sell',0)}\n"
        f"Cooldown: {cd_txt}, skips={getattr(app.stats,'cooldown_skips',0)}\n"
        f"Session: {getattr(app.stats,'session_state',None) or '-'}\n"
        f"Now(UTC): {_fmt_ts_z(getattr(app.stats,'now_utc',None)) if getattr(app.stats,'now_utc',None) else '-'}\n"
        f"Last error: {getattr(app.stats,'last_error',None) or '-'}"
    )
    await message.answer("<pre>" + html.escape(txt) + "</pre>")
