from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo
import asyncio
import pandas as pd

from .cache import BarCache, Stats, cache_file_path
from .config import AppConfig
from .models import Bar
from .utils_time import utc_now, floor_time
from .indicators import ema, rsi, bollinger, macd, vwap, atr, supertrend
from .signals import compute_signal, SignalDecision
from .charting import plot_chart
from .tradernet import TraderNetClient
from .state_store import apply_state_if_same_session, build_state_from_app, save_state

@dataclass
class Strategy2Runtime:
    position: str = "FLAT"       # "FLAT" | "LONG"
    entry_price: float | None = None
    atr_stop: float | None = None
    entry_ts: datetime | None = None


@dataclass
class AppState:
    cfg: AppConfig
    tn: TraderNetClient
    cache: BarCache
    stats: Stats
    state_loaded: bool = False

    last_chart_path: str | None = None
    last_signal_sent: str | None = None
    last_signal_sent_ts: object | None = None

    strategy_id: int = 1
    strategy2: Strategy2Runtime = field(default_factory=Strategy2Runtime)

    def set_strategy(self, strategy_id: int) -> None:
        sid = int(strategy_id)
        if sid not in (1, 2):
            return
        self.strategy_id = sid
        # сбрасываем runtime, чтобы не тащить стопы/позицию между стратегиями
        self.strategy2 = Strategy2Runtime()
        self.last_signal_sent = None
        self.last_signal_sent_ts = None

    def persist_state(self, now_utc: datetime) -> None:
        sid = _session_id(now_utc, self.cfg.display_tz)
        st = build_state_from_app(app=self, session_id=sid)
        save_state(self.cfg.cache_dir, st)


def _session_id(now_utc: datetime, tz_name: str) -> str:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    local = now_utc.astimezone(ZoneInfo(tz_name))
    return local.date().isoformat()


def _maybe_reset_session(app: AppState, now_utc: datetime) -> None:
    sid = _session_id(now_utc, app.cfg.display_tz)
    prev = getattr(app.stats, "session_id", None)
    if prev != sid:
        setattr(app.stats, "session_id", sid)
        # очищаем историю сигналов -> на графике будет только текущая сессия
        try:
            app.stats.signal_history.clear()
        except Exception:
            app.stats.signal_history = []
        # сброс локальных защит от дублей
        app.last_signal_sent = None
        app.last_signal_sent_ts = None
        # сброс runtime для стратегии #2
        app.strategy2 = Strategy2Runtime()


def _bars_to_df(bars: list[Bar]) -> pd.DataFrame:
    df = pd.DataFrame(
        [{
            "ts": b.ts,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "synth": bool(getattr(b, "synthetic", False)),
        } for b in bars]
    )
    if not df.empty:
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def _add_indicators(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    if df.empty:
        return df

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    # strategy #1 base
    df["ema_fast"] = ema(close, cfg.signal.ema_fast)
    df["ema_slow"] = ema(close, cfg.signal.ema_slow)
    df["rsi"] = rsi(close, cfg.signal.rsi_period)
    mid, upper, lower = bollinger(close, cfg.signal.bb_period, cfg.signal.bb_std)
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower

    # strategy #2 extras
    df["ema_trend"] = ema(close, getattr(cfg.signal, "ema_trend_period", 200))
    m_line, m_sig, m_hist = macd(
        close,
        fast=getattr(cfg.signal, "macd_fast", 12),
        slow=getattr(cfg.signal, "macd_slow", 26),
        signal=getattr(cfg.signal, "macd_signal", 9),
    )
    df["macd"] = m_line
    df["macd_signal"] = m_sig
    df["macd_hist"] = m_hist

    df["vwap"] = vwap(
        df,
        tz_name=cfg.display_tz,
        reset_daily=True,
        price_mode=getattr(cfg.signal, "vwap_price_mode", "typical"),
    )

    df["atr"] = atr(df, period=getattr(cfg.signal, "atr_period", 14))

    st_line, st_dir = supertrend(
        df,
        period=getattr(cfg.signal, "supertrend_period", 10),
        multiplier=getattr(cfg.signal, "supertrend_mult", 3.0),
    )
    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir

    vol_ma_period = int(getattr(cfg.signal, "vol_ma_period", 20))
    vol = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
    df["vol_ma"] = vol.rolling(vol_ma_period).mean()

    return df


def _min_bars_for_indicators(cfg: AppConfig) -> int:
    s = cfg.signal
    # учитываем EMA200/супертренд/ATR/MACD чтобы было корректно
    need = max(
        int(getattr(s, "ema_slow", 21)),
        int(getattr(s, "bb_period", 20)),
        int(getattr(s, "rsi_period", 14)),
        int(getattr(s, "ema_trend_period", 200)),
        int(getattr(s, "atr_period", 14)),
        int(getattr(s, "supertrend_period", 10)),
        int(getattr(s, "macd_slow", 26)),
    )
    return need + 5


def _record_signal(app: AppState, action: str, bar_ts_utc: datetime, price: float | None) -> None:
    """
    Храним сигналы текущей сессии:
      (action, ts, price)
    Цена = close закрытого бара, на котором сработал сигнал.
    """
    try:
        hist = app.stats.signal_history
    except Exception:
        app.stats.signal_history = []
        hist = app.stats.signal_history

    # нормализуем цену
    p = float(price) if isinstance(price, (int, float)) else None

    # защита от дублей/спама
    if hist:
        last = hist[-1]
        # поддержка старого формата (action, ts)
        last_action = last[0] if isinstance(last, (tuple, list)) and len(last) >= 1 else None
        last_ts = last[1] if isinstance(last, (tuple, list)) and len(last) >= 2 else None

        # один и тот же бар/сигнал
        if last_action == action and last_ts == bar_ts_utc:
            return
        # подряд одинаковое действие не пишем
        if last_action == action:
            return

    hist.append((action, bar_ts_utc, p))

    # лимит на сессию
    max_keep = 200
    if len(hist) > max_keep:
        del hist[:-max_keep]

    app.stats.last_signal = action
    app.stats.last_signal_ts = bar_ts_utc
    app.stats.last_signal_price = p



def _replay_signal_history_from_cache(app: AppState, max_signals: int = 200) -> None:
    """
    Пересчитываем сигналы по истории и оставляем ТОЛЬКО текущую (по дате NY) сессию,
    чтобы после рестарта внутри дня на графике были “все предыдущие сигналы”.
    """
    cfg = app.cfg
    bars_real = [b for b in app.cache.to_list() if not getattr(b, "synthetic", False)]
    df = _bars_to_df(bars_real)
    if df.empty:
        app.stats.signal_history = []
        return

    df = _add_indicators(df, cfg)
    min_bars = _min_bars_for_indicators(cfg)

    # определим “текущую сессию” по последнему бару кеша
    last_ts = pd.to_datetime(df["ts"].iloc[-1], utc=True, errors="coerce")
    sid = _session_id(last_ts.to_pydatetime(), cfg.display_tz) if pd.notna(last_ts) else None
    setattr(app.stats, "session_id", sid)

    hist: list[tuple[str, datetime]] = []
    prev_action: str | None = None

    for i in range(min_bars, len(df) + 1):
        window = df.iloc[:i]
        dec = compute_signal(window, cfg.signal, strategy_id=app.strategy_id, runtime_state=app.strategy2)
        if dec.action not in ("BUY", "SELL"):
            continue

        ts_bar = window["ts"].iloc[-1]
        ts_bar_dt = ts_bar if isinstance(ts_bar, datetime) else pd.to_datetime(ts_bar, utc=True, errors="coerce").to_pydatetime()

        # фильтр по текущей сессии
        if sid is not None:
            if _session_id(ts_bar_dt, cfg.display_tz) != sid:
                continue

        if prev_action == dec.action:
            continue

        bar_price = None
        try:
            bar_price = float(window["close"].iloc[-1])
        except Exception:
            bar_price = None

        hist.append((dec.action, ts_bar_dt, bar_price))

        prev_action = dec.action

    app.stats.signal_history = hist[-max_signals:]
    if app.stats.signal_history:
        last = app.stats.signal_history[-1]

        # поддержка форматов (action, ts) и (action, ts, price)
        app.stats.last_signal = last[0] if len(last) >= 1 else None
        app.stats.last_signal_ts = last[1] if len(last) >= 2 else None
        app.stats.last_signal_price = last[2] if len(last) >= 3 else None


def _adjust_date_to_for_closed_session(date_to_utc: datetime) -> datetime:
    dt = date_to_utc
    while dt.weekday() >= 5:
        dt = dt - timedelta(days=1)
    return dt


def _is_extended_session_open(now_utc: datetime, tz_name: str) -> bool:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(tz_name)
    local = now_utc.astimezone(tz)
    if local.weekday() >= 5:
        return False
    t = local.timetz().replace(tzinfo=None)
    return time(4, 0) <= t < time(20, 0)


async def bootstrap_history(app: AppState) -> None:
    cfg = app.cfg
    tf = cfg.timeframe_minutes

    cache_path = cache_file_path(cfg.cache_dir, cfg.symbol, tf)
    app.stats.cache_file = str(cache_path)

    try:
        n = app.cache.load_from_file(cache_path)
        app.stats.cache_load = f"HIT ({n})" if n > 0 else "MISS"
    except Exception as e:
        app.stats.cache_load = f"ERROR: {repr(e)}"

    app.stats.bars_real = sum(1 for b in app.cache.to_list() if not getattr(b, "synthetic", False))
    app.stats.bars_synth = sum(1 for b in app.cache.to_list() if getattr(b, "synthetic", False))

    _replay_signal_history_from_cache(app)

    min_bars = _min_bars_for_indicators(cfg)
    if len(app.cache) >= min_bars:
        app.stats.bootstrap_result = "SKIP (cache sufficient)"
        return

    now = utc_now()
    date_to = floor_time(now, tf)
    date_to = _adjust_date_to_for_closed_session(date_to)
    date_from = date_to - timedelta(days=cfg.history_lookback_days)

    bars_needed = min(
        app.cache.maxlen,
        int((cfg.history_lookback_days * 24 * 60) / max(1, tf)) + 10,
    )
    bars_needed = min(bars_needed, 800)

    app.stats.bootstrap_attempt = f"from={date_from.isoformat()} to={date_to.isoformat()} count={-int(bars_needed)}"

    bars: list[Bar] = []
    last_exc: Exception | None = None

    for need in (bars_needed, max(200, bars_needed // 2), 200):
        try:
            bars = await asyncio.wait_for(
                app.tn.get_hloc(
                    symbol=cfg.symbol,
                    timeframe_minutes=tf,
                    count=-int(need),
                    date_from_utc=date_from,
                    date_to_utc=date_to,
                ),
                timeout=cfg.tradernet_timeout_seconds + 5,
            )
            last_exc = None
            break
        except Exception as e:
            last_exc = e

    if last_exc is not None:
        app.stats.bootstrap_result = f"ERROR: {repr(last_exc)}"
        app.stats.last_error = repr(last_exc)
        return

    # merge by ts
    uniq: dict[datetime, Bar] = {b.ts: b for b in app.cache.to_list() if not getattr(b, "synthetic", False)}
    for b in bars:
        b.synthetic = False
        uniq[b.ts] = b

    merged = sorted(uniq.values(), key=lambda x: x.ts)
    app.cache.clear()
    app.cache.extend(merged[-app.cache.maxlen:])

    app.stats.bars_real = len(app.cache)
    app.stats.bars_synth = 0
    app.stats.bootstrap_result = f"OK (fetched {len(bars)}, merged {len(app.cache)})"

    _replay_signal_history_from_cache(app)

    try:
        if len(app.cache) > 0:
            app.cache.save_to_file(cache_path)
            app.stats.cache_save = f"OK ({len(app.cache)})"
        else:
            app.stats.cache_save = "SKIP (empty)"
    except Exception as e:
        app.stats.cache_save = f"ERROR: {repr(e)}"


def _cooldown_active(app: AppState) -> tuple[bool, int]:
    if app.stats.last_signal_sent_at is None:
        return False, 0
    cd = int(app.cfg.cooldown_seconds)
    if cd <= 0:
        return False, 0
    now = utc_now()
    left = int((app.stats.last_signal_sent_at + timedelta(seconds=cd) - now).total_seconds())
    return (left > 0), max(left, 0)


async def polling_loop(app: AppState, send_signal_cb) -> None:
    cfg = app.cfg
    tf = cfg.timeframe_minutes

    cache_path = cache_file_path(cfg.cache_dir, cfg.symbol, tf)
    app.stats.cache_file = str(cache_path)

    last_persist = utc_now()

    while True:
        rolled = False

        try:
            app.stats.ticks += 1
            now = utc_now()

            # 1) restore state once
            if not app.state_loaded:
                sid = _session_id(now, cfg.display_tz)
                applied = apply_state_if_same_session(app=app, current_session_id=sid)
                app.state_loaded = True
                if applied:
                    # лучше вынести в state_status, но пока оставим как есть
                    app.stats.last_error = "State restored from state.json"

            _maybe_reset_session(app, now)

            session_open = _is_extended_session_open(now, cfg.display_tz)
            app.stats.session_state = "OPEN" if session_open else "CLOSED"
            app.stats.now_utc = now

            # 2) tick -> update bar cache
            ltp = await app.tn.get_quote_ltp(cfg.symbol)

            if session_open:
                slot = floor_time(now, tf)
                last = app.cache.last()

                if last is None:
                    app.cache.append(
                        Bar(ts=slot, open=ltp, high=ltp, low=ltp, close=ltp, volume=0.0, synthetic=False)
                    )
                    app.stats.bars_real += 1
                else:
                    if slot > last.ts:
                        rolled = True
                        app.cache.append(
                            Bar(ts=slot, open=ltp, high=ltp, low=ltp, close=ltp, volume=0.0, synthetic=False)
                        )
                        app.stats.bars_real += 1
                    elif slot == last.ts:
                        b = last
                        b.high = max(b.high, ltp)
                        b.low = min(b.low, ltp)
                        b.close = ltp
                        app.cache.replace_last(b)

            bars_all = app.cache.to_list()
            bars_real = [b for b in bars_all if not getattr(b, "synthetic", False)][-cfg.chart_bars:]

            df = _bars_to_df(bars_real)
            df = _add_indicators(df, cfg)

            decision = SignalDecision("HOLD", "Нет данных для сигнала.", {})
            df_sig = pd.DataFrame()

            # 3) compute signal on CLOSED bars (exclude forming bar when session open)
            if len(df) >= 2:
                df_sig = df.iloc[:-1].copy() if session_open else df.copy()
                df_sig = _add_indicators(df_sig, cfg)
                if len(df_sig) > 0:
                    decision = compute_signal(
                        df_sig,
                        cfg.signal,
                        strategy_id=app.strategy_id,
                        runtime_state=app.strategy2,
                    )

            # 4) build "regular" chart.png (for /chart etc.)
            chart_path = cfg.cache_dir / "chart.png"
            await asyncio.to_thread(
                plot_chart,
                df,
                chart_path,
                f"{cfg.symbol} {cfg.timeframe_minutes}m",
                decision.action,
                getattr(app.stats, "signal_history", None),
            )
            app.last_chart_path = str(chart_path)

            # 5) persist cache/state periodically
            if len(app.cache) > 0 and (rolled or (now - last_persist) >= timedelta(seconds=60)):
                try:
                    app.cache.save_to_file(cache_path)
                    app.stats.cache_save = f"OK ({len(app.cache)})"
                except Exception as e:
                    app.stats.cache_save = f"ERROR: {repr(e)}"

                try:
                    app.persist_state(now)
                except Exception as e:
                    app.stats.last_error = f"persist_state: {repr(e)}"

                last_persist = now

            # 6) on bar roll -> send signal if any
            if rolled and decision.action in ("BUY", "SELL") and len(df_sig) > 0:
                sig_ts = df_sig["ts"].iloc[-1]

                sig_price = None
                try:
                    sig_price = float(df_sig["close"].iloc[-1])
                except Exception:
                    sig_price = None

                # normalize ts
                if not isinstance(sig_ts, datetime):
                    sig_ts = pd.to_datetime(sig_ts, utc=True, errors="coerce").to_pydatetime()

                # record to history BEFORE plotting for signal-message
                _record_signal(app, decision.action, sig_ts, sig_price)

                # dedup by exact action+ts
                if app.last_signal_sent == decision.action and app.last_signal_sent_ts == sig_ts:
                    pass
                else:
                    active, _left = _cooldown_active(app)
                    if active and app.last_signal_sent == decision.action:
                        app.stats.cooldown_skips += 1
                    else:
                        app.last_signal_sent = decision.action
                        app.last_signal_sent_ts = sig_ts

                        if decision.action == "BUY":
                            app.stats.signals_buy += 1
                        else:
                            app.stats.signals_sell += 1

                        app.stats.last_signal = decision.action
                        app.stats.last_signal_ts = sig_ts
                        app.stats.last_signal_price = sig_price
                        app.stats.last_signal_sent_at = utc_now()

                        # IMPORTANT: build a dedicated chart for this signal AFTER _record_signal,
                        # so triangles for current signal are already in signal_history.
                        signal_chart_path = cfg.cache_dir / f"signal_{decision.action}_{sig_ts:%Y%m%d_%H%M}.png"
                        try:
                            await asyncio.to_thread(
                                plot_chart,
                                df,  # можно df_sig, если хочешь рисовать только закрытые бары
                                signal_chart_path,
                                f"{cfg.symbol} {cfg.timeframe_minutes}m",
                                decision.action,
                                getattr(app.stats, "signal_history", None),
                            )
                        except Exception as e:
                            # если отрисовка сигнального графика упала — отправим обычный chart.png
                            app.stats.last_error = f"plot_chart(signal): {repr(e)}"
                            signal_chart_path = chart_path

                        try:
                            await send_signal_cb(decision, str(signal_chart_path), df_sig)
                            # после успешной отправки — сохраняем state (чтобы entry_ts/atr_stop не терялись)
                            try:
                                app.persist_state(now)
                            except Exception as e:
                                app.stats.last_error = f"persist_state(after_send): {repr(e)}"
                        except Exception as e:
                            app.stats.last_error = f"send_signal_cb: {repr(e)}"
                            try:
                                app.persist_state(now)
                            except Exception:
                                pass

            # НЕ затираем last_error каждую итерацию:
            # app.stats.last_error = None

        except Exception as e:
            app.stats.last_error = repr(e)

        await asyncio.sleep(cfg.poll_seconds)
