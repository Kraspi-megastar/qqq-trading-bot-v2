"""
scheduler.py — основной цикл опроса цен и генерации сигналов.

Изменения v3:
  - Убраны дублирующие _bars_to_df / _add_indicators / _min_bars_for_indicators;
    теперь используется pipeline.py.
  - compute_signal() больше не мутирует runtime_state; новое состояние
    применяется явно через _apply_new_state().
  - Strategy2Runtime заменён на Strategy2State из signals.py.
  - Добавлено логирование каждого решения в JSONL (logs/signals.jsonl).
  - Retry при ошибке get_quote_ltp с экспоненциальным backoff.
  - Убраны getattr-заглушки: все поля AppConfig и SignalConfig типизированы.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .cache import BarCache, Stats, cache_file_path
from .config import AppConfig
from .models import Bar
from .pipeline import bars_to_df, add_indicators, min_bars_for_indicators
from .signals import compute_signal, SignalDecision, Strategy2State
from .options import OptionPosition, OptionConfig, get_option_recommendation
from .trades import TradeJournal
from .charting import plot_chart
from .tradernet import TraderNetClient
from .state_store import apply_state_if_same_session, build_state_from_app, save_state
from .utils_time import utc_now, floor_time

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# AppState
# ────────────────────────────────────────────────────────────────────────────

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
    strategy2: Strategy2State = field(default_factory=Strategy2State)
    option_position: OptionPosition | None = None  # стратегия #1: открытая опционная позиция
    trade_journal: TradeJournal | None = None       # журнал опционных сделок

    def set_strategy(self, strategy_id: int) -> None:
        """
        Переключение стратегии. Открытая опционная позиция ПЕРЕНОСИТСЯ
        в новую стратегию (не сбрасывается) — продолжаем её вести.

        Strategy2State синхронизируется с опционной позицией: если открыт CALL,
        стратегия #2 считается LONG; если открыт PUT или FLAT — стратегия стартует FLAT
        (стратегия #2 управляет своим выходом по своим правилам только для CALL/LONG).
        """
        sid = int(strategy_id)
        if sid not in (1, 2):
            return
        self.strategy_id = sid

        # Синхронизируем Strategy2State с текущей опционной позицией
        if self.option_position is not None and self.option_position.option_type == "CALL":
            self.strategy2 = Strategy2State(
                position="LONG",
                entry_price=self.option_position.entry_underlying,
                atr_stop=None,
                entry_ts=None,
            )
        else:
            # PUT или FLAT — стратегия #2 не умеет вести PUT через свой state,
            # но позиция сохраняется и управляется через _resolve_action по сигналам
            self.strategy2 = Strategy2State()

        # сбрасываем только дедупликацию отправки, позицию НЕ трогаем
        self.last_signal_sent = None
        self.last_signal_sent_ts = None

    def persist_state(self, now_utc: datetime) -> None:
        sid = _session_id(now_utc, self.cfg.display_tz)
        st = build_state_from_app(app=self, session_id=sid)
        save_state(self.cfg.cache_dir, st)


# ────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ────────────────────────────────────────────────────────────────────────────

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
        try:
            app.stats.signal_history.clear()
        except Exception:
            app.stats.signal_history = []
        app.last_signal_sent = None
        app.last_signal_sent_ts = None
        app.strategy2 = Strategy2State()
        # option_position НЕ сбрасываем при смене дня — позиция закрывается
        # только сигналом SELL/BUY, а не молча по календарю.
        # trade_journal тоже сохраняется между сессиями.


def _is_extended_session_open(now_utc: datetime, tz_name: str) -> bool:
    """Расширенная сессия (премаркет+RTH+афтермаркет): 4:00–20:00 ET.
    Используется для сбора баров."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(tz_name)
    local = now_utc.astimezone(tz)
    if local.weekday() >= 5:
        return False
    t = local.timetz().replace(tzinfo=None)
    return time(4, 0) <= t < time(20, 0)


def _is_rth_open(now_utc: datetime, tz_name: str) -> bool:
    """Основная сессия (Regular Trading Hours): 9:30–16:00 ET, будни.
    Только в это время торгуются опционы."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    local = now_utc.astimezone(ZoneInfo(tz_name))
    if local.weekday() >= 5:
        return False
    t = local.timetz().replace(tzinfo=None)
    return time(9, 30) <= t < time(16, 0)


def _apply_new_state(app: AppState, decision: SignalDecision) -> None:
    """Применяет decision.new_state к app.strategy2 (если оно есть)."""
    if decision.new_state is not None:
        app.strategy2 = decision.new_state


def _record_signal(
    app: AppState,
    action: str,
    bar_ts_utc: datetime,
    price: float | None,
) -> None:
    try:
        hist = app.stats.signal_history
    except Exception:
        app.stats.signal_history = []
        hist = app.stats.signal_history

    p = float(price) if isinstance(price, (int, float)) else None

    if hist:
        last = hist[-1]
        last_action = last[0] if isinstance(last, (tuple, list)) and len(last) >= 1 else None
        last_ts = last[1] if isinstance(last, (tuple, list)) and len(last) >= 2 else None
        if last_action == action and last_ts == bar_ts_utc:
            return
        if last_action == action:
            return

    hist.append((action, bar_ts_utc, p))

    max_keep = 200
    if len(hist) > max_keep:
        del hist[:-max_keep]

    app.stats.last_signal = action
    app.stats.last_signal_ts = bar_ts_utc
    app.stats.last_signal_price = p


def _log_decision_jsonl(
    cfg: AppConfig,
    decision: SignalDecision,
    bar_ts: object,
    close: float | None,
) -> None:
    """Пишет каждое решение в logs/signals.jsonl для последующего бэктестинга."""
    try:
        log_dir = Path(cfg.cache_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "signals.jsonl"

        ts_str = bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts)

        record = {
            "ts": ts_str,
            "action": decision.action,
            "reason": decision.reason,
            "close": close,
            "strategy": decision.details.get("strategy"),
            "details": {
                k: v for k, v in decision.details.items()
                if isinstance(v, (int, float, bool, str, type(None)))
            },
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("_log_decision_jsonl error: %s", repr(e))


def _replay_signal_history_from_cache(app: AppState, max_signals: int = 200) -> None:
    cfg = app.cfg
    bars_real = [b for b in app.cache.to_list() if not getattr(b, "synthetic", False)]
    df = bars_to_df(bars_real)
    if df.empty:
        app.stats.signal_history = []
        return

    df = add_indicators(df, cfg)
    min_b = min_bars_for_indicators(cfg)

    last_ts = pd.to_datetime(df["ts"].iloc[-1], utc=True, errors="coerce")
    sid = _session_id(last_ts.to_pydatetime(), cfg.display_tz) if pd.notna(last_ts) else None
    setattr(app.stats, "session_id", sid)

    hist: list = []
    prev_action: str | None = None

    # replay не мутирует app.strategy2 — передаём свежую копию
    replay_state = Strategy2State()

    for i in range(min_b, len(df) + 1):
        window = df.iloc[:i]
        dec = compute_signal(window, cfg.signal, strategy_id=app.strategy_id, state=replay_state)

        # применяем new_state для корректного replay стратегии #2
        if dec.new_state is not None:
            replay_state = dec.new_state

        if dec.action not in ("BUY", "SELL"):
            continue

        ts_bar = window["ts"].iloc[-1]
        ts_bar_dt = (
            ts_bar
            if isinstance(ts_bar, datetime)
            else pd.to_datetime(ts_bar, utc=True, errors="coerce").to_pydatetime()
        )

        if sid is not None and _session_id(ts_bar_dt, cfg.display_tz) != sid:
            continue
        if prev_action == dec.action:
            continue

        bar_price = None
        try:
            bar_price = float(window["close"].iloc[-1])
        except Exception:
            pass

        hist.append((dec.action, ts_bar_dt, bar_price))
        prev_action = dec.action

    app.stats.signal_history = hist[-max_signals:]
    if app.stats.signal_history:
        last = app.stats.signal_history[-1]
        app.stats.last_signal = last[0] if len(last) >= 1 else None
        app.stats.last_signal_ts = last[1] if len(last) >= 2 else None
        app.stats.last_signal_price = last[2] if len(last) >= 3 else None


def _adjust_date_to_for_closed_session(date_to_utc: datetime) -> datetime:
    dt = date_to_utc
    while dt.weekday() >= 5:
        dt = dt - timedelta(days=1)
    return dt


def _cooldown_active(app: AppState) -> tuple[bool, int]:
    if app.stats.last_signal_sent_at is None:
        return False, 0
    cd = int(app.cfg.cooldown_seconds)
    if cd <= 0:
        return False, 0
    now = utc_now()
    left = int((app.stats.last_signal_sent_at + timedelta(seconds=cd) - now).total_seconds())
    return (left > 0), max(left, 0)


# ────────────────────────────────────────────────────────────────────────────
# Retry для get_quote_ltp
# ────────────────────────────────────────────────────────────────────────────

async def _get_ltp_with_retry(tn: TraderNetClient, symbol: str, retries: int = 3) -> float:
    delay = 0.5
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return await tn.get_quote_ltp(symbol)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
    raise last_exc  # type: ignore[misc]


# ────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ────────────────────────────────────────────────────────────────────────────

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

    min_b = min_bars_for_indicators(cfg)
    if len(app.cache) >= min_b:
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

    app.stats.bootstrap_attempt = (
        f"from={date_from.isoformat()} to={date_to.isoformat()} count={-int(bars_needed)}"
    )

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

    uniq: dict[datetime, Bar] = {
        b.ts: b for b in app.cache.to_list() if not getattr(b, "synthetic", False)
    }
    for b in bars:
        b.synthetic = False
        uniq[b.ts] = b

    merged = sorted(uniq.values(), key=lambda x: x.ts)
    app.cache.clear()
    app.cache.extend(merged[-app.cache.maxlen :])

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


# ────────────────────────────────────────────────────────────────────────────
# Главный цикл
# ────────────────────────────────────────────────────────────────────────────

def _option_cfg(cfg: AppConfig) -> OptionConfig:
    """Конвертирует AppConfig.option в options.OptionConfig."""
    o = cfg.option
    return OptionConfig(
        enabled=o.enabled,
        min_dte=o.min_dte,
        strike_step=o.strike_step,
        underlying_symbol=o.underlying_symbol,
        target_delta=getattr(o, "target_delta", 0.375),
        max_expiry_tries=getattr(o, "max_expiry_tries", 4),
        risk_free_rate=getattr(o, "risk_free_rate", 0.05),
        require_validation=getattr(o, "require_validation", False),
    )


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

            # 1) Восстановление состояния при первом тике
            if not app.state_loaded:
                sid = _session_id(now, cfg.display_tz)
                applied = apply_state_if_same_session(app=app, current_session_id=sid)
                app.state_loaded = True
                if applied:
                    app.stats.last_error = "State restored from state.json"

            _maybe_reset_session(app, now)

            session_open = _is_extended_session_open(now, cfg.display_tz)
            app.stats.session_state = "OPEN" if session_open else "CLOSED"
            app.stats.now_utc = now

            # 2) Получаем LTP с retry
            ltp = await _get_ltp_with_retry(app.tn, cfg.symbol)

            if session_open:
                slot = floor_time(now, tf)
                last = app.cache.last()

                if last is None:
                    app.cache.append(
                        Bar(ts=slot, open=ltp, high=ltp, low=ltp, close=ltp, volume=0.0, synthetic=False)
                    )
                    app.stats.bars_real += 1
                elif slot > last.ts:
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

            bars_real = [
                b for b in app.cache.to_list() if not getattr(b, "synthetic", False)
            ][-cfg.chart_bars :]

            df = bars_to_df(bars_real)
            df = add_indicators(df, cfg)

            decision = SignalDecision("HOLD", "Нет данных для сигнала.", {})
            df_sig = pd.DataFrame()

            # 3) Сигнал только по ЗАКРЫТЫМ барам (исключаем формирующийся)
            if len(df) >= 2:
                df_sig = df.iloc[:-1].copy() if session_open else df.copy()
                df_sig = add_indicators(df_sig, cfg)
                if len(df_sig) > 0:
                    decision = compute_signal(
                        df_sig,
                        cfg.signal,
                        strategy_id=app.strategy_id,
                        state=app.strategy2,
                    )
                    # Применяем новое состояние стратегии #2 (чистая функция)
                    _apply_new_state(app, decision)

            # 4) Логируем каждое решение
            try:
                sig_close: float | None = None
                if len(df_sig) > 0:
                    sig_close = float(df_sig["close"].iloc[-1])
                sig_ts_log = df_sig["ts"].iloc[-1] if len(df_sig) > 0 else now
                _log_decision_jsonl(cfg, decision, sig_ts_log, sig_close)
            except Exception:
                pass

            # 5) График
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

            # 6) Периодическое сохранение кэша и состояния
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

            # 7) При смене бара — отправляем сигнал
            if rolled and decision.action in ("BUY", "SELL") and len(df_sig) > 0:
                sig_ts = df_sig["ts"].iloc[-1]
                sig_price: float | None = None
                try:
                    sig_price = float(df_sig["close"].iloc[-1])
                except Exception:
                    pass

                if not isinstance(sig_ts, datetime):
                    sig_ts = pd.to_datetime(sig_ts, utc=True, errors="coerce").to_pydatetime()

                # Определяем что произойдёт с опционом ДО отправки.
                # Сообщение шлём только если позиция реально меняется (OPEN/CLOSE).
                # HOLD (уже в позиции) и пропуск из-за закрытого рынка — НЕ шлём.
                rec = None
                if app.cfg.option.enabled:
                    rth_open = _is_rth_open(now, cfg.display_tz)
                    atr_v = None
                    try:
                        if len(df_sig) > 0 and "atr" in df_sig.columns:
                            av = df_sig["atr"].iloc[-1]
                            atr_v = float(av) if pd.notna(av) else None
                    except Exception:
                        atr_v = None
                    try:
                        rec = await get_option_recommendation(
                            signal=decision.action,
                            underlying_price=float(sig_price) if sig_price else float(df["close"].iloc[-1]),
                            cfg=_option_cfg(cfg),
                            current_position=app.option_position,
                            market_open=rth_open,
                            atr=atr_v,
                            tn=app.tn,
                        )
                    except Exception as e:
                        app.stats.last_error = f"option_rec: {repr(e)}"
                        rec = None

                # Решаем, нужно ли вообще отправлять сообщение
                should_send = True
                if rec is not None:
                    if rec.action_type == "HOLD":
                        # уже в позиции в ту же сторону, либо пропуск — не шлём
                        should_send = False
                    if rec.skipped_market_closed and rec.action_type != "CLOSE":
                        should_send = False

                _record_signal(app, decision.action, sig_ts, sig_price)

                # Дедупликация: тот же action на том же баре
                if app.last_signal_sent == decision.action and app.last_signal_sent_ts == sig_ts:
                    should_send = False

                if not should_send:
                    # Фиксируем что обработали этот бар, но не шлём дубликат
                    app.last_signal_sent = decision.action
                    app.last_signal_sent_ts = sig_ts
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

                        signal_chart_path = (
                            cfg.cache_dir / f"signal_{decision.action}_{sig_ts:%Y%m%d_%H%M}.png"
                        )
                        try:
                            await asyncio.to_thread(
                                plot_chart,
                                df,
                                signal_chart_path,
                                f"{cfg.symbol} {cfg.timeframe_minutes}m",
                                decision.action,
                                getattr(app.stats, "signal_history", None),
                            )
                        except Exception as e:
                            app.stats.last_error = f"plot_chart(signal): {repr(e)}"
                            signal_chart_path = chart_path

                        try:
                            # Передаём готовую рекомендацию в callback
                            await send_signal_cb(decision, str(signal_chart_path), df_sig, rec)
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

        except Exception as e:
            app.stats.last_error = repr(e)
            logger.exception("polling_loop iteration error")

        await asyncio.sleep(cfg.poll_seconds)
