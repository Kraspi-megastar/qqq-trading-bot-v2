"""
config.py — загрузка конфигурации из .env

Изменения v3:
  - Добавлены параметры OptionConfig (MIN_DTE, STRIKE_STEP, STRIKE_OFFSET).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class SignalConfig:
    # strategy #1
    bb_period: int
    bb_std: float
    ema_fast: int
    ema_slow: int
    rsi_period: int
    rsi_buy: float
    rsi_sell: float
    rsi_buy_bounce: float
    rsi_sell_bounce: float
    near_bb_tol: float
    bounce_lookback: int

    # strategy #2
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vwap_price_mode: str = "typical"
    supertrend_period: int = 10
    supertrend_mult: float = 3.0
    atr_period: int = 14
    atr_stop_mult: float = 3.0
    vol_ma_period: int = 20
    ema_trend_period: int = 200


@dataclass(frozen=True)
class OptionConfig:
    """Настройки генерации опционных сигналов."""
    enabled: bool = True
    min_dte: int = 1               # минимум дней до экспирации
    strike_step: float = 1.0       # шаг страйка ($1 для QQQ)
    underlying_symbol: str = "QQQ.US"
    target_delta: float = 0.375    # целевая дельта для выбора страйка (0.35–0.40)
    max_expiry_tries: int = 8      # сколько дней перебрать в поисках торгуемой экспирации
    risk_free_rate: float = 0.05   # безрисковая ставка для Black-Scholes
    require_validation: bool = False  # True = не открывать без подтверждения TraderNet
    max_dte: int = 4               # максимальный срок до экспирации (дней)
    # Временные окна RTH (минуты от начала дня по ET); опционы только в основную сессию
    open_blackout_min: int = 10    # не открывать первые N минут после 9:30
    close_blackout_min: int = 15   # не открывать/закрывать за N минут до 16:00
    force_close_min: int = 15      # принудительно закрывать за N минут до 16:00 (не держать ночь)


@dataclass(frozen=True)
class ConsensusConfig:
    """Настройки ансамбля трёх источников (#1, #2, ML)."""
    enabled: bool = True
    agree_window_bars: int = 12
    weight_s1: float = 1.0
    weight_s2: float = 1.0
    weight_ml: float = 1.0
    ml_min_edge: float = 0.05
    conflict_score_threshold: float = 1.0


@dataclass(frozen=True)
class ExecutionConfig:
    """Настройки боевого исполнения для ОДНОГО счёта. ПО УМОЛЧАНИЮ ВЫКЛЮЧЕНО."""
    enabled: bool = False
    account_id: str = "ffa"       # короткий id счёта (ffa, tfos, ...)
    label: str = "FFA"            # человекочитаемое имя для сообщений
    mode: str = "semi_auto"        # semi_auto | auto | off
    public_key: str = ""
    private_key: str = ""
    position_pct: float = 5.0
    max_position_pct: float = 10.0
    max_contracts: int = 50
    max_orders_per_day: int = 20
    max_notional_per_trade: float = 50000.0
    hold_overnight_min_dte: int = 99      # 99 = не держать ночь никогда
    block_new_position_if_dte_lte: int = 0
    require_reconcile: bool = True
    confirm_timeout_sec: int = 300
    underlying_symbol: str = "QQQ.US"


@dataclass(frozen=True)
class AppConfig:
    telegram_bot_token: str
    telegram_channel_id: int

    symbol: str
    timeframe_minutes: int

    poll_seconds: int
    chart_bars: int
    cache_dir: Path

    tradernet_api_url: str
    tradernet_quotes_url: str
    tradernet_sid: str | None
    tradernet_timeout_seconds: int

    history_lookback_days: int
    display_tz: str
    signal: SignalConfig
    option: OptionConfig
    consensus: ConsensusConfig
    executions: tuple[ExecutionConfig, ...]

    cooldown_seconds: int
    strategy_id: int


def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _default_cache_dir() -> Path:
    base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home())
    return Path(base) / "qqq_trading_bot_cache"


def _load_executions() -> tuple[ExecutionConfig, ...]:
    """
    Загружает список торговых счетов из env.

    EXEC_ACCOUNTS="ffa,tfos" — список id счетов через запятую (по умолчанию "ffa").
    Для каждого счёта переменные с префиксом: {ID}_ENABLED, {ID}_PUBLIC_KEY и т.д.
    Пример для tfos: TFOS_ENABLED, TFOS_PUBLIC_KEY, TFOS_PRIVATE_KEY, TFOS_POSITION_PCT...

    Обратная совместимость: для "ffa" при отсутствии префиксных переменных
    используются старые (EXEC_ENABLED, TRADERNET_PUBLIC_KEY, EXEC_POSITION_PCT...).
    """
    ids = [x.strip().lower() for x in os.getenv("EXEC_ACCOUNTS", "ffa").split(",") if x.strip()]
    out: list[ExecutionConfig] = []

    for acc_id in ids:
        P = acc_id.upper()  # префикс переменных

        def g(suffix: str, legacy: str | None = None, default: str = "") -> str:
            # сначала префиксная переменная, затем (для ffa) legacy, затем default
            v = os.getenv(f"{P}_{suffix}")
            if v is not None:
                return v
            if acc_id == "ffa" and legacy is not None:
                v = os.getenv(legacy)
                if v is not None:
                    return v
            return default

        label = g("LABEL", default=P)
        cfg = ExecutionConfig(
            enabled=g("ENABLED", "EXEC_ENABLED", "0") == "1",
            account_id=acc_id,
            label=label,
            mode=g("MODE", "EXEC_MODE", "semi_auto"),
            public_key=g("PUBLIC_KEY", "TRADERNET_PUBLIC_KEY", ""),
            private_key=g("PRIVATE_KEY", "TRADERNET_PRIVATE_KEY", ""),
            position_pct=float(g("POSITION_PCT", "EXEC_POSITION_PCT", "5.0")),
            max_position_pct=float(g("MAX_POSITION_PCT", "EXEC_MAX_POSITION_PCT", "10.0")),
            max_contracts=int(g("MAX_CONTRACTS", "EXEC_MAX_CONTRACTS", "50")),
            max_orders_per_day=int(g("MAX_ORDERS_PER_DAY", "EXEC_MAX_ORDERS_PER_DAY", "20")),
            max_notional_per_trade=float(g("MAX_NOTIONAL", "EXEC_MAX_NOTIONAL", "50000")),
            hold_overnight_min_dte=int(g("HOLD_OVERNIGHT_MIN_DTE", "EXEC_HOLD_OVERNIGHT_MIN_DTE", "99")),
            block_new_position_if_dte_lte=int(g("BLOCK_NEW_IF_DTE_LTE", "EXEC_BLOCK_NEW_IF_DTE_LTE", "0")),
            require_reconcile=g("REQUIRE_RECONCILE", "EXEC_REQUIRE_RECONCILE", "1") == "1",
            confirm_timeout_sec=int(g("CONFIRM_TIMEOUT_SEC", "EXEC_CONFIRM_TIMEOUT_SEC", "300")),
            underlying_symbol=os.getenv("SYMBOL", "QQQ.US"),
        )
        out.append(cfg)

    return tuple(out)


def load_config() -> AppConfig:
    load_dotenv()

    cache_dir = Path(os.getenv("CACHE_DIR", str(_default_cache_dir()))).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    signal = SignalConfig(
        bb_period=int(os.getenv("BB_PERIOD", "20")),
        bb_std=float(os.getenv("BB_STD", "2.0")),
        ema_fast=int(os.getenv("EMA_FAST", "9")),
        ema_slow=int(os.getenv("EMA_SLOW", "21")),
        rsi_period=int(os.getenv("RSI_PERIOD", "14")),
        rsi_buy=float(os.getenv("RSI_BUY", "35")),
        rsi_sell=float(os.getenv("RSI_SELL", "65")),
        rsi_buy_bounce=float(os.getenv("RSI_BUY_BOUNCE", "40")),
        rsi_sell_bounce=float(os.getenv("RSI_SELL_BOUNCE", "60")),
        near_bb_tol=float(os.getenv("NEAR_BB_TOL", "0.0025")),
        bounce_lookback=int(os.getenv("BOUNCE_LOOKBACK", "3")),
        macd_fast=int(os.getenv("MACD_FAST", "12")),
        macd_slow=int(os.getenv("MACD_SLOW", "26")),
        macd_signal=int(os.getenv("MACD_SIGNAL", "9")),
        vwap_price_mode=os.getenv("VWAP_PRICE_MODE", "typical"),
        supertrend_period=int(os.getenv("SUPERTREND_PERIOD", "10")),
        supertrend_mult=float(os.getenv("SUPERTREND_MULT", "3.0")),
        atr_period=int(os.getenv("ATR_PERIOD", "14")),
        atr_stop_mult=float(os.getenv("ATR_STOP_MULT", "3.0")),
        vol_ma_period=int(os.getenv("VOL_MA_PERIOD", "20")),
        ema_trend_period=int(os.getenv("EMA_TREND_PERIOD", "200")),
    )

    option = OptionConfig(
        enabled=os.getenv("OPTION_SIGNALS_ENABLED", "1") == "1",
        min_dte=int(os.getenv("OPTION_MIN_DTE", "1")),
        strike_step=float(os.getenv("OPTION_STRIKE_STEP", "1.0")),
        underlying_symbol=os.getenv("OPTION_UNDERLYING", "QQQ.US"),
        target_delta=float(os.getenv("OPTION_TARGET_DELTA", "0.375")),
        max_expiry_tries=int(os.getenv("OPTION_MAX_EXPIRY_TRIES", "4")),
        risk_free_rate=float(os.getenv("OPTION_RISK_FREE_RATE", "0.05")),
        require_validation=os.getenv("OPTION_REQUIRE_VALIDATION", "0") == "1",
        max_dte=int(os.getenv("OPTION_MAX_DTE", "4")),
        open_blackout_min=int(os.getenv("OPTION_OPEN_BLACKOUT_MIN", "10")),
        close_blackout_min=int(os.getenv("OPTION_CLOSE_BLACKOUT_MIN", "15")),
        force_close_min=int(os.getenv("OPTION_FORCE_CLOSE_MIN", "15")),
    )

    return AppConfig(
        telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
        telegram_channel_id=int(_require("TELEGRAM_CHANNEL_ID")),
        symbol=os.getenv("SYMBOL", "QQQ.US"),
        timeframe_minutes=int(os.getenv("TIMEFRAME_MINUTES", "5")),
        poll_seconds=int(os.getenv("POLL_SECONDS", "10")),
        chart_bars=int(os.getenv("CHART_BARS", "240")),
        cache_dir=cache_dir,
        tradernet_api_url=os.getenv("TRADERNET_API_URL", "https://tradernet.ru/api/"),
        tradernet_quotes_url=os.getenv(
            "TRADERNET_QUOTES_URL", "https://tradernet.ru/securities/export"
        ),
        tradernet_sid=os.getenv("TRADERNET_SID") or None,
        tradernet_timeout_seconds=int(os.getenv("TRADERNET_TIMEOUT_SECONDS", "20")),
        history_lookback_days=int(os.getenv("HISTORY_LOOKBACK_DAYS", "7")),
        display_tz=os.getenv("DISPLAY_TZ", "America/New_York"),
        signal=signal,
        option=option,
        executions=_load_executions(),
        consensus=ConsensusConfig(
            enabled=os.getenv("CONSENSUS_ENABLED", "1") == "1",
            agree_window_bars=int(os.getenv("CONSENSUS_WINDOW_BARS", "12")),
            weight_s1=float(os.getenv("CONSENSUS_WEIGHT_S1", "1.0")),
            weight_s2=float(os.getenv("CONSENSUS_WEIGHT_S2", "1.0")),
            weight_ml=float(os.getenv("CONSENSUS_WEIGHT_ML", "1.0")),
            ml_min_edge=float(os.getenv("CONSENSUS_ML_MIN_EDGE", "0.05")),
            conflict_score_threshold=float(os.getenv("CONSENSUS_CONFLICT_THRESHOLD", "1.0")),
        ),
        cooldown_seconds=int(os.getenv("COOLDOWN_SECONDS", "300")),
        strategy_id=int(os.getenv("STRATEGY_ID", "1")),
    )
