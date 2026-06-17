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
    """Настройки генерации опционных сигналов (стратегия #1)."""
    enabled: bool = True
    min_dte: int = 1               # минимум дней до экспирации
    strike_step: float = 1.0       # шаг страйка ($1 для QQQ)
    strike_offset: int = 0         # 0=ATM, 1=1 шаг OTM и т.д.
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
        strike_offset=int(os.getenv("OPTION_STRIKE_OFFSET", "0")),
        underlying_symbol=os.getenv("OPTION_UNDERLYING", "QQQ.US"),
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
        cooldown_seconds=int(os.getenv("COOLDOWN_SECONDS", "300")),
        strategy_id=int(os.getenv("STRATEGY_ID", "1")),
    )
