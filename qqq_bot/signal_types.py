from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional

LegacyAction = Literal["BUY", "SELL", "HOLD"]


class SignalType(str, Enum):
    """Semantic strategy signal types.

    Important: SELL is intentionally NOT used as a semantic signal here.
    The legacy BUY/SELL/HOLD action is derived from these explicit types.
    """

    HOLD = "HOLD"
    OPEN_LONG = "OPEN_LONG"
    CLOSE_LONG = "CLOSE_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_SHORT = "CLOSE_SHORT"


class SignalMode(str, Enum):
    NONE = "none"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    EXIT = "exit"


class TradeDirection(str, Enum):
    NONE = "none"
    LONG = "long"
    SHORT = "short"


@dataclass
class MLApproval:
    """Runtime ML probabilities and approvals.

    The outcome model predicts future move outcomes, not BUY/SELL labels:
      - long_05atr / long_10atr: probability of upward move of +0.5/+1.0 ATR
      - short_05atr / short_10atr: probability of downward move of -0.5/-1.0 ATR
    """

    enabled: bool = False
    model_loaded: bool = False
    mode: str = "advisory"
    long_05atr: float = 0.0
    long_10atr: float = 0.0
    short_05atr: float = 0.0
    short_10atr: float = 0.0
    long_ok: bool = False
    short_ok: bool = False
    reason: str = "ml_not_used"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_loaded": self.model_loaded,
            "mode": self.mode,
            "long_05atr": self.long_05atr,
            "long_10atr": self.long_10atr,
            "short_05atr": self.short_05atr,
            "short_10atr": self.short_10atr,
            "long_ok": self.long_ok,
            "short_ok": self.short_ok,
            "reason": self.reason,
        }


@dataclass
class StrategyDecision:
    """Canonical signal object consumed by Telegram and options layers."""

    signal_type: SignalType
    action: LegacyAction
    mode: SignalMode = SignalMode.NONE
    direction: TradeDirection = TradeDirection.NONE
    reason: str = ""
    bar_ts: Optional[str] = None
    close: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    vwap: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema200: Optional[float] = None
    macd_hist: Optional[float] = None
    supertrend_up: Optional[bool] = None
    regular_session: bool = False
    strong_move: bool = False
    option_open_allowed: bool = False
    ml: MLApproval = field(default_factory=MLApproval)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open_signal(self) -> bool:
        return self.signal_type in {SignalType.OPEN_LONG, SignalType.OPEN_SHORT}

    @property
    def is_close_signal(self) -> bool:
        return self.signal_type in {SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT}

    @property
    def is_long_signal(self) -> bool:
        return self.signal_type in {SignalType.OPEN_LONG, SignalType.CLOSE_SHORT}

    @property
    def is_short_signal(self) -> bool:
        return self.signal_type in {SignalType.OPEN_SHORT, SignalType.CLOSE_LONG}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "action": self.action,
            "mode": self.mode.value,
            "direction": self.direction.value,
            "reason": self.reason,
            "bar_ts": self.bar_ts,
            "close": self.close,
            "rsi": self.rsi,
            "atr": self.atr,
            "vwap": self.vwap,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "ema200": self.ema200,
            "macd_hist": self.macd_hist,
            "supertrend_up": self.supertrend_up,
            "regular_session": self.regular_session,
            "strong_move": self.strong_move,
            "option_open_allowed": self.option_open_allowed,
            "ml": self.ml.as_dict(),
            "details": self.details,
        }

    def telegram_signal_line(self) -> str:
        if self.signal_type == SignalType.OPEN_LONG:
            return "🟢 BUY / OPEN_LONG"
        if self.signal_type == SignalType.CLOSE_LONG:
            return "🔴 SELL / CLOSE_LONG"
        if self.signal_type == SignalType.OPEN_SHORT:
            return "🔴 SELL / OPEN_SHORT"
        if self.signal_type == SignalType.CLOSE_SHORT:
            return "🟢 BUY / CLOSE_SHORT"
        return "⚪ HOLD"


def legacy_action_for_signal_type(signal_type: SignalType) -> LegacyAction:
    if signal_type in {SignalType.OPEN_LONG, SignalType.CLOSE_SHORT}:
        return "BUY"
    if signal_type in {SignalType.CLOSE_LONG, SignalType.OPEN_SHORT}:
        return "SELL"
    return "HOLD"


def normalize_signal_type(value: Any) -> SignalType:
    if isinstance(value, SignalType):
        return value
    if value is None:
        return SignalType.HOLD
    value_s = str(value).strip().upper()
    aliases = {
        "BUY": SignalType.OPEN_LONG,
        "SELL": SignalType.CLOSE_LONG,  # conservative legacy fallback: SELL closes long only
        "LONG": SignalType.OPEN_LONG,
        "SHORT": SignalType.OPEN_SHORT,
        "EXIT_LONG": SignalType.CLOSE_LONG,
        "EXIT_SHORT": SignalType.CLOSE_SHORT,
    }
    return aliases.get(value_s, SignalType(value_s) if value_s in SignalType._value2member_map_ else SignalType.HOLD)
