from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .market_session import is_regular_session
from .signal_types import (
    MLApproval,
    SignalMode,
    SignalType,
    StrategyDecision,
    TradeDirection,
    legacy_action_for_signal_type,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


@dataclass
class Strategy2Config:
    regular_only: bool = True
    allow_short_entries: bool = True
    rsi_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 200
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    supertrend_period: int = 10
    supertrend_mult: float = 3.0
    volume_ma_period: int = 20
    breakout_lookback: int = 12
    pullback_lookback: int = 5
    min_volume_ratio_breakout: float = 1.05
    min_atr_pct_breakout: float = 0.0012
    long_rsi_min: float = 52.0
    short_rsi_max: float = 48.0
    breakout_rsi_min: float = 55.0
    breakout_short_rsi_max: float = 45.0
    max_extension_atr_pullback: float = 1.2
    exit_on_macd_zero: bool = True
    exit_on_supertrend_flip: bool = True
    exit_on_vwap_loss: bool = True

    @classmethod
    def from_env(cls) -> "Strategy2Config":
        return cls(
            regular_only=_env_bool("STR2_REGULAR_ONLY", True),
            allow_short_entries=_env_bool("STR2_ALLOW_SHORT_ENTRIES", True),
            rsi_period=_env_int("RSI_PERIOD", 14),
            ema_fast=_env_int("EMA_FAST", 9),
            ema_slow=_env_int("EMA_SLOW", 21),
            ema_trend=_env_int("STR2_EMA_TREND", 200),
            bb_period=_env_int("BB_PERIOD", 20),
            bb_std=_env_float("BB_STD", 2.0),
            atr_period=_env_int("STR2_ATR_PERIOD", 14),
            supertrend_period=_env_int("STR2_SUPERTREND_PERIOD", 10),
            supertrend_mult=_env_float("STR2_SUPERTREND_MULT", 3.0),
            volume_ma_period=_env_int("STR2_VOLUME_MA_PERIOD", 20),
            breakout_lookback=_env_int("STR2_BREAKOUT_LOOKBACK", 12),
            pullback_lookback=_env_int("STR2_PULLBACK_LOOKBACK", 5),
            min_volume_ratio_breakout=_env_float("STR2_MIN_VOLUME_RATIO_BREAKOUT", 1.05),
            min_atr_pct_breakout=_env_float("STR2_MIN_ATR_PCT_BREAKOUT", 0.0012),
            long_rsi_min=_env_float("STR2_LONG_RSI_MIN", 52.0),
            short_rsi_max=_env_float("STR2_SHORT_RSI_MAX", 48.0),
            breakout_rsi_min=_env_float("STR2_BREAKOUT_RSI_MIN", 55.0),
            breakout_short_rsi_max=_env_float("STR2_BREAKOUT_SHORT_RSI_MAX", 45.0),
            max_extension_atr_pullback=_env_float("STR2_MAX_EXTENSION_ATR_PULLBACK", 1.2),
        )


def _column(df: pd.DataFrame, *names: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"Missing required column. Tried: {names}; available={list(df.columns)}")


def normalize_bars(bars: Any) -> pd.DataFrame:
    if isinstance(bars, pd.DataFrame):
        df = bars.copy()
    else:
        df = pd.DataFrame(bars)
    if df.empty:
        return df

    rename = {}
    for target, candidates in {
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c", "last"),
        "volume": ("volume", "vol", "v"),
        "ts": ("ts", "time", "timestamp", "datetime", "bar_ts", "BarTS"),
    }.items():
        for c in candidates:
            if c in df.columns:
                rename[c] = target
                break
            for real in df.columns:
                if str(real).lower() == c.lower():
                    rename[real] = target
                    break
            if target in rename.values():
                break
    df = df.rename(columns=rename)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise KeyError(f"bars must contain {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["ts"] = df.index.astype(str)
        else:
            raise KeyError("bars must contain timestamp column: ts/time/timestamp/bar_ts")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=max(2, period // 2)).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    sig = ema(macd_line, signal)
    hist = macd_line - sig
    return macd_line, sig, hist


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period, min_periods=max(3, period // 2)).mean()
    sd = close.rolling(period, min_periods=max(3, period // 2)).std(ddof=0)
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    return lower, mid, upper


def vwap(df: pd.DataFrame) -> pd.Series:
    # Intraday cumulative VWAP. Reset by New York date if timestamps are parseable.
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)
    try:
        ts = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
        groups = ts.dt.date
    except Exception:
        groups = pd.Series(0, index=df.index)
    pv = typical * vol.fillna(0.0)
    cum_pv = pv.groupby(groups).cumsum()
    cum_vol = df["volume"].groupby(groups).cumsum().replace(0, np.nan)
    out = cum_pv / cum_vol
    return out.fillna(typical)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    atr_s = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    basic_upper = hl2 + multiplier * atr_s
    basic_lower = hl2 - multiplier * atr_s
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend_up = pd.Series(True, index=df.index)

    for i in range(1, len(df)):
        if pd.isna(atr_s.iloc[i]):
            trend_up.iloc[i] = trend_up.iloc[i - 1]
            continue
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if trend_up.iloc[i - 1]:
            trend_up.iloc[i] = not (df["close"].iloc[i] < final_lower.iloc[i])
        else:
            trend_up.iloc[i] = df["close"].iloc[i] > final_upper.iloc[i]
    return trend_up, final_lower, final_upper


def add_indicators(df: pd.DataFrame, cfg: Strategy2Config) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg.ema_fast)
    out["ema_slow"] = ema(out["close"], cfg.ema_slow)
    out["ema200"] = ema(out["close"], cfg.ema_trend)
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["vwap"] = vwap(out)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(
        out["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = bollinger(out["close"], cfg.bb_period, cfg.bb_std)
    out["volume_ma"] = out["volume"].rolling(cfg.volume_ma_period, min_periods=3).mean().replace(0, np.nan)
    out["volume_ratio"] = (out["volume"] / out["volume_ma"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["supertrend_up"], out["supertrend_lower"], out["supertrend_upper"] = supertrend(
        out, cfg.supertrend_period, cfg.supertrend_mult
    )
    lb = max(2, cfg.breakout_lookback)
    out["recent_high"] = out["high"].shift(1).rolling(lb, min_periods=2).max()
    out["recent_low"] = out["low"].shift(1).rolling(lb, min_periods=2).min()
    return out


def _ml_from_any(obj: Any) -> MLApproval:
    if obj is None:
        return MLApproval(enabled=False, model_loaded=False, reason="ml_not_passed")
    if isinstance(obj, MLApproval):
        return obj
    if isinstance(obj, dict):
        return MLApproval(
            enabled=bool(obj.get("enabled", True)),
            model_loaded=bool(obj.get("model_loaded", obj.get("loaded", True))),
            mode=str(obj.get("mode", "advisory")),
            long_05atr=float(obj.get("long_05atr", obj.get("long", 0.0)) or 0.0),
            long_10atr=float(obj.get("long_10atr", 0.0) or 0.0),
            short_05atr=float(obj.get("short_05atr", obj.get("short", 0.0)) or 0.0),
            short_10atr=float(obj.get("short_10atr", 0.0) or 0.0),
            long_ok=bool(obj.get("long_ok", False)),
            short_ok=bool(obj.get("short_ok", False)),
            reason=str(obj.get("reason", "ml_dict")),
        )
    return MLApproval(enabled=True, model_loaded=True, reason="ml_object_unknown")


def generate_str2_decision(
    bars: Any,
    *,
    cfg: Optional[Strategy2Config] = None,
    ml_decision: Any = None,
    current_position: str = "FLAT",
) -> StrategyDecision:
    """Generate explicit STR#2 decision.

    Indicators are calculated on the full provided dataset, including extended-hours bars.
    Signal generation is gated to regular session by default.
    """
    cfg = cfg or Strategy2Config.from_env()
    ml = _ml_from_any(ml_decision)
    df = normalize_bars(bars)
    if len(df) < max(cfg.ema_slow, cfg.bb_period, cfg.atr_period, cfg.supertrend_period) + 5:
        return StrategyDecision(
            signal_type=SignalType.HOLD,
            action="HOLD",
            reason=f"STR#2 HOLD: not enough bars ({len(df)})",
            ml=ml,
        )

    ind = add_indicators(df, cfg)
    row = ind.iloc[-1]
    prev = ind.iloc[-2]
    ts = str(row["ts"])
    regular = is_regular_session(ts)
    close = float(row["close"])
    atr_v = float(row["atr"] if not pd.isna(row["atr"]) else 0.0)
    rsi_v = float(row["rsi"])
    vwap_v = float(row["vwap"])
    ema_fast_v = float(row["ema_fast"])
    ema_slow_v = float(row["ema_slow"])
    ema200_v = float(row["ema200"])
    macd_hist_v = float(row["macd_hist"] if not pd.isna(row["macd_hist"]) else 0.0)
    atr_pct_v = float(row["atr_pct"] if not pd.isna(row["atr_pct"]) else 0.0)
    volume_ratio_v = float(row["volume_ratio"] if not pd.isna(row["volume_ratio"]) else 1.0)
    super_up = bool(row["supertrend_up"])

    base_details: Dict[str, Any] = {
        "volume_ratio": volume_ratio_v,
        "atr_pct": atr_pct_v,
        "recent_high": None if pd.isna(row["recent_high"]) else float(row["recent_high"]),
        "recent_low": None if pd.isna(row["recent_low"]) else float(row["recent_low"]),
        "bb_lower": None if pd.isna(row["bb_lower"]) else float(row["bb_lower"]),
        "bb_mid": None if pd.isna(row["bb_mid"]) else float(row["bb_mid"]),
        "bb_upper": None if pd.isna(row["bb_upper"]) else float(row["bb_upper"]),
    }

    if cfg.regular_only and not regular:
        return StrategyDecision(
            signal_type=SignalType.HOLD,
            action="HOLD",
            reason="STR#2 HOLD: outside regular session; indicators still use extended-hours data",
            bar_ts=ts,
            close=close,
            rsi=rsi_v,
            atr=atr_v,
            vwap=vwap_v,
            ema_fast=ema_fast_v,
            ema_slow=ema_slow_v,
            ema200=ema200_v,
            macd_hist=macd_hist_v,
            supertrend_up=super_up,
            regular_session=regular,
            ml=ml,
            details=base_details,
        )

    # Exit logic is evaluated before new entries when a position is known.
    pos = str(current_position or "FLAT").upper()
    long_exit = (
        (cfg.exit_on_macd_zero and macd_hist_v <= 0)
        or (cfg.exit_on_supertrend_flip and not super_up)
        or (cfg.exit_on_vwap_loss and close < vwap_v and rsi_v < 50)
    )
    short_exit = (
        (cfg.exit_on_macd_zero and macd_hist_v >= 0)
        or (cfg.exit_on_supertrend_flip and super_up)
        or (cfg.exit_on_vwap_loss and close > vwap_v and rsi_v > 50)
    )
    if pos in {"LONG", "CALL", "OPEN_LONG"} and long_exit:
        return _decision(
            SignalType.CLOSE_LONG,
            SignalMode.EXIT,
            TradeDirection.LONG,
            "STR#2 CLOSE_LONG: macd_zero_exit or trend/vwap exit",
            ts,
            row,
            regular,
            strong=False,
            option_allowed=False,
            ml=ml,
            details=base_details,
        )
    if pos in {"SHORT", "PUT", "OPEN_SHORT"} and short_exit:
        return _decision(
            SignalType.CLOSE_SHORT,
            SignalMode.EXIT,
            TradeDirection.SHORT,
            "STR#2 CLOSE_SHORT: macd_zero_exit or trend/vwap exit",
            ts,
            row,
            regular,
            strong=False,
            option_allowed=False,
            ml=ml,
            details=base_details,
        )

    recent_high = row["recent_high"]
    recent_low = row["recent_low"]
    macd_cross_up = row["macd_hist"] > 0 and prev["macd_hist"] <= 0
    macd_cross_down = row["macd_hist"] < 0 and prev["macd_hist"] >= 0
    trend_up = close > vwap_v and close > ema200_v and super_up and ema_fast_v >= ema_slow_v
    trend_down = close < vwap_v and close < ema200_v and (not super_up) and ema_fast_v <= ema_slow_v
    high_break = not pd.isna(recent_high) and close > float(recent_high)
    low_break = not pd.isna(recent_low) and close < float(recent_low)
    strong_long = (
        trend_up
        and (high_break or macd_cross_up)
        and rsi_v >= cfg.breakout_rsi_min
        and volume_ratio_v >= cfg.min_volume_ratio_breakout
        and atr_pct_v >= cfg.min_atr_pct_breakout
    )
    strong_short = (
        trend_down
        and (low_break or macd_cross_down)
        and rsi_v <= cfg.breakout_short_rsi_max
        and volume_ratio_v >= cfg.min_volume_ratio_breakout
        and atr_pct_v >= cfg.min_atr_pct_breakout
    )

    # Pullback mode: trend already exists, entry after controlled reset to EMA/VWAP.
    dist_fast_atr = abs(close - ema_fast_v) / atr_v if atr_v > 0 else 99.0
    touched_fast_recent = (ind["low"].tail(cfg.pullback_lookback) <= ind["ema_fast"].tail(cfg.pullback_lookback)).any()
    touched_fast_recent_short = (ind["high"].tail(cfg.pullback_lookback) >= ind["ema_fast"].tail(cfg.pullback_lookback)).any()
    pullback_long = (
        trend_up
        and macd_hist_v > 0
        and rsi_v >= cfg.long_rsi_min
        and touched_fast_recent
        and close > ema_fast_v
        and dist_fast_atr <= cfg.max_extension_atr_pullback
    )
    pullback_short = (
        cfg.allow_short_entries
        and trend_down
        and macd_hist_v < 0
        and rsi_v <= cfg.short_rsi_max
        and touched_fast_recent_short
        and close < ema_fast_v
        and dist_fast_atr <= cfg.max_extension_atr_pullback
    )

    if strong_long:
        return _decision(
            SignalType.OPEN_LONG,
            SignalMode.BREAKOUT,
            TradeDirection.LONG,
            "STR#2 OPEN_LONG breakout: MACD↑ + close>VWAP/EMA200 + ST↑ + volume/ATR filters OK",
            ts,
            row,
            regular,
            strong=True,
            option_allowed=True,
            ml=ml,
            details=base_details,
        )
    if cfg.allow_short_entries and strong_short:
        return _decision(
            SignalType.OPEN_SHORT,
            SignalMode.BREAKOUT,
            TradeDirection.SHORT,
            "STR#2 OPEN_SHORT breakout: MACD↓ + close<VWAP/EMA200 + ST↓ + volume/ATR filters OK",
            ts,
            row,
            regular,
            strong=True,
            option_allowed=True,
            ml=ml,
            details=base_details,
        )
    if pullback_long:
        return _decision(
            SignalType.OPEN_LONG,
            SignalMode.PULLBACK,
            TradeDirection.LONG,
            "STR#2 OPEN_LONG pullback: trend up + EMA/VWAP reset + momentum resumed",
            ts,
            row,
            regular,
            strong=False,
            option_allowed=True,
            ml=ml,
            details=base_details,
        )
    if pullback_short:
        return _decision(
            SignalType.OPEN_SHORT,
            SignalMode.PULLBACK,
            TradeDirection.SHORT,
            "STR#2 OPEN_SHORT pullback: trend down + EMA/VWAP reset + momentum resumed",
            ts,
            row,
            regular,
            strong=False,
            option_allowed=True,
            ml=ml,
            details=base_details,
        )

    return _decision(
        SignalType.HOLD,
        SignalMode.NONE,
        TradeDirection.NONE,
        "STR#2 HOLD: no breakout/pullback setup",
        ts,
        row,
        regular,
        strong=False,
        option_allowed=False,
        ml=ml,
        details=base_details,
    )


def _decision(
    signal_type: SignalType,
    mode: SignalMode,
    direction: TradeDirection,
    reason: str,
    ts: str,
    row: pd.Series,
    regular: bool,
    *,
    strong: bool,
    option_allowed: bool,
    ml: MLApproval,
    details: Dict[str, Any],
) -> StrategyDecision:
    return StrategyDecision(
        signal_type=signal_type,
        action=legacy_action_for_signal_type(signal_type),
        mode=mode,
        direction=direction,
        reason=reason,
        bar_ts=ts,
        close=float(row["close"]),
        rsi=float(row["rsi"]),
        atr=float(row["atr"] if not pd.isna(row["atr"]) else 0.0),
        vwap=float(row["vwap"]),
        ema_fast=float(row["ema_fast"]),
        ema_slow=float(row["ema_slow"]),
        ema200=float(row["ema200"]),
        macd_hist=float(row["macd_hist"] if not pd.isna(row["macd_hist"]) else 0.0),
        supertrend_up=bool(row["supertrend_up"]),
        regular_session=regular,
        strong_move=strong,
        option_open_allowed=option_allowed,
        ml=ml,
        details=details,
    )


# -----------------------------------------------------------------------------
# Backward-compatible project API
# -----------------------------------------------------------------------------
# Older scheduler.py / handlers.py import SignalDecision and compute_signal from
# this module. Keep that API, but return the richer StrategyDecision object.
SignalDecision = StrategyDecision


def _bool(v: Any) -> bool:
    return bool(v) if v is not None else False


def _cross_up(prev_a: float, prev_b: float, a: float, b: float) -> bool:
    return (prev_a <= prev_b) and (a > b)


def _compute_strategy_1(df: pd.DataFrame, cfg: Any) -> StrategyDecision:
    """Original RSI/EMA/Bollinger strategy, returned as StrategyDecision."""
    if df is None or len(df) == 0:
        return StrategyDecision(SignalType.HOLD, "HOLD", reason="Нет данных.", details={"strategy": 1})

    last = df.iloc[-1]
    price = float(last.get("close"))
    rsi_v = float(last.get("rsi")) if pd.notna(last.get("rsi")) else float("nan")
    ema_f = float(last.get("ema_fast")) if pd.notna(last.get("ema_fast")) else float("nan")
    ema_s = float(last.get("ema_slow")) if pd.notna(last.get("ema_slow")) else float("nan")
    bb_l = float(last.get("bb_lower")) if pd.notna(last.get("bb_lower")) else float("nan")
    bb_m = float(last.get("bb_mid")) if pd.notna(last.get("bb_mid")) else float("nan")
    bb_u = float(last.get("bb_upper")) if pd.notna(last.get("bb_upper")) else float("nan")
    ts = str(last.get("ts", ""))

    ema_up = ema_f > ema_s
    ema_dn = ema_f < ema_s

    thr = price * float(getattr(cfg, "near_bb_tol", 0.0025))
    near_l = ((price - bb_l) <= thr) and (price <= bb_m)
    near_u = ((bb_u - price) <= thr) and (price >= bb_m)

    n = int(getattr(cfg, "bounce_lookback", 3))
    bounce_l = False
    bounce_u = False
    if len(df) >= n + 1 and all(col in df.columns for col in ["close", "bb_lower", "bb_upper"]):
        prev = df.iloc[-(n + 1):-1]
        prev_close = prev["close"].astype(float)
        prev_l = prev["bb_lower"].astype(float)
        prev_u = prev["bb_upper"].astype(float)
        bounce_l = ((prev_close < prev_l).any()) and (price > bb_l)
        bounce_u = ((prev_close > prev_u).any()) and (price < bb_u)

    bb_buy_ok = near_l or bounce_l
    bb_sell_ok = near_u or bounce_u

    rsi_buy_ok = (near_l and rsi_v <= float(getattr(cfg, "rsi_buy", 40))) or (
        bounce_l and rsi_v <= float(getattr(cfg, "rsi_buy_bounce", 45))
    )
    rsi_sell_ok = (near_u and rsi_v >= float(getattr(cfg, "rsi_sell", 60))) or (
        bounce_u and rsi_v >= float(getattr(cfg, "rsi_sell_bounce", 55))
    )

    buy_score = int(ema_up) + int(bb_buy_ok) + int(rsi_buy_ok)
    sell_score = int(ema_dn) + int(bb_sell_ok) + int(rsi_sell_ok)

    buy = (buy_score >= 2) and rsi_buy_ok
    sell = (sell_score >= 2) and rsi_sell_ok

    details = {
        "strategy": 1,
        "rsi": rsi_v,
        "price": price,
        "nearL": near_l,
        "nearU": near_u,
        "bounceL": bounce_l,
        "bounceU": bounce_u,
        "ema_up": ema_up,
        "ema_dn": ema_dn,
        "buy_score": buy_score,
        "sell_score": sell_score,
    }

    if buy and not sell:
        return StrategyDecision(
            SignalType.OPEN_LONG,
            "BUY",
            mode=SignalMode.PULLBACK,
            direction=TradeDirection.LONG,
            reason=f"BUY: 2/3 подтверждений (RSI обязателен). buy_score={buy_score}",
            bar_ts=ts,
            close=price,
            rsi=rsi_v,
            regular_session=is_regular_session(ts) if ts else False,
            option_open_allowed=False,
            details=details,
        )
    if sell and not buy:
        return StrategyDecision(
            SignalType.OPEN_SHORT,
            "SELL",
            mode=SignalMode.PULLBACK,
            direction=TradeDirection.SHORT,
            reason=f"SELL: 2/3 подтверждений (RSI обязателен). sell_score={sell_score}",
            bar_ts=ts,
            close=price,
            rsi=rsi_v,
            regular_session=is_regular_session(ts) if ts else False,
            option_open_allowed=False,
            details=details,
        )

    return StrategyDecision(
        SignalType.HOLD,
        "HOLD",
        reason=f"HOLD: buy_score={buy_score}, sell_score={sell_score}",
        bar_ts=ts,
        close=price,
        rsi=rsi_v,
        regular_session=is_regular_session(ts) if ts else False,
        details=details,
    )


def _strategy2_config_from_signal_config(cfg: Any) -> Strategy2Config:
    """Map AppConfig.signal + env overrides into new STR#2 config."""
    base = Strategy2Config.from_env()
    return Strategy2Config(
        regular_only=base.regular_only,
        allow_short_entries=base.allow_short_entries,
        rsi_period=int(getattr(cfg, "rsi_period", base.rsi_period)),
        ema_fast=int(getattr(cfg, "ema_fast", base.ema_fast)),
        ema_slow=int(getattr(cfg, "ema_slow", base.ema_slow)),
        ema_trend=int(getattr(cfg, "ema_trend_period", base.ema_trend)),
        bb_period=int(getattr(cfg, "bb_period", base.bb_period)),
        bb_std=float(getattr(cfg, "bb_std", base.bb_std)),
        macd_fast=int(getattr(cfg, "macd_fast", base.macd_fast)),
        macd_slow=int(getattr(cfg, "macd_slow", base.macd_slow)),
        macd_signal=int(getattr(cfg, "macd_signal", base.macd_signal)),
        atr_period=int(getattr(cfg, "atr_period", base.atr_period)),
        supertrend_period=int(getattr(cfg, "supertrend_period", base.supertrend_period)),
        supertrend_mult=float(getattr(cfg, "supertrend_mult", base.supertrend_mult)),
        volume_ma_period=int(getattr(cfg, "vol_ma_period", base.volume_ma_period)),
        breakout_lookback=base.breakout_lookback,
        pullback_lookback=base.pullback_lookback,
        min_volume_ratio_breakout=base.min_volume_ratio_breakout,
        min_atr_pct_breakout=base.min_atr_pct_breakout,
        long_rsi_min=base.long_rsi_min,
        short_rsi_max=base.short_rsi_max,
        breakout_rsi_min=base.breakout_rsi_min,
        breakout_short_rsi_max=base.breakout_short_rsi_max,
        max_extension_atr_pullback=base.max_extension_atr_pullback,
        exit_on_macd_zero=base.exit_on_macd_zero,
        exit_on_supertrend_flip=base.exit_on_supertrend_flip,
        exit_on_vwap_loss=base.exit_on_vwap_loss,
    )


def _parse_bar_ts(ts: Any):
    try:
        return pd.to_datetime(ts, utc=True, errors="coerce").to_pydatetime()
    except Exception:
        return None


def _mutate_runtime_state(runtime_state: Any, decision: StrategyDecision, cfg: Any) -> None:
    """Keep existing AppState.strategy2 runtime compatible with explicit signals."""
    if runtime_state is None:
        return

    close = float(decision.close) if decision.close is not None else None
    atr_v = float(decision.atr) if decision.atr is not None else None
    stop_mult = float(getattr(cfg, "atr_stop_mult", 3.0))
    ts_dt = _parse_bar_ts(decision.bar_ts)

    def long_stop() -> Optional[float]:
        if close is None or atr_v is None or atr_v <= 0:
            return None
        return close - stop_mult * atr_v

    def short_stop() -> Optional[float]:
        if close is None or atr_v is None or atr_v <= 0:
            return None
        return close + stop_mult * atr_v

    current_pos = str(getattr(runtime_state, "position", "FLAT") or "FLAT").upper()

    if decision.signal_type == SignalType.OPEN_LONG:
        runtime_state.position = "LONG"
        runtime_state.entry_price = close
        runtime_state.entry_ts = ts_dt
        runtime_state.atr_stop = long_stop()
        return

    if decision.signal_type == SignalType.CLOSE_LONG and current_pos in {"LONG", "CALL", "OPEN_LONG"}:
        runtime_state.position = "FLAT"
        runtime_state.entry_price = None
        runtime_state.entry_ts = None
        runtime_state.atr_stop = None
        return

    if decision.signal_type == SignalType.OPEN_SHORT:
        runtime_state.position = "SHORT"
        runtime_state.entry_price = close
        runtime_state.entry_ts = ts_dt
        runtime_state.atr_stop = short_stop()
        return

    if decision.signal_type == SignalType.CLOSE_SHORT and current_pos in {"SHORT", "PUT", "OPEN_SHORT"}:
        runtime_state.position = "FLAT"
        runtime_state.entry_price = None
        runtime_state.entry_ts = None
        runtime_state.atr_stop = None
        return

    # HOLD: trail the ATR stop for diagnostics/persistence only.
    if decision.signal_type == SignalType.HOLD and close is not None and atr_v is not None and atr_v > 0:
        if current_pos == "LONG":
            cand = long_stop()
            old = getattr(runtime_state, "atr_stop", None)
            runtime_state.atr_stop = max(float(old), float(cand)) if old is not None and cand is not None else cand
        elif current_pos == "SHORT":
            cand = short_stop()
            old = getattr(runtime_state, "atr_stop", None)
            runtime_state.atr_stop = min(float(old), float(cand)) if old is not None and cand is not None else cand


def compute_signal(
    df: pd.DataFrame,
    cfg: Any,
    strategy_id: int = 1,
    runtime_state: Any | None = None,
    ml_decision: Any = None,
    current_position: str | None = None,
) -> StrategyDecision:
    """Project-compatible signal entrypoint.

    Strategy #2 returns explicit semantic signal types:
    OPEN_LONG / CLOSE_LONG / OPEN_SHORT / CLOSE_SHORT / HOLD.
    The legacy .action remains BUY/SELL/HOLD for charting and Telegram compatibility.
    """
    if int(strategy_id) != 2:
        return _compute_strategy_1(df, cfg)

    pos = current_position or getattr(runtime_state, "position", "FLAT") if runtime_state is not None else (current_position or "FLAT")
    s2cfg = _strategy2_config_from_signal_config(cfg)
    decision = generate_str2_decision(df, cfg=s2cfg, ml_decision=ml_decision, current_position=pos)
    _mutate_runtime_state(runtime_state, decision, cfg)
    return decision


def generate_signal(bars: Any, **kwargs: Any) -> StrategyDecision:
    return generate_str2_decision(bars, **kwargs)
