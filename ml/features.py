from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


EPS = 1e-9


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.set_index("timestamp")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex or a 'timestamp' column.")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out.sort_index()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100 - (100 / (1 + rs))


def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return lower, mid, upper


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"].fillna(0.0)
    cum_pv = pv.groupby(df.index.date).cumsum()
    cum_v = df["volume"].fillna(0.0).groupby(df.index.date).cumsum()
    return cum_pv / (cum_v + EPS)


def add_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_datetime_index(df)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    if "ema9" not in out.columns:
        out["ema9"] = ema(out["close"], 9)
    if "ema21" not in out.columns:
        out["ema21"] = ema(out["close"], 21)
    if "rsi" not in out.columns:
        out["rsi"] = rsi(out["close"], 14)
    if {"bb_lower", "bb_mid", "bb_upper"} - set(out.columns):
        bb_lower, bb_mid, bb_upper = bollinger(out["close"], 20, 2.0)
        out["bb_lower"] = bb_lower
        out["bb_mid"] = bb_mid
        out["bb_upper"] = bb_upper
    if "atr" not in out.columns:
        out["atr"] = atr(out, 14)
    if "vwap" not in out.columns:
        out["vwap"] = vwap(out)

    return out


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_base_indicators(df)

    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"].pct_change(3)
    out["ret_6"] = out["close"].pct_change(6)
    out["ret_12"] = out["close"].pct_change(12)

    out["range_pct"] = (out["high"] - out["low"]) / (out["close"] + EPS)
    out["body_pct"] = (out["close"] - out["open"]) / (out["open"] + EPS)
    out["upper_wick_pct"] = (out["high"] - np.maximum(out["open"], out["close"])) / (out["close"] + EPS)
    out["lower_wick_pct"] = (np.minimum(out["open"], out["close"]) - out["low"]) / (out["close"] + EPS)

    out["vol_ma_20"] = out["volume"].rolling(20).mean()
    out["vol_ma_50"] = out["volume"].rolling(50).mean()
    out["rel_volume_20"] = out["volume"] / (out["vol_ma_20"] + EPS)
    out["rel_volume_50"] = out["volume"] / (out["vol_ma_50"] + EPS)

    out["ema_spread"] = (out["ema9"] - out["ema21"]) / (out["close"] + EPS)
    out["price_vs_ema9"] = (out["close"] - out["ema9"]) / (out["close"] + EPS)
    out["price_vs_ema21"] = (out["close"] - out["ema21"]) / (out["close"] + EPS)
    out["price_vs_vwap"] = (out["close"] - out["vwap"]) / (out["close"] + EPS)

    out["rsi_delta_1"] = out["rsi"].diff(1)
    out["rsi_delta_3"] = out["rsi"].diff(3)

    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (out["close"] + EPS)
    out["bb_pos"] = (out["close"] - out["bb_lower"]) / ((out["bb_upper"] - out["bb_lower"]) + EPS)
    out["atr_pct"] = out["atr"] / (out["close"] + EPS)

    minute = out.index.hour * 60 + out.index.minute
    out["minute_of_day"] = minute.astype(float)
    out["hour_utc"] = out.index.hour.astype(float)
    out["day_of_week"] = out.index.dayofweek.astype(float)

    # Approximate U.S. regular session in UTC; DST is ignored here intentionally.
    out["is_regular_session"] = ((minute >= 14 * 60 + 30) & (minute <= 21 * 60)).astype(int)
    out["is_premarket"] = ((minute >= 9 * 60) & (minute < 14 * 60 + 30)).astype(int)
    out["is_afterhours"] = ((minute > 21 * 60) & (minute <= 24 * 60)).astype(int)

    session_date = pd.Series(out.index.date, index=out.index)
    out["session_open"] = out.groupby(session_date)["open"].transform("first")
    out["session_high"] = out.groupby(session_date)["high"].cummax()
    out["session_low"] = out.groupby(session_date)["low"].cummin()
    out["ret_from_session_open"] = (out["close"] - out["session_open"]) / (out["session_open"] + EPS)
    out["dist_to_session_high"] = (out["session_high"] - out["close"]) / (out["close"] + EPS)
    out["dist_to_session_low"] = (out["close"] - out["session_low"]) / (out["close"] + EPS)

    return out


def add_strategy_context_features(
    df: pd.DataFrame,
    strategy_context: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    out = df.copy()
    strategy_context = strategy_context or {}

    defaults = {
        "buy_score": 0.0,
        "sell_score": 0.0,
        "nearU": 0,
        "nearL": 0,
        "bounceU": 0,
        "bounceL": 0,
        "bb_ok": 0,
        "rsi_ok": 0,
        "ema_up": 0,
        "ema_dn": 0,
        "signal_is_buy": 0,
        "signal_is_sell": 0,
        "position_is_long": 0,
        "position_is_short": 0,
        "bars_since_last_signal": np.nan,
        "symbol_hash": 0.0,
        "timeframe_minutes": 5.0,
    }
    defaults.update(strategy_context)

    for key, value in defaults.items():
        if isinstance(value, bool):
            value = int(value)
        elif value is None:
            value = np.nan
        out[key] = value

    return out


DEFAULT_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "rel_volume_20", "rel_volume_50",
    "ema_spread", "price_vs_ema9", "price_vs_ema21", "price_vs_vwap",
    "rsi", "rsi_delta_1", "rsi_delta_3",
    "bb_width", "bb_pos", "atr_pct",
    "minute_of_day", "hour_utc", "day_of_week",
    "is_regular_session", "is_premarket", "is_afterhours",
    "ret_from_session_open", "dist_to_session_high", "dist_to_session_low",
    "buy_score", "sell_score", "nearU", "nearL", "bounceU", "bounceL",
    "bb_ok", "rsi_ok", "ema_up", "ema_dn",
    "signal_is_buy", "signal_is_sell",
    "position_is_long", "position_is_short",
    "bars_since_last_signal", "timeframe_minutes",
]


def build_feature_frame(
    df: pd.DataFrame,
    strategy_context: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    out = add_contextual_features(df)
    out = add_strategy_context_features(out, strategy_context=strategy_context)
    return out


def latest_feature_row(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    strategy_context: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or DEFAULT_FEATURES
    feats = build_feature_frame(df, strategy_context=strategy_context)
    row = feats[feature_columns].tail(1).copy()
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return row
