from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9
MARKET_TZ = "America/New_York"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time_col = None
    for candidate in ("timestamp", "ts", "datetime", "date"):
        if candidate in out.columns:
            time_col = candidate
            break

    if time_col is not None:
        out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
        out = out.dropna(subset=[time_col]).set_index(time_col)

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex or one of timestamp/ts/datetime/date columns.")

    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _num(s: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ema(series: pd.Series, span: int) -> pd.Series:
    return _num(series).ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    close = _num(series)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = _num(series)
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return lower, mid, upper


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = _num(df["high"])
    low = _num(df["low"])
    close = _num(df["close"])
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = _num(series)
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def vwap(df: pd.DataFrame, market_tz: str = MARKET_TZ) -> pd.Series:
    high = _num(df["high"])
    low = _num(df["low"])
    close = _num(df["close"])
    volume = _num(df.get("volume", 0.0)).fillna(0.0)
    typical = (high + low + close) / 3.0
    local_dates = df.index.tz_convert(market_tz).date
    pv = typical * volume
    cum_pv = pv.groupby(local_dates).cumsum()
    cum_v = volume.groupby(local_dates).cumsum()
    return cum_pv / (cum_v + EPS)


def _minutes_from_midnight_local(index: pd.DatetimeIndex, market_tz: str) -> pd.Series:
    local = index.tz_convert(market_tz)
    return pd.Series(local.hour * 60 + local.minute, index=index, dtype="float64")


def _copy_alias(out: pd.DataFrame, target: str, aliases: tuple[str, ...]) -> None:
    if target in out.columns:
        return
    for alias in aliases:
        if alias in out.columns:
            out[target] = out[alias]
            return


def add_base_indicators(df: pd.DataFrame, market_tz: str = MARKET_TZ) -> pd.DataFrame:
    out = _ensure_datetime_index(df)

    required = {"open", "high", "low", "close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if "volume" not in out.columns:
        out["volume"] = 0.0

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = _num(out[col])

    _copy_alias(out, "ema_fast", ("ema9",))
    _copy_alias(out, "ema_slow", ("ema21",))
    _copy_alias(out, "ema_trend", ("ema200",))

    if "ema_fast" not in out.columns:
        out["ema_fast"] = ema(out["close"], 9)
    if "ema_slow" not in out.columns:
        out["ema_slow"] = ema(out["close"], 21)
    if "ema_trend" not in out.columns:
        out["ema_trend"] = ema(out["close"], 200)
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
        out["vwap"] = vwap(out, market_tz=market_tz)
    if "macd_hist" not in out.columns:
        m_line, m_sig, m_hist = macd(out["close"], 12, 26, 9)
        out["macd"] = out.get("macd", m_line)
        out["macd_signal"] = out.get("macd_signal", m_sig)
        out["macd_hist"] = m_hist
    if "supertrend_up" not in out.columns:
        if "supertrend_dir" in out.columns:
            out["supertrend_up"] = (_num(out["supertrend_dir"]) > 0).astype(int)
        else:
            out["supertrend_up"] = (out["close"] > out["ema_trend"]).astype(int)

    return out


def add_contextual_features(df: pd.DataFrame, market_tz: str = MARKET_TZ) -> pd.DataFrame:
    out = add_base_indicators(df, market_tz=market_tz)
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    atr_safe = out["atr"].replace(0, np.nan)

    for n in (1, 3, 6, 12, 24, 48):
        out[f"ret_{n}"] = close.pct_change(n)
        out[f"logret_{n}"] = np.log(close / close.shift(n))

    out["range_atr"] = (high - low) / (atr_safe + EPS)
    out["body_atr"] = (close - open_) / (atr_safe + EPS)
    out["upper_wick_atr"] = (high - np.maximum(open_, close)) / (atr_safe + EPS)
    out["lower_wick_atr"] = (np.minimum(open_, close) - low) / (atr_safe + EPS)

    out["realized_vol_12"] = out["logret_1"].rolling(12, min_periods=6).std(ddof=0)
    out["realized_vol_24"] = out["logret_1"].rolling(24, min_periods=12).std(ddof=0)

    out["volume_ma_20"] = out["volume"].rolling(20, min_periods=5).mean()
    out["volume_ma_50"] = out["volume"].rolling(50, min_periods=10).mean()
    out["volume_ratio_20"] = out["volume"] / (out["volume_ma_20"] + EPS)
    out["volume_ratio_50"] = out["volume"] / (out["volume_ma_50"] + EPS)

    out["rsi_delta_3"] = out["rsi"].diff(3)
    out["macd_hist_delta_3"] = out["macd_hist"].diff(3)

    out["close_vs_vwap_atr"] = (close - out["vwap"]) / (atr_safe + EPS)
    out["close_vs_ema_fast_atr"] = (close - out["ema_fast"]) / (atr_safe + EPS)
    out["close_vs_ema_slow_atr"] = (close - out["ema_slow"]) / (atr_safe + EPS)
    out["close_vs_ema_trend_atr"] = (close - out["ema_trend"]) / (atr_safe + EPS)
    out["ema_fast_vs_slow_atr"] = (out["ema_fast"] - out["ema_slow"]) / (atr_safe + EPS)

    bb_width = out["bb_upper"] - out["bb_lower"]
    out["bb_width_atr"] = bb_width / (atr_safe + EPS)
    out["bb_pos"] = (close - out["bb_lower"]) / (bb_width + EPS)

    recent_high_12 = high.shift(1).rolling(12, min_periods=3).max()
    recent_low_12 = low.shift(1).rolling(12, min_periods=3).min()
    recent_high_48 = high.shift(1).rolling(48, min_periods=12).max()
    recent_low_48 = low.shift(1).rolling(48, min_periods=12).min()
    out["dist_recent_high_12_atr"] = (recent_high_12 - close) / (atr_safe + EPS)
    out["dist_recent_low_12_atr"] = (close - recent_low_12) / (atr_safe + EPS)
    out["dist_recent_high_48_atr"] = (recent_high_48 - close) / (atr_safe + EPS)
    out["dist_recent_low_48_atr"] = (close - recent_low_48) / (atr_safe + EPS)
    out["breakout_20"] = (close > high.shift(1).rolling(20, min_periods=10).max()).astype(int)
    out["breakdown_20"] = (close < low.shift(1).rolling(20, min_periods=10).min()).astype(int)

    minute = _minutes_from_midnight_local(out.index, market_tz)
    rth_open = 9 * 60 + 30
    rth_close = 16 * 60
    out["is_rth"] = ((minute >= rth_open) & (minute < rth_close)).astype(int)
    out["is_premarket"] = ((minute >= 4 * 60) & (minute < rth_open)).astype(int)
    out["is_afterhours"] = ((minute >= rth_close) & (minute < 20 * 60)).astype(int)
    out["minutes_from_open"] = minute - rth_open
    out["minutes_to_close"] = rth_close - minute
    out["minute_of_day_sin"] = np.sin(2.0 * math.pi * minute / 1440.0)
    out["minute_of_day_cos"] = np.cos(2.0 * math.pi * minute / 1440.0)

    local = out.index.tz_convert(market_tz)
    dow = pd.Series(local.dayofweek, index=out.index, dtype="float64")
    out["day_of_week_sin"] = np.sin(2.0 * math.pi * dow / 5.0)
    out["day_of_week_cos"] = np.cos(2.0 * math.pi * dow / 5.0)

    session_date = pd.Series(local.date, index=out.index)
    session_open = out.groupby(session_date)["open"].transform("first")
    session_high = out.groupby(session_date)["high"].cummax()
    session_low = out.groupby(session_date)["low"].cummin()
    out["ret_from_session_open"] = close / (session_open + EPS) - 1.0
    out["dist_session_high_atr"] = (session_high - close) / (atr_safe + EPS)
    out["dist_session_low_atr"] = (close - session_low) / (atr_safe + EPS)

    return out


def add_strategy_context_features(
    df: pd.DataFrame,
    strategy_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    ctx = dict(strategy_context or {})
    defaults: dict[str, Any] = {
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
        "strategy_id": 0,
        "timeframe_minutes": 5.0,
    }
    defaults.update(ctx)

    for key, value in defaults.items():
        if key in {"symbol", "timeframe", "session"}:
            continue
        if isinstance(value, bool):
            value = int(value)
        if value is None:
            value = np.nan
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


DEFAULT_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
    "logret_1", "logret_3", "logret_6", "logret_12", "logret_24", "logret_48",
    "range_atr", "body_atr", "upper_wick_atr", "lower_wick_atr",
    "realized_vol_12", "realized_vol_24",
    "volume_ratio_20", "volume_ratio_50",
    "rsi", "rsi_delta_3", "macd_hist", "macd_hist_delta_3", "atr", "atr_pct",
    "close_vs_vwap_atr", "close_vs_ema_fast_atr", "close_vs_ema_slow_atr",
    "close_vs_ema_trend_atr", "ema_fast_vs_slow_atr",
    "bb_width_atr", "bb_pos",
    "dist_recent_high_12_atr", "dist_recent_low_12_atr",
    "dist_recent_high_48_atr", "dist_recent_low_48_atr",
    "breakout_20", "breakdown_20", "supertrend_up",
    "is_rth", "is_premarket", "is_afterhours",
    "minutes_from_open", "minutes_to_close", "minute_of_day_sin", "minute_of_day_cos",
    "day_of_week_sin", "day_of_week_cos",
    "ret_from_session_open", "dist_session_high_atr", "dist_session_low_atr",
    "buy_score", "sell_score", "nearU", "nearL", "bounceU", "bounceL",
    "bb_ok", "rsi_ok", "ema_up", "ema_dn", "signal_is_buy", "signal_is_sell",
    "position_is_long", "position_is_short", "strategy_id", "timeframe_minutes",
]


def build_feature_frame(
    df: pd.DataFrame,
    strategy_context: dict[str, Any] | None = None,
    market_tz: str = MARKET_TZ,
) -> pd.DataFrame:
    out = add_contextual_features(df, market_tz=market_tz)
    out["atr_pct"] = out["atr"] / (out["close"] + EPS)
    out = add_strategy_context_features(out, strategy_context=strategy_context)
    return out


def latest_feature_row(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    strategy_context: dict[str, Any] | None = None,
    market_tz: str = MARKET_TZ,
) -> pd.DataFrame:
    feature_columns = feature_columns or DEFAULT_FEATURES
    feats = build_feature_frame(df, strategy_context=strategy_context, market_tz=market_tz)
    missing = [c for c in feature_columns if c not in feats.columns]
    for col in missing:
        feats[col] = 0.0
    row = feats[feature_columns].tail(1).copy()
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return row
