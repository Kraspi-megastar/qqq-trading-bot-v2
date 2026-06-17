from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def ema(close: pd.Series, period: int) -> pd.Series:
    period = max(1, int(period))
    return close.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    period = max(1, int(period))
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))

    # pandas>=3: fillna(method=...) удалён → используем bfill()
    out = out.bfill()
    return out.fillna(50.0)


def bollinger(close: pd.Series, period: int, std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    period = max(1, int(period))
    m = close.rolling(period).mean()
    s = close.rolling(period).std(ddof=0)
    upper = m + float(std) * s
    lower = m - float(std) * s
    return m, upper, lower


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = max(1, int(fast))
    slow = max(fast + 1, int(slow))
    signal = max(1, int(signal))

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    period = max(1, int(period))

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def vwap(
    df: pd.DataFrame,
    tz_name: str = "America/New_York",
    reset_daily: bool = True,
    price_mode: str = "typical",  # "typical" или "close"
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df.get("high", close), errors="coerce")
    low = pd.to_numeric(df.get("low", close), errors="coerce")
    vol = pd.to_numeric(df.get("volume", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)

    price = close if price_mode == "close" else (high + low + close) / 3.0

    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    local_date = ts.dt.tz_convert(tz_name).dt.date if reset_daily else pd.Series(0, index=df.index)

    vol_w = vol.copy()
    if float(vol_w.sum()) <= 0.0:
        vol_w[:] = 1.0

    pv = price * vol_w
    cum_pv = pv.groupby(local_date).cumsum()
    cum_vol = vol_w.groupby(local_date).cumsum().replace(0.0, np.nan)

    out = cum_pv / cum_vol

    # pandas>=3: fillna(method=...) удалён → bfill/ffill
    return out.bfill().ffill()


def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    period = max(1, int(period))
    multiplier = float(multiplier)

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    atr_v = atr(df, period=period)
    hl2 = (high + low) / 2.0

    basic_ub = hl2 + multiplier * atr_v
    basic_lb = hl2 - multiplier * atr_v

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    st_line = pd.Series(index=df.index, dtype=float)
    st_dir = pd.Series(index=df.index, dtype=int)

    st_dir.iloc[0] = 1
    st_line.iloc[0] = basic_lb.iloc[0]

    for i in range(1, len(df)):
        if (basic_ub.iloc[i] < final_ub.iloc[i - 1]) or (close.iloc[i - 1] > final_ub.iloc[i - 1]):
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i - 1]

        if (basic_lb.iloc[i] > final_lb.iloc[i - 1]) or (close.iloc[i - 1] < final_lb.iloc[i - 1]):
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i - 1]

        prev_dir = int(st_dir.iloc[i - 1])

        if prev_dir == 1:
            if close.iloc[i] < final_lb.iloc[i]:
                st_dir.iloc[i] = -1
                st_line.iloc[i] = final_ub.iloc[i]
            else:
                st_dir.iloc[i] = 1
                st_line.iloc[i] = final_lb.iloc[i]
        else:
            if close.iloc[i] > final_ub.iloc[i]:
                st_dir.iloc[i] = 1
                st_line.iloc[i] = final_lb.iloc[i]
            else:
                st_dir.iloc[i] = -1
                st_line.iloc[i] = final_ub.iloc[i]

    return st_line, st_dir
