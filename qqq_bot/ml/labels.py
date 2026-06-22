from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

EPS = 1e-9
TieBreak = Literal["neutral", "bar_close", "long", "short"]


def make_directional_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    threshold: float = 0.0015,
) -> pd.DataFrame:
    out = df.copy()
    future_ret = out["close"].shift(-horizon) / (out["close"] + EPS) - 1.0
    out["future_ret"] = future_ret
    out["target_up"] = (future_ret > threshold).astype(int)
    out["target_down"] = (future_ret < -threshold).astype(int)
    return out


def make_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    atr_mult: float = 0.5,
    min_atr_pct: float = 0.0005,
    tie_break: TieBreak = "neutral",
) -> pd.DataFrame:
    """Create first-touch ATR barrier labels.

    target_long_win=1 means the upper barrier (+atr_mult*ATR) was touched before
    the lower barrier within the next `horizon` bars. target_short_win is symmetric.
    Rows without a valid ATR or with no barrier touch are labeled 0 for both targets.
    """
    out = df.copy()
    n = len(out)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)

    if "atr" in out.columns:
        atr = pd.to_numeric(out["atr"], errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError("ATR column is required for barrier labels.")

    target_long = np.zeros(n, dtype=int)
    target_short = np.zeros(n, dtype=int)
    first_touch = np.full(n, "none", dtype=object)
    bars_to_touch = np.full(n, np.nan, dtype=float)
    upper_barrier = close + atr_mult * atr
    lower_barrier = close - atr_mult * atr

    for i in range(n):
        c = close[i]
        a = atr[i]
        if not np.isfinite(c) or not np.isfinite(a) or c <= 0 or a <= 0:
            continue
        if a / max(abs(c), EPS) < min_atr_pct:
            continue
        end = min(n, i + horizon + 1)
        if i + 1 >= end:
            continue
        up = c + atr_mult * a
        dn = c - atr_mult * a
        touched = "none"
        touched_after = np.nan
        for j in range(i + 1, end):
            hit_up = bool(high[j] >= up) if np.isfinite(high[j]) else False
            hit_dn = bool(low[j] <= dn) if np.isfinite(low[j]) else False
            if hit_up and hit_dn:
                if tie_break == "long":
                    touched = "long"
                elif tie_break == "short":
                    touched = "short"
                elif tie_break == "bar_close":
                    touched = "long" if close[j] >= c else "short"
                else:
                    touched = "tie"
                touched_after = float(j - i)
                break
            if hit_up:
                touched = "long"
                touched_after = float(j - i)
                break
            if hit_dn:
                touched = "short"
                touched_after = float(j - i)
                break
        first_touch[i] = touched
        bars_to_touch[i] = touched_after
        if touched == "long":
            target_long[i] = 1
        elif touched == "short":
            target_short[i] = 1

    future_close = out["close"].shift(-horizon)
    out["future_ret"] = future_close / (out["close"] + EPS) - 1.0
    out["upper_barrier"] = upper_barrier
    out["lower_barrier"] = lower_barrier
    out["first_touch"] = first_touch
    out["bars_to_touch"] = bars_to_touch
    out["target_long_win"] = target_long
    out["target_short_win"] = target_short
    return out


def make_signal_quality_label(
    df: pd.DataFrame,
    base_signal_col: str = "base_signal",
    horizon: int = 12,
    atr_mult: float = 0.5,
    threshold: float = 0.0015,
    label_engine: Literal["barrier", "return"] = "barrier",
) -> pd.DataFrame:
    out = df.copy()
    signal = out[base_signal_col].astype(str).str.upper().fillna("HOLD")

    if label_engine == "barrier":
        out = make_barrier_labels(out, horizon=horizon, atr_mult=atr_mult)
        out["target_signal_quality"] = 0
        out.loc[signal.eq("BUY"), "target_signal_quality"] = out.loc[signal.eq("BUY"), "target_long_win"]
        out.loc[signal.eq("SELL"), "target_signal_quality"] = out.loc[signal.eq("SELL"), "target_short_win"]
        return out

    out = make_directional_labels(out, horizon=horizon, threshold=threshold)
    out["target_signal_quality"] = 0
    out.loc[signal.eq("BUY"), "target_signal_quality"] = out.loc[signal.eq("BUY"), "target_up"]
    out.loc[signal.eq("SELL"), "target_signal_quality"] = out.loc[signal.eq("SELL"), "target_down"]
    return out
