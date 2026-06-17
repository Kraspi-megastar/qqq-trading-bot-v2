from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def make_directional_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    threshold: float = 0.0015,
) -> pd.DataFrame:
    out = df.copy()
    future_ret = out["close"].shift(-horizon) / out["close"] - 1.0
    out["future_ret"] = future_ret
    out["target_up"] = (future_ret > threshold).astype(int)
    out["target_down"] = (future_ret < -threshold).astype(int)
    return out


def make_signal_quality_label(
    df: pd.DataFrame,
    base_signal_col: str = "base_signal",
    horizon: int = 3,
    threshold: float = 0.0015,
) -> pd.DataFrame:
    out = make_directional_labels(df, horizon=horizon, threshold=threshold)
    signal = out[base_signal_col].astype(str).str.upper().fillna("HOLD")

    out["target_signal_quality"] = 0
    out.loc[signal.eq("BUY"), "target_signal_quality"] = out.loc[signal.eq("BUY"), "target_up"]
    out.loc[signal.eq("SELL"), "target_signal_quality"] = out.loc[signal.eq("SELL"), "target_down"]
    return out
