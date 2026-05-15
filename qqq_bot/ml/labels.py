from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    """Return a single Series even if df has duplicated column names."""
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    value = df[col]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value


def make_directional_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    threshold: float = 0.0015,
) -> pd.DataFrame:
    out = df.copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated(keep="first")].copy()

    close = pd.to_numeric(_as_series(out, "close"), errors="coerce")
    future_ret = close.shift(-horizon) / close - 1.0

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

    signal = _as_series(out, base_signal_col, default="HOLD")
    signal = signal.astype(str).str.upper().fillna("HOLD")

    out["target_signal_quality"] = 0
    buy_mask = signal.eq("BUY")
    sell_mask = signal.eq("SELL")

    out.loc[buy_mask, "target_signal_quality"] = out.loc[buy_mask, "target_up"]
    out.loc[sell_mask, "target_signal_quality"] = out.loc[sell_mask, "target_down"]
    return out
