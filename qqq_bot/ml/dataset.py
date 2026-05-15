from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .features import DEFAULT_FEATURES, build_feature_frame
from .labels import make_directional_labels, make_signal_quality_label


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df.copy()


def _as_series(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    value = df[col]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value


def _normalize_signal_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = _dedupe_columns(df)
    if "base_signal" not in out.columns:
        out["base_signal"] = "HOLD"

    base = _as_series(out, "base_signal", default="HOLD").astype(str).str.upper().fillna("HOLD")
    out["base_signal"] = base
    out["signal_is_buy"] = base.eq("BUY").astype(int)
    out["signal_is_sell"] = base.eq("SELL").astype(int)
    return out


def _numeric_extra_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("decision_") or col.startswith("detail_"):
            continue
        series = _as_series(df, col)
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            cols.append(col)
    return cols


def prepare_training_dataset(
    bars: pd.DataFrame,
    strategy_feature_columns: Optional[Iterable[str]] = None,
    target_mode: str = "signal_quality",
    target_horizon: int = 3,
    target_threshold: float = 0.0015,
    use_signal_only_rows: bool = True,
    only_regular_session: bool = False,
) -> DatasetBundle:
    """
    Build an ML-ready frame.

    target_mode='signal_quality': target=1 means a BUY/SELL strategy signal worked.
    target_mode='up' or 'down': target is directional and uses all rows by default.
    """
    df = _normalize_signal_cols(bars)

    feature_frame = build_feature_frame(df)
    feature_frame = _dedupe_columns(feature_frame)

    # Do not allow non-feature/label columns from the original frame to collide
    # with the explicit label columns appended below.
    for col in ("base_signal", "target_signal_quality", "target_up", "target_down", "future_ret"):
        if col in feature_frame.columns:
            feature_frame = feature_frame.drop(columns=[col])

    for col in ("signal_is_buy", "signal_is_sell"):
        feature_frame[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()

    row_context_cols = [
        "buy_score", "sell_score", "nearU", "nearL", "bounceU", "bounceL",
        "bb_ok", "rsi_ok", "ema_up", "ema_dn",
        "macd_cross_up", "above_vwap", "st_up", "trend_ok", "vol_ok",
        "macd_zero_exit", "atr_exit", "atr_stop_mult",
        "position_is_long", "position_is_short", "bars_since_last_signal",
        "timeframe_minutes",
    ]
    for col in row_context_cols:
        if col in df.columns:
            feature_frame[col] = pd.to_numeric(_as_series(df, col), errors="coerce").to_numpy()

    strategy_feature_columns = list(strategy_feature_columns or [])
    for col in strategy_feature_columns:
        if col in df.columns and col not in feature_frame.columns:
            feature_frame[col] = _as_series(df, col).to_numpy()

    exclude = {
        "timestamp", "ts", "open", "high", "low", "close", "volume", "synthetic",
        "base_signal", "decision_reason", "future_ret",
        "target_up", "target_down", "target_signal_quality",
    }
    for col in _numeric_extra_columns(df, exclude=exclude):
        if col not in feature_frame.columns:
            feature_frame[col] = pd.to_numeric(_as_series(df, col), errors="coerce").to_numpy()

    label_input = pd.concat(
        [
            feature_frame.reset_index(drop=True),
            df[["base_signal"]].reset_index(drop=True),
        ],
        axis=1,
    )
    label_input = _dedupe_columns(label_input)

    if target_mode == "signal_quality":
        labeled = make_signal_quality_label(
            label_input,
            base_signal_col="base_signal",
            horizon=target_horizon,
            threshold=target_threshold,
        )
        target_column = "target_signal_quality"
    elif target_mode == "up":
        labeled = make_directional_labels(
            label_input,
            horizon=target_horizon,
            threshold=target_threshold,
        )
        target_column = "target_up"
    elif target_mode == "down":
        labeled = make_directional_labels(
            label_input,
            horizon=target_horizon,
            threshold=target_threshold,
        )
        target_column = "target_down"
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    if only_regular_session and "is_regular_session" in labeled.columns:
        labeled = labeled[labeled["is_regular_session"] == 1].copy()

    if use_signal_only_rows and "base_signal" in labeled.columns:
        signal_mask = labeled["base_signal"].astype(str).str.upper().isin(["BUY", "SELL"])
        labeled = labeled[signal_mask].copy()

    non_feature = {
        "base_signal", target_column, "future_ret",
        "target_up", "target_down", "target_signal_quality",
        "decision_reason", "timestamp", "ts",
    }

    default_cols = [c for c in DEFAULT_FEATURES if c in labeled.columns]
    extra_cols = [
        c for c in labeled.columns
        if c not in set(default_cols)
        and c not in non_feature
        and (pd.api.types.is_numeric_dtype(labeled[c]) or pd.api.types.is_bool_dtype(labeled[c]))
    ]

    feature_columns: list[str] = []
    seen: set[str] = set()
    for col in default_cols + extra_cols:
        if col not in seen:
            feature_columns.append(col)
            seen.add(col)

    labeled = labeled.replace([np.inf, -np.inf], np.nan)
    labeled = labeled.dropna(subset=[target_column]).copy()
    labeled[feature_columns] = labeled[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return DatasetBundle(frame=labeled, feature_columns=feature_columns, target_column=target_column)


def split_train_valid_test(
    frame: pd.DataFrame,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    train = frame.iloc[:train_end].copy()
    valid = frame.iloc[train_end:valid_end].copy()
    test = frame.iloc[valid_end:].copy()
    return train, valid, test
