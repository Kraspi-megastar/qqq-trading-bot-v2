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


def _normalize_signal_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "base_signal" not in out.columns:
        out["base_signal"] = "HOLD"
    out["base_signal"] = out["base_signal"].astype(str).str.upper()
    out["signal_is_buy"] = out["base_signal"].eq("BUY").astype(int)
    out["signal_is_sell"] = out["base_signal"].eq("SELL").astype(int)
    return out


def prepare_training_dataset(
    bars: pd.DataFrame,
    strategy_feature_columns: Optional[Iterable[str]] = None,
    target_mode: str = "signal_quality",
    target_horizon: int = 3,
    target_threshold: float = 0.0015,
    use_signal_only_rows: bool = True,
    only_regular_session: bool = False,
) -> DatasetBundle:
    df = bars.copy()
    df = _normalize_signal_cols(df)
    feature_frame = build_feature_frame(df)

    strategy_feature_columns = list(strategy_feature_columns or [])
    for col in strategy_feature_columns:
        if col in df.columns and col not in feature_frame.columns:
            feature_frame[col] = df[col]

    if target_mode == "signal_quality":
        labeled = make_signal_quality_label(
            pd.concat([feature_frame, df[["base_signal"]]], axis=1),
            base_signal_col="base_signal",
            horizon=target_horizon,
            threshold=target_threshold,
        )
        target_column = "target_signal_quality"
    elif target_mode == "up":
        labeled = make_directional_labels(
            feature_frame,
            horizon=target_horizon,
            threshold=target_threshold,
        )
        target_column = "target_up"
    elif target_mode == "down":
        labeled = make_directional_labels(
            feature_frame,
            horizon=target_horizon,
            threshold=target_threshold,
        )
        target_column = "target_down"
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    if only_regular_session and "is_regular_session" in labeled.columns:
        labeled = labeled[labeled["is_regular_session"] == 1].copy()

    if use_signal_only_rows and "base_signal" in df.columns:
        labeled = labeled[df["base_signal"].isin(["BUY", "SELL"])].copy()

    feature_columns = [c for c in DEFAULT_FEATURES if c in labeled.columns]
    labeled = labeled.replace([np.inf, -np.inf], np.nan)
    labeled = labeled.dropna(subset=[target_column]).copy()
    labeled[feature_columns] = labeled[feature_columns].fillna(0.0)

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
