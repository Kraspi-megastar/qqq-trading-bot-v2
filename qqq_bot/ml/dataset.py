from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .features import DEFAULT_FEATURES, MARKET_TZ, build_feature_frame
from .labels import make_barrier_labels, make_directional_labels, make_signal_quality_label


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
    target_mode: str
    metadata: dict = field(default_factory=dict)

    @property
    def target_column(self) -> str:
        return self.target_columns[0]


def _normalize_signal_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "base_signal" not in out.columns:
        if "action" in out.columns:
            out["base_signal"] = out["action"]
        elif "signal" in out.columns:
            out["base_signal"] = out["signal"]
        else:
            out["base_signal"] = "HOLD"
    out["base_signal"] = out["base_signal"].astype(str).str.upper().replace({"LONG": "BUY", "SHORT": "SELL"})
    out["signal_is_buy"] = out["base_signal"].eq("BUY").astype(int)
    out["signal_is_sell"] = out["base_signal"].eq("SELL").astype(int)
    return out


def prepare_training_dataset(
    bars: pd.DataFrame,
    strategy_feature_columns: Iterable[str] | None = None,
    target_mode: str = "barrier",
    target_horizon: int = 12,
    target_threshold: float = 0.0015,
    target_atr_mult: float = 0.5,
    use_signal_only_rows: bool | None = None,
    only_regular_session: bool = True,
    market_tz: str = MARKET_TZ,
) -> DatasetBundle:
    df = _normalize_signal_cols(bars)
    strategy_feature_columns = list(strategy_feature_columns or [])

    feature_frame = build_feature_frame(df, market_tz=market_tz)
    for col in strategy_feature_columns:
        if col in df.columns and col not in feature_frame.columns:
            feature_frame[col] = df[col]

    target_mode = target_mode.lower()
    labeled_input = feature_frame.copy()
    if "base_signal" not in labeled_input.columns:
        labeled_input["base_signal"] = df["base_signal"].to_numpy()

    if target_mode in {"barrier", "atr_barrier", "long_short"}:
        labeled = make_barrier_labels(labeled_input, horizon=target_horizon, atr_mult=target_atr_mult)
        target_columns = ["target_long_win", "target_short_win"]
        normalized_mode = "barrier"
    elif target_mode in {"signal_quality", "quality"}:
        labeled = make_signal_quality_label(
            labeled_input,
            base_signal_col="base_signal",
            horizon=target_horizon,
            atr_mult=target_atr_mult,
            threshold=target_threshold,
            label_engine="barrier",
        )
        target_columns = ["target_signal_quality"]
        normalized_mode = "signal_quality"
    elif target_mode in {"long_win", "long"}:
        labeled = make_barrier_labels(labeled_input, horizon=target_horizon, atr_mult=target_atr_mult)
        target_columns = ["target_long_win"]
        normalized_mode = "long_win"
    elif target_mode in {"short_win", "short"}:
        labeled = make_barrier_labels(labeled_input, horizon=target_horizon, atr_mult=target_atr_mult)
        target_columns = ["target_short_win"]
        normalized_mode = "short_win"
    elif target_mode == "up":
        labeled = make_directional_labels(labeled_input, horizon=target_horizon, threshold=target_threshold)
        target_columns = ["target_up"]
        normalized_mode = "up"
    elif target_mode == "down":
        labeled = make_directional_labels(labeled_input, horizon=target_horizon, threshold=target_threshold)
        target_columns = ["target_down"]
        normalized_mode = "down"
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    if only_regular_session and "is_rth" in labeled.columns:
        labeled = labeled[labeled["is_rth"] == 1].copy()

    if use_signal_only_rows is None:
        use_signal_only_rows = normalized_mode == "signal_quality"
    if use_signal_only_rows and "base_signal" in labeled.columns:
        labeled = labeled[labeled["base_signal"].isin(["BUY", "SELL"])].copy()

    feature_columns = [c for c in DEFAULT_FEATURES if c in labeled.columns]
    for col in feature_columns:
        labeled[col] = pd.to_numeric(labeled[col], errors="coerce")
    labeled = labeled.replace([np.inf, -np.inf], np.nan)
    labeled = labeled.dropna(subset=target_columns).copy()
    labeled[feature_columns] = labeled[feature_columns].fillna(0.0)

    metadata = {
        "target_horizon": int(target_horizon),
        "target_threshold": float(target_threshold),
        "target_atr_mult": float(target_atr_mult),
        "only_regular_session": bool(only_regular_session),
        "market_tz": market_tz,
    }
    return DatasetBundle(
        frame=labeled,
        feature_columns=feature_columns,
        target_columns=target_columns,
        target_mode=normalized_mode,
        metadata=metadata,
    )


def split_train_valid_test(
    frame: pd.DataFrame,
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
    purge_gap: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0 <= valid_frac < 1:
        raise ValueError("valid_frac must be in [0, 1)")
    if train_frac + valid_frac >= 1:
        raise ValueError("train_frac + valid_frac must be < 1")

    ordered = frame.sort_index().copy()
    n = len(ordered)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    gap = max(0, int(purge_gap))

    train = ordered.iloc[: max(0, train_end - gap)].copy()
    valid = ordered.iloc[min(n, train_end + gap) : max(train_end + gap, valid_end - gap)].copy()
    test = ordered.iloc[min(n, valid_end + gap) :].copy()
    return train, valid, test
