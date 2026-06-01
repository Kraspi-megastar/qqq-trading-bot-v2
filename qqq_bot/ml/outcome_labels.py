from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from qqq_bot.signals import Strategy2Config, add_indicators, normalize_bars


def add_outcome_labels(
    bars: Any,
    *,
    horizons: Iterable[int] = (6, 12),
    atr_targets: Iterable[float] = (0.5, 1.0),
    cfg: Strategy2Config | None = None,
) -> pd.DataFrame:
    """Create outcome labels: will price move +/- target ATR within next N bars.

    For every bar t and horizon h:
      long_{target}atr_h{h}=1 if max(high[t+1:t+h]) >= close[t] + target*ATR[t]
      short_{target}atr_h{h}=1 if min(low[t+1:t+h]) <= close[t] - target*ATR[t]
    """
    cfg = cfg or Strategy2Config.from_env()
    df = normalize_bars(bars)
    ind = add_indicators(df, cfg)
    labels = pd.DataFrame(index=ind.index)
    close = ind["close"]
    atr = ind["atr"]
    for h in horizons:
        future_high = pd.concat([ind["high"].shift(-i) for i in range(1, h + 1)], axis=1).max(axis=1)
        future_low = pd.concat([ind["low"].shift(-i) for i in range(1, h + 1)], axis=1).min(axis=1)
        for target in atr_targets:
            key = str(target).replace(".", "")
            labels[f"long_{key}atr_h{h}"] = (future_high >= close + target * atr).astype(float)
            labels[f"short_{key}atr_h{h}"] = (future_low <= close - target * atr).astype(float)
    labels["ts"] = ind["ts"].values
    labels["close"] = ind["close"].values
    labels["atr"] = ind["atr"].values
    return labels
