from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

# Reuse indicator functions from qqq_bot.signals to avoid duplicated math.
from qqq_bot.signals import Strategy2Config, add_indicators, normalize_bars

FEATURE_COLUMNS: List[str] = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "rsi",
    "macd_hist",
    "atr_pct",
    "volume_ratio",
    "close_vs_vwap_atr",
    "close_vs_ema_fast_atr",
    "close_vs_ema_slow_atr",
    "close_vs_ema200_atr",
    "bb_pos",
    "dist_recent_high_atr",
    "dist_recent_low_atr",
    "supertrend_up",
    "minute_of_day_sin",
    "minute_of_day_cos",
]


def build_feature_frame(bars: Any, cfg: Strategy2Config | None = None) -> pd.DataFrame:
    cfg = cfg or Strategy2Config.from_env()
    df = normalize_bars(bars)
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    ind = add_indicators(df, cfg)
    out = pd.DataFrame(index=ind.index)
    close = ind["close"]
    atr = ind["atr"].replace(0, np.nan)
    for n in (1, 3, 6, 12):
        out[f"ret_{n}"] = close.pct_change(n)
    out["rsi"] = ind["rsi"] / 100.0
    out["macd_hist"] = ind["macd_hist"] / atr
    out["atr_pct"] = ind["atr_pct"]
    out["volume_ratio"] = ind["volume_ratio"].clip(0, 10)
    out["close_vs_vwap_atr"] = (close - ind["vwap"]) / atr
    out["close_vs_ema_fast_atr"] = (close - ind["ema_fast"]) / atr
    out["close_vs_ema_slow_atr"] = (close - ind["ema_slow"]) / atr
    out["close_vs_ema200_atr"] = (close - ind["ema200"]) / atr
    out["bb_pos"] = (close - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"]).replace(0, np.nan)
    out["dist_recent_high_atr"] = (ind["recent_high"] - close) / atr
    out["dist_recent_low_atr"] = (close - ind["recent_low"]) / atr
    out["supertrend_up"] = ind["supertrend_up"].astype(float)

    try:
        ts = pd.to_datetime(ind["ts"], utc=True).dt.tz_convert("America/New_York")
        minute = ts.dt.hour * 60 + ts.dt.minute
        out["minute_of_day_sin"] = np.sin(2 * np.pi * minute / (24 * 60))
        out["minute_of_day_cos"] = np.cos(2 * np.pi * minute / (24 * 60))
    except Exception:
        out["minute_of_day_sin"] = 0.0
        out["minute_of_day_cos"] = 0.0

    out = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


def latest_features(bars: Any) -> Tuple[pd.DataFrame, dict]:
    x = build_feature_frame(bars)
    x = x.dropna()
    if x.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), {"reason": "not_enough_features"}
    return x.tail(1), {"reason": "ok", "rows": len(x)}
