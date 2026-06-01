from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_bot.ml.features_outcome import build_feature_frame
from qqq_bot.ml.outcome_labels import add_outcome_labels
from qqq_bot.signals import Strategy2Config, normalize_bars


def build_outcome_dataset(bars: Any, *, horizon: int = 12) -> pd.DataFrame:
    cfg = Strategy2Config.from_env()
    df = normalize_bars(bars)
    x = build_feature_frame(df, cfg)
    y = add_outcome_labels(df, horizons=(horizon,), atr_targets=(0.5, 1.0), cfg=cfg)
    out = pd.concat([df[["ts", "open", "high", "low", "close", "volume"]], x, y.drop(columns=["ts", "close"], errors="ignore")], axis=1)
    return out.dropna().reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV/JSONL file with OHLCV bars")
    p.add_argument("--output", default="data/ml_outcome_dataset.csv")
    p.add_argument("--horizon", type=int, default=12)
    args = p.parse_args()
    path = Path(args.input)
    if path.suffix.lower() == ".jsonl":
        bars = pd.read_json(path, lines=True)
    else:
        bars = pd.read_csv(path)
    ds = build_outcome_dataset(bars, horizon=args.horizon)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(out, index=False)
    print(f"saved {len(ds)} rows to {out}")


if __name__ == "__main__":
    main()
