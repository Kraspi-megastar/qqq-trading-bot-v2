from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .dataset import prepare_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/ml_training_frame.parquet")
    parser.add_argument("--target-mode", default="barrier")
    parser.add_argument("--target-horizon", type=int, default=12)
    parser.add_argument("--target-atr-mult", type=float, default=0.5)
    parser.add_argument("--all-sessions", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    bars = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    ds = prepare_training_dataset(
        bars,
        target_mode=args.target_mode,
        target_horizon=args.target_horizon,
        target_atr_mult=args.target_atr_mult,
        only_regular_session=not args.all_sessions,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        ds.frame.to_csv(out, index=True)
    else:
        ds.frame.to_parquet(out, index=True)
    print(f"Saved {len(ds.frame)} rows, {len(ds.feature_columns)} features to {out}")


if __name__ == "__main__":
    main()
