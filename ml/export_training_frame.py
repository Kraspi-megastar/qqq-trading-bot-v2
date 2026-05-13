from __future__ import annotations

import argparse
import pandas as pd

from .dataset import prepare_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Bars parquet/csv")
    parser.add_argument("--output", required=True, help="Output parquet/csv")
    parser.add_argument("--target-mode", default="signal_quality", choices=["signal_quality", "up", "down"])
    args = parser.parse_args()

    bars = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    ds = prepare_training_dataset(
        bars=bars,
        target_mode=args.target_mode,
        use_signal_only_rows=(args.target_mode == "signal_quality"),
        only_regular_session=True,
    )
    if args.output.endswith(".parquet"):
        ds.frame.to_parquet(args.output, index=True)
    else:
        ds.frame.to_csv(args.output, index=True)
    print(f"Saved {len(ds.frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
