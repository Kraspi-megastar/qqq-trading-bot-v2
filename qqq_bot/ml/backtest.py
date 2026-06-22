from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import prepare_training_dataset
from .predict import MLPredictor


def run_backtest(
    input_path: str,
    model_dir: str,
    target_horizon: int = 12,
    target_atr_mult: float = 0.5,
    only_regular_session: bool = True,
) -> dict:
    path = Path(input_path)
    bars = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    ds = prepare_training_dataset(
        bars,
        target_mode="barrier",
        target_horizon=target_horizon,
        target_atr_mult=target_atr_mult,
        only_regular_session=only_regular_session,
    )
    predictor = MLPredictor(model_dir=model_dir, metadata_path=str(Path(model_dir) / "metadata.json"))
    df = ds.frame.copy()
    X = df[ds.feature_columns]

    if "long" in predictor.models:
        df["long_prob"] = predictor.models["long"].predict_proba(X)[:, 1]
    else:
        df["long_prob"] = 0.5
    if "short" in predictor.models:
        df["short_prob"] = predictor.models["short"].predict_proba(X)[:, 1]
    else:
        df["short_prob"] = 0.5

    long_thr = predictor.thresholds.get("long", 0.62)
    short_thr = predictor.thresholds.get("short", 0.62)
    df["take_long"] = (df["long_prob"] >= long_thr) & (df["long_prob"] > df["short_prob"])
    df["take_short"] = (df["short_prob"] >= short_thr) & (df["short_prob"] > df["long_prob"])

    long_win_rate = float(df.loc[df["take_long"], "target_long_win"].mean()) if df["take_long"].any() else None
    short_win_rate = float(df.loc[df["take_short"], "target_short_win"].mean()) if df["take_short"].any() else None
    summary = {
        "rows": int(len(df)),
        "long_trades": int(df["take_long"].sum()),
        "short_trades": int(df["take_short"].sum()),
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "avg_long_prob": float(np.mean(df["long_prob"])),
        "avg_short_prob": float(np.mean(df["short_prob"])),
        "long_threshold": float(long_thr),
        "short_threshold": float(short_thr),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", default="qqq_bot/ml/models")
    parser.add_argument("--target-horizon", type=int, default=12)
    parser.add_argument("--target-atr-mult", type=float, default=0.5)
    parser.add_argument("--all-sessions", action="store_true")
    args = parser.parse_args()
    run_backtest(
        input_path=args.input,
        model_dir=args.model_dir,
        target_horizon=args.target_horizon,
        target_atr_mult=args.target_atr_mult,
        only_regular_session=not args.all_sessions,
    )


if __name__ == "__main__":
    main()
