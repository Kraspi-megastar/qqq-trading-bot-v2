from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import prepare_training_dataset
from .train import train_model
from .predict import MLPredictor


def run_backtest(
    input_path: str,
    model_dir: str,
    target_horizon: int = 3,
    target_threshold: float = 0.0015,
) -> dict:
    bars = pd.read_parquet(input_path) if input_path.endswith(".parquet") else pd.read_csv(input_path)

    if not Path(model_dir, "qqq_signal_filter.joblib").exists():
        train_model(
            input_path=input_path,
            model_dir=model_dir,
            model_name="xgboost",
            target_mode="signal_quality",
            target_horizon=target_horizon,
            target_threshold=target_threshold,
        )

    predictor = MLPredictor(
        model_path=str(Path(model_dir, "qqq_signal_filter.joblib")),
        metadata_path=str(Path(model_dir, "metadata.json")),
    )
    threshold = float(predictor.metadata.get("threshold", 0.5))

    ds = prepare_training_dataset(
        bars=bars,
        target_mode="signal_quality",
        target_horizon=target_horizon,
        target_threshold=target_threshold,
        use_signal_only_rows=True,
        only_regular_session=True,
    )
    df = ds.frame.copy()
    probs = predictor.model.predict_proba(df[ds.feature_columns])[:, 1]
    df["prob"] = probs
    df["take_trade"] = (df["prob"] >= threshold).astype(int)
    df["trade_return"] = np.where(df["target_signal_quality"] == 1, target_threshold, -target_threshold)
    df["filtered_return"] = df["trade_return"] * df["take_trade"]

    summary = {
        "rows": int(len(df)),
        "threshold": threshold,
        "win_rate_all": float((df["trade_return"] > 0).mean()),
        "win_rate_filtered": float((df.loc[df["take_trade"] == 1, "trade_return"] > 0).mean()) if (df["take_trade"] == 1).any() else None,
        "sum_return_all": float(df["trade_return"].sum()),
        "sum_return_filtered": float(df["filtered_return"].sum()),
        "trades_all": int(len(df)),
        "trades_filtered": int(df["take_trade"].sum()),
        "avg_return_filtered": float(df.loc[df["take_trade"] == 1, "trade_return"].mean()) if (df["take_trade"] == 1).any() else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", default="qqq_bot/ml/models")
    parser.add_argument("--target-horizon", type=int, default=3)
    parser.add_argument("--target-threshold", type=float, default=0.0015)
    args = parser.parse_args()
    run_backtest(
        input_path=args.input,
        model_dir=args.model_dir,
        target_horizon=args.target_horizon,
        target_threshold=args.target_threshold,
    )


if __name__ == "__main__":
    main()
