from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .dataset import prepare_training_dataset, split_train_valid_test


def _build_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "logreg":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1500, class_weight="balanced")),
            ]
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("xgboost is not installed. Use model=logreg or install xgboost.") from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    idx = int(np.nanargmax(f1_scores))
    return float(thresholds[idx]), float(f1_scores[idx])


def train_model(
    input_path: str,
    model_dir: str,
    model_name: str = "xgboost",
    target_mode: str = "signal_quality",
    target_horizon: int = 3,
    target_threshold: float = 0.0015,
    only_regular_session: bool = True,
) -> dict[str, Any]:
    bars = pd.read_parquet(input_path) if input_path.endswith(".parquet") else pd.read_csv(input_path)
    ds = prepare_training_dataset(
        bars=bars,
        target_mode=target_mode,
        target_horizon=target_horizon,
        target_threshold=target_threshold,
        use_signal_only_rows=(target_mode == "signal_quality"),
        only_regular_session=only_regular_session,
    )

    train, valid, test = split_train_valid_test(ds.frame)

    X_train = train[ds.feature_columns]
    y_train = train[ds.target_column].astype(int)

    X_valid = valid[ds.feature_columns]
    y_valid = valid[ds.target_column].astype(int)

    X_test = test[ds.feature_columns]
    y_test = test[ds.target_column].astype(int)

    model = _build_model(model_name)
    model.fit(X_train, y_train)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    threshold, best_f1 = _best_threshold(y_valid.to_numpy(), valid_prob)

    model_path = Path(model_dir) / "qqq_signal_filter.joblib"
    metadata_path = Path(model_dir) / "metadata.json"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "model_path": str(model_path),
        "feature_columns": ds.feature_columns,
        "target_column": ds.target_column,
        "target_mode": target_mode,
        "target_horizon": target_horizon,
        "target_threshold": target_threshold,
        "threshold": threshold,
        "valid_roc_auc": float(roc_auc_score(y_valid, valid_prob)) if len(set(y_valid)) > 1 else None,
        "valid_pr_auc": float(average_precision_score(y_valid, valid_prob)) if len(set(y_valid)) > 1 else None,
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)) if len(set(y_test)) > 1 else None,
        "test_pr_auc": float(average_precision_score(y_test, test_prob)) if len(set(y_test)) > 1 else None,
        "best_valid_f1": best_f1,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== VALID report @ threshold {:.4f} ===".format(threshold))
    print(classification_report(y_valid, (valid_prob >= threshold).astype(int), digits=4))
    print("=== TEST report @ threshold {:.4f} ===".format(threshold))
    print(classification_report(y_test, (test_prob >= threshold).astype(int), digits=4))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to parquet/csv with bars and optional base_signal column.")
    parser.add_argument("--model-dir", default="qqq_bot/ml/models")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "logreg"])
    parser.add_argument("--target-mode", default="signal_quality", choices=["signal_quality", "up", "down"])
    parser.add_argument("--target-horizon", type=int, default=3)
    parser.add_argument("--target-threshold", type=float, default=0.0015)
    parser.add_argument("--all-sessions", action="store_true")
    args = parser.parse_args()

    train_model(
        input_path=args.input,
        model_dir=args.model_dir,
        model_name=args.model,
        target_mode=args.target_mode,
        target_horizon=args.target_horizon,
        target_threshold=args.target_threshold,
        only_regular_session=not args.all_sessions,
    )


if __name__ == "__main__":
    main()
