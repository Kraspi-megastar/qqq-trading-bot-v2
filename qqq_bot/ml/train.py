from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import prepare_training_dataset, split_train_valid_test


def _build_model(model_name: str):
    name = model_name.lower()
    if name == "auto":
        for candidate in ("lightgbm", "xgboost", "gbdt", "logreg"):
            try:
                return _build_model(candidate)
            except RuntimeError:
                continue
        return _build_model("logreg")

    if name == "logreg":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )

    if name in {"gbdt", "histgb", "sklearn_gbdt"}:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=180,
                        learning_rate=0.035,
                        max_depth=3,
                        min_samples_leaf=20,
                        random_state=42,
                    ),
                ),
            ]
        )

    if name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_leaf=25,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("lightgbm is not installed") from exc
        return LGBMClassifier(
            n_estimators=450,
            learning_rate=0.025,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("xgboost is not installed") from exc
        return XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("catboost is not installed") from exc
        return CatBoostClassifier(
            iterations=450,
            depth=5,
            learning_rate=0.025,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _load_bars(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _safe_metric(fn, y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    try:
        if len(set(y_true.astype(int))) < 2:
            return None
        return float(fn(y_true, y_prob))
    except Exception:
        return None


def _select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.56,
) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, {"best_f1": 0.0, "precision": 0.0, "recall": 0.0}

    p = precision[:-1]
    r = recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    ok = np.where(p >= min_precision)[0]
    if len(ok) > 0:
        idx = int(ok[np.argmax(r[ok])])
    else:
        idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), {
        "best_f1": float(f1[idx]),
        "precision": float(p[idx]),
        "recall": float(r[idx]),
    }


def _fit_one(
    *,
    target_name: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_precision: float,
    model_path: Path,
) -> dict[str, Any]:
    if len(set(y_train.astype(int))) < 2:
        raise RuntimeError(f"Target {target_name} has only one class in train split.")

    model = _build_model(model_name)
    model.fit(X_train, y_train.astype(int))

    valid_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    threshold, threshold_stats = _select_threshold(
        y_valid.to_numpy(dtype=int),
        valid_prob,
        min_precision=min_precision,
    )

    joblib.dump(model, model_path)

    valid_pred = (valid_prob >= threshold).astype(int)
    test_pred = (test_prob >= threshold).astype(int)

    print(f"\n=== {target_name} | VALID @ threshold {threshold:.4f} ===")
    print(classification_report(y_valid, valid_pred, digits=4, zero_division=0))
    print(f"=== {target_name} | TEST @ threshold {threshold:.4f} ===")
    print(classification_report(y_test, test_pred, digits=4, zero_division=0))

    return {
        "target": target_name,
        "model_file": model_path.name,
        "threshold": threshold,
        "threshold_selection": threshold_stats,
        "valid_roc_auc": _safe_metric(roc_auc_score, y_valid, valid_prob),
        "valid_pr_auc": _safe_metric(average_precision_score, y_valid, valid_prob),
        "valid_brier": _safe_metric(brier_score_loss, y_valid, valid_prob),
        "test_roc_auc": _safe_metric(roc_auc_score, y_test, test_prob),
        "test_pr_auc": _safe_metric(average_precision_score, y_test, test_prob),
        "test_brier": _safe_metric(brier_score_loss, y_test, test_prob),
        "valid_positive_rate": float(np.mean(y_valid)),
        "test_positive_rate": float(np.mean(y_test)),
    }


def train_model(
    input_path: str,
    model_dir: str,
    model_name: str = "auto",
    target_mode: str = "barrier",
    target_horizon: int = 12,
    target_threshold: float = 0.0015,
    target_atr_mult: float = 0.5,
    only_regular_session: bool = True,
    min_precision: float = 0.56,
) -> dict[str, Any]:
    bars = _load_bars(input_path)
    ds = prepare_training_dataset(
        bars=bars,
        target_mode=target_mode,
        target_horizon=target_horizon,
        target_threshold=target_threshold,
        target_atr_mult=target_atr_mult,
        only_regular_session=only_regular_session,
    )

    train, valid, test = split_train_valid_test(ds.frame, purge_gap=target_horizon)
    if min(len(train), len(valid), len(test)) == 0:
        raise RuntimeError(
            f"Not enough rows after chronological split: train={len(train)}, valid={len(valid)}, test={len(test)}"
        )

    X_train = train[ds.feature_columns]
    X_valid = valid[ds.feature_columns]
    X_test = test[ds.feature_columns]

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_meta: dict[str, Any] = {}
    for target_col in ds.target_columns:
        if target_col == "target_long_win":
            key = "long"
        elif target_col == "target_short_win":
            key = "short"
        else:
            key = "signal"
        targets_meta[key] = _fit_one(
            target_name=target_col,
            model_name=model_name,
            X_train=X_train,
            y_train=train[target_col].astype(int),
            X_valid=X_valid,
            y_valid=valid[target_col].astype(int),
            X_test=X_test,
            y_test=test[target_col].astype(int),
            min_precision=min_precision,
            model_path=out_dir / f"{key}_model.joblib",
        )

    metadata = {
        "version": 2,
        "model_name": model_name,
        "model_dir": str(out_dir),
        "feature_columns": ds.feature_columns,
        "target_mode": ds.target_mode,
        "target_columns": ds.target_columns,
        "targets": targets_meta,
        **ds.metadata,
        "split": {
            "method": "chronological_train_valid_test_with_purge_gap",
            "purge_gap": int(target_horizon),
            "n_train": int(len(train)),
            "n_valid": int(len(valid)),
            "n_test": int(len(test)),
            "train_start": str(train.index.min()),
            "train_end": str(train.index.max()),
            "valid_start": str(valid.index.min()),
            "valid_end": str(valid.index.max()),
            "test_start": str(test.index.min()),
            "test_end": str(test.index.max()),
        },
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== METADATA ===")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV/parquet/jsonl with bars and optional base_signal/action column.")
    parser.add_argument("--model-dir", default="qqq_bot/ml/models")
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "lightgbm", "xgboost", "catboost", "gbdt", "histgb", "random_forest", "logreg"],
    )
    parser.add_argument(
        "--target-mode",
        default="barrier",
        choices=["barrier", "signal_quality", "long_win", "short_win", "up", "down"],
    )
    parser.add_argument("--target-horizon", type=int, default=12)
    parser.add_argument("--target-threshold", type=float, default=0.0015)
    parser.add_argument("--target-atr-mult", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.56)
    parser.add_argument("--all-sessions", action="store_true")
    args = parser.parse_args()

    train_model(
        input_path=args.input,
        model_dir=args.model_dir,
        model_name=args.model,
        target_mode=args.target_mode,
        target_horizon=args.target_horizon,
        target_threshold=args.target_threshold,
        target_atr_mult=args.target_atr_mult,
        only_regular_session=not args.all_sessions,
        min_precision=args.min_precision,
    )


if __name__ == "__main__":
    main()
