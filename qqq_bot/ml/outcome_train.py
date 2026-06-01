from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from qqq_bot.ml.features_outcome import FEATURE_COLUMNS


def train(dataset_csv: str, *, model_dir: str = "qqq_bot/ml/models", horizon: int = 12) -> None:
    df = pd.read_csv(dataset_csv)
    target_cols = {
        "long_05atr": f"long_05atr_h{horizon}",
        "long_10atr": f"long_10atr_h{horizon}",
        "short_05atr": f"short_05atr_h{horizon}",
        "short_10atr": f"short_10atr_h{horizon}",
    }
    missing = [c for c in FEATURE_COLUMNS + list(target_cols.values()) if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in dataset: {missing}")
    data = df.dropna(subset=FEATURE_COLUMNS + list(target_cols.values())).reset_index(drop=True)
    if len(data) < 500:
        print(f"WARNING: only {len(data)} rows. Model will be weak; collect more bars.")
    x = data[FEATURE_COLUMNS]
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for name, col in target_cols.items():
        y = data[col].astype(int)
        split = int(len(data) * 0.8)
        x_train, x_test = x.iloc[:split], x.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=30,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)[:, 1] if len(x_test) else []
        auc = None
        if len(set(y_test)) > 1 and len(proba):
            auc = float(roc_auc_score(y_test, proba))
        metrics[name] = {
            "target_col": col,
            "positive_rate_train": float(y_train.mean()) if len(y_train) else None,
            "positive_rate_test": float(y_test.mean()) if len(y_test) else None,
            "roc_auc_test": auc,
            "rows": int(len(data)),
        }
        joblib.dump(clf, out_dir / f"outcome_{name}.joblib")
        print(name, metrics[name])

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "horizon_bars": horizon,
        "feature_columns": FEATURE_COLUMNS,
        "targets": target_cols,
        "metrics": metrics,
        "model_type": "RandomForestClassifier",
    }
    (out_dir / "outcome_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"models saved to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-dir", default="qqq_bot/ml/models")
    p.add_argument("--horizon", type=int, default=12)
    args = p.parse_args()
    train(args.dataset, model_dir=args.model_dir, horizon=args.horizon)


if __name__ == "__main__":
    main()
