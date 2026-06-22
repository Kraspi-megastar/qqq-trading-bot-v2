from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .features import MARKET_TZ, latest_feature_row


class MLPredictor:
    def __init__(self, model_dir: str | None = None, metadata_path: str | None = None, model_path: str | None = None):
        import joblib

        if model_dir is None and metadata_path is None and model_path is not None:
            metadata_path = str(Path(model_path).with_name("metadata.json"))
            model_dir = str(Path(model_path).parent)
        self.model_dir = Path(model_dir or Path(metadata_path or "").parent or "qqq_bot/ml/models")
        self.metadata_path = Path(metadata_path or self.model_dir / "metadata.json")
        self.metadata: dict[str, Any] = {}
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        self.feature_columns: list[str] | None = self.metadata.get("feature_columns")
        self.market_tz = str(self.metadata.get("market_tz", MARKET_TZ))
        self.models: dict[str, Any] = {}
        self.thresholds: dict[str, float] = {}

        targets = self.metadata.get("targets") or {}
        if targets:
            for key, meta in targets.items():
                model_file = meta.get("model_file")
                if not model_file:
                    continue
                path = self.model_dir / model_file
                if path.exists():
                    self.models[key] = joblib.load(path)
                    self.thresholds[key] = float(meta.get("threshold", 0.5))
        elif model_path and Path(model_path).exists():
            self.models["signal"] = joblib.load(model_path)
            self.thresholds["signal"] = float(self.metadata.get("threshold", 0.5))

        if not self.models:
            raise FileNotFoundError(f"No ML model files found in {self.model_dir}")

    def predict_latest(
        self,
        bars: pd.DataFrame,
        strategy_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = latest_feature_row(
            bars,
            feature_columns=self.feature_columns,
            strategy_context=strategy_context,
            market_tz=self.market_tz,
        )
        out: dict[str, Any] = {
            "feature_columns": list(row.columns),
            "feature_values": row.iloc[0].to_dict(),
            "target_mode": self.metadata.get("target_mode"),
            "thresholds": dict(self.thresholds),
        }
        for key, model in self.models.items():
            out[f"{key}_prob"] = float(model.predict_proba(row)[0, 1])
        if "long" in self.models and "short" in self.models:
            out["long_prob"] = out["long_prob"]
            out["short_prob"] = out["short_prob"]
        elif "signal" in self.models:
            prob = float(out["signal_prob"])
            signal = str((strategy_context or {}).get("base_signal", "")).upper()
            if signal == "SELL":
                out["long_prob"] = 1.0 - prob
                out["short_prob"] = prob
            else:
                out["long_prob"] = prob
                out["short_prob"] = 1.0 - prob
        else:
            out["long_prob"] = float(out.get("long_prob", 0.5))
            out["short_prob"] = float(out.get("short_prob", 0.5))
        return out
