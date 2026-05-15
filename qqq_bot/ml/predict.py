from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from .features import latest_feature_row


class MLPredictor:
    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = joblib.load(model_path)
        self.metadata: dict[str, Any] = {}
        if metadata_path and Path(metadata_path).exists():
            self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.feature_columns = self.metadata.get("feature_columns")

    def predict_latest(
        self,
        bars: pd.DataFrame,
        strategy_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        row = latest_feature_row(
            bars,
            feature_columns=self.feature_columns,
            strategy_context=strategy_context,
        )
        prob = float(self.model.predict_proba(row)[0, 1])
        return {
            "probability": prob,
            "threshold": float(self.metadata.get("threshold", 0.5)),
            "feature_columns": list(row.columns),
            "feature_values": row.iloc[0].to_dict(),
        }
