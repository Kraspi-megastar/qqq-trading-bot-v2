from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from qqq_bot.ml.features_outcome import FEATURE_COLUMNS, latest_features
from qqq_bot.signal_types import MLApproval


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


class MLOutcomeService:
    """Outcome-based ML filter.

    This replaces BUY/SELL classification with outcome probabilities:
      - probability of +0.5 ATR / +1.0 ATR within the next N bars
      - probability of -0.5 ATR / -1.0 ATR within the next N bars
    """

    def __init__(self, model_dir: str | None = None, enabled: Optional[bool] = None):
        self.model_dir = Path(model_dir or os.getenv("ML_OUTCOME_MODEL_DIR", "qqq_bot/ml/models"))
        self.enabled = bool(enabled if enabled is not None else os.getenv("ML_ENABLED", "true").lower() in {"1", "true", "yes", "on"})
        self.long_05_min = _env_float("ML_LONG_05ATR_MIN", 0.55)
        self.long_10_min = _env_float("ML_LONG_10ATR_MIN", 0.40)
        self.short_05_min = _env_float("ML_SHORT_05ATR_MIN", 0.55)
        self.short_10_min = _env_float("ML_SHORT_10ATR_MIN", 0.40)
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.load()

    @property
    def model_loaded(self) -> bool:
        return all(k in self.models for k in ("long_05atr", "long_10atr", "short_05atr", "short_10atr"))

    def load(self) -> None:
        self.models = {}
        for name in ("long_05atr", "long_10atr", "short_05atr", "short_10atr"):
            path = self.model_dir / f"outcome_{name}.joblib"
            if path.exists():
                self.models[name] = joblib.load(path)
        meta = self.model_dir / "outcome_metadata.json"
        if meta.exists():
            try:
                self.metadata = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                self.metadata = {}

    def predict(self, bars: Any) -> MLApproval:
        if not self.enabled:
            return MLApproval(enabled=False, model_loaded=self.model_loaded, reason="ml_disabled")
        if not self.model_loaded:
            return MLApproval(enabled=True, model_loaded=False, reason="outcome_models_not_loaded")
        x, info = latest_features(bars)
        if x.empty:
            return MLApproval(enabled=True, model_loaded=True, reason=info.get("reason", "features_empty"))
        x = x[FEATURE_COLUMNS]
        probs: Dict[str, float] = {}
        for name, model in self.models.items():
            try:
                probs[name] = float(model.predict_proba(x)[0, 1])
            except Exception:
                probs[name] = 0.0
        long_ok = probs["long_05atr"] >= self.long_05_min and probs["long_10atr"] >= self.long_10_min
        short_ok = probs["short_05atr"] >= self.short_05_min and probs["short_10atr"] >= self.short_10_min
        return MLApproval(
            enabled=True,
            model_loaded=True,
            mode="outcome",
            long_05atr=probs["long_05atr"],
            long_10atr=probs["long_10atr"],
            short_05atr=probs["short_05atr"],
            short_10atr=probs["short_10atr"],
            long_ok=long_ok,
            short_ok=short_ok,
            reason="outcome_model_ok",
        )


# Compatibility name close to the existing project style.
MLTradingService = MLOutcomeService
