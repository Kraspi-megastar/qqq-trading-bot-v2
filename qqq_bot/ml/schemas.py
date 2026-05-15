from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MLConfig:
    enabled: bool = True
    mode: str = "advisory"  # advisory | soft | hard
    model_path: str = "qqq_bot/ml/models/qqq_signal_filter.joblib"
    metadata_path: str = "qqq_bot/ml/models/metadata.json"
    long_threshold: float = 0.62
    short_threshold: float = 0.62
    min_bars: int = 80
    log_path: str = "logs/ml_predictions.jsonl"
    only_regular_session: bool = False
    block_weak_signals: bool = False

    def validate(self) -> None:
        allowed = {"advisory", "soft", "hard"}
        if self.mode not in allowed:
            raise ValueError(f"Unsupported ML mode: {self.mode}. Expected one of {allowed}")
        if not (0.0 <= self.long_threshold <= 1.0):
            raise ValueError("long_threshold must be within [0, 1]")
        if not (0.0 <= self.short_threshold <= 1.0):
            raise ValueError("short_threshold must be within [0, 1]")
        if self.min_bars < 1:
            raise ValueError("min_bars must be positive")


@dataclass
class MLDecision:
    enabled: bool
    mode: str
    base_signal: str
    final_signal: str
    allow_long: bool
    allow_short: bool
    long_prob: float
    short_prob: float
    reason: str
    features_used: Optional[List[str]] = None
    debug: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
