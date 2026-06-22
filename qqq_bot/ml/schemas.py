from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MLConfig:
    enabled: bool = True
    mode: str = "advisory"  # advisory | soft | hard
    model_dir: str = "qqq_bot/ml/models"
    model_path: str = "qqq_bot/ml/models/qqq_signal_filter.joblib"  # legacy fallback
    metadata_path: str = "qqq_bot/ml/models/metadata.json"
    long_threshold: float = 0.62
    short_threshold: float = 0.62
    min_bars: int = 240
    log_path: str = "logs/ml_predictions.jsonl"
    only_regular_session: bool = True
    block_weak_signals: bool = True
    block_exits: bool = False
    min_edge: float = 0.05

    def validate(self) -> None:
        allowed = {"advisory", "soft", "hard"}
        if self.mode not in allowed:
            raise ValueError(f"Unsupported ML mode: {self.mode}. Expected one of {allowed}")
        for name in ("long_threshold", "short_threshold", "min_edge"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


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
    features_used: list[str] | None = None
    debug: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
