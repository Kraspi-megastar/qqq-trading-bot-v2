from __future__ import annotations

from .service import MLTradingService


def apply_ml_gate(bars, base_decision, strategy_context: dict, current_position: str | None = None):
    """Minimal example. Production integration is already implemented in scheduler.py."""
    ml = MLTradingService.from_env()
    result = ml.decide(
        bars=bars,
        base_signal=base_decision.action,
        strategy_context=strategy_context,
        current_position=current_position,
    )
    return result
