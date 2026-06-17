from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .predict import MLPredictor
from .schemas import MLConfig, MLDecision
from .telemetry import append_jsonl


class MLTradingService:
    def __init__(self, config: MLConfig):
        self.config = config
        self.config.validate()
        self.predictor: Optional[MLPredictor] = None

        if self.config.enabled and Path(self.config.model_path).exists():
            self.predictor = MLPredictor(
                model_path=self.config.model_path,
                metadata_path=self.config.metadata_path if Path(self.config.metadata_path).exists() else None,
            )

    @classmethod
    def from_env(cls) -> "MLTradingService":
        cfg = MLConfig(
            enabled=os.getenv("ML_ENABLED", "1") == "1",
            mode=os.getenv("ML_MODE", "advisory"),
            model_path=os.getenv("ML_MODEL_PATH", "qqq_bot/ml/models/qqq_signal_filter.joblib"),
            metadata_path=os.getenv("ML_METADATA_PATH", "qqq_bot/ml/models/metadata.json"),
            long_threshold=float(os.getenv("ML_LONG_THRESHOLD", "0.62")),
            short_threshold=float(os.getenv("ML_SHORT_THRESHOLD", "0.62")),
            min_bars=int(os.getenv("ML_MIN_BARS", "80")),
            log_path=os.getenv("ML_LOG_PATH", "logs/ml_predictions.jsonl"),
            only_regular_session=os.getenv("ML_ONLY_REGULAR_SESSION", "1") == "1",
            block_weak_signals=os.getenv("ML_BLOCK_WEAK_SIGNALS", "1") == "1",
        )
        return cls(cfg)

    def _empty_decision(self, base_signal: str, reason: str) -> MLDecision:
        return MLDecision(
            enabled=self.config.enabled,
            mode=self.config.mode,
            base_signal=base_signal,
            final_signal=base_signal,
            allow_long=base_signal == "BUY",
            allow_short=base_signal == "SELL",
            long_prob=0.0,
            short_prob=0.0,
            reason=reason,
            features_used=None,
            debug=None,
        )

    def decide(
        self,
        bars: pd.DataFrame,
        base_signal: str,
        strategy_context: Optional[dict[str, Any]] = None,
        current_position: Optional[str] = None,
    ) -> MLDecision:
        base_signal = str(base_signal).upper()

        if not self.config.enabled:
            return self._empty_decision(base_signal, "ml_disabled")

        if self.predictor is None:
            return self._empty_decision(base_signal, "model_not_loaded")

        if bars is None or len(bars) < self.config.min_bars:
            return self._empty_decision(base_signal, "insufficient_bars")

        strategy_context = dict(strategy_context or {})
        session = str(strategy_context.get("session", "")).lower()

        if self.config.only_regular_session and session not in {"regular", "rth"}:
            return self._empty_decision(base_signal, "session_filtered")

        if base_signal == "HOLD":
            pred = self.predictor.predict_latest(bars, strategy_context=strategy_context)
            decision = MLDecision(
                enabled=True,
                mode=self.config.mode,
                base_signal=base_signal,
                final_signal="HOLD",
                allow_long=False,
                allow_short=False,
                long_prob=float(pred["probability"]),
                short_prob=float(1.0 - pred["probability"]),
                reason="hold_base_signal",
                features_used=pred["feature_columns"],
                debug={"threshold": pred["threshold"]},
            )
            self._log(strategy_context, decision)
            return decision

        if base_signal == "BUY":
            strategy_context["signal_is_buy"] = 1
            strategy_context["signal_is_sell"] = 0
        elif base_signal == "SELL":
            strategy_context["signal_is_buy"] = 0
            strategy_context["signal_is_sell"] = 1

        if str(current_position).upper() == "LONG":
            strategy_context["position_is_long"] = 1
        elif str(current_position).upper() == "SHORT":
            strategy_context["position_is_short"] = 1

        pred = self.predictor.predict_latest(bars, strategy_context=strategy_context)
        long_prob = float(pred["probability"])
        short_prob = float(1.0 - pred["probability"])

        allow_long = long_prob >= self.config.long_threshold
        allow_short = short_prob >= self.config.short_threshold

        final_signal = base_signal
        reason = "advisory_no_override"

        if self.config.mode in {"soft", "hard"} and self.config.block_weak_signals:
            if base_signal == "BUY" and not allow_long:
                final_signal = "HOLD"
                reason = "buy_blocked_by_ml"
            elif base_signal == "SELL" and not allow_short:
                final_signal = "HOLD"
                reason = "sell_blocked_by_ml"
            else:
                reason = "signal_confirmed_by_ml"

        decision = MLDecision(
            enabled=True,
            mode=self.config.mode,
            base_signal=base_signal,
            final_signal=final_signal,
            allow_long=allow_long,
            allow_short=allow_short,
            long_prob=long_prob,
            short_prob=short_prob,
            reason=reason,
            features_used=pred["feature_columns"],
            debug={"threshold": pred["threshold"]},
        )
        self._log(strategy_context, decision)
        return decision

    def _log(self, strategy_context: dict[str, Any], decision: MLDecision) -> None:
        payload = {
            "symbol": strategy_context.get("symbol"),
            "timeframe": strategy_context.get("timeframe"),
            "session": strategy_context.get("session"),
            **decision.to_dict(),
        }
        append_jsonl(self.config.log_path, payload)
