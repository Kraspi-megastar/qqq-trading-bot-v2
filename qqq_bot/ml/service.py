from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import MLConfig, MLDecision
from .telemetry import append_jsonl


class MLTradingService:
    def __init__(self, config: MLConfig):
        self.config = config
        self.config.validate()
        self.predictor = None
        self.load_error: str | None = None

        if self.config.enabled:
            try:
                from .predict import MLPredictor

                metadata_exists = Path(self.config.metadata_path).exists()
                legacy_model_exists = Path(self.config.model_path).exists()
                if metadata_exists or legacy_model_exists:
                    self.predictor = MLPredictor(
                        model_dir=self.config.model_dir,
                        metadata_path=self.config.metadata_path if metadata_exists else None,
                        model_path=self.config.model_path if legacy_model_exists else None,
                    )
            except Exception as exc:
                self.load_error = repr(exc)
                self.predictor = None

    @classmethod
    def from_env(cls) -> "MLTradingService":
        model_dir = os.getenv("ML_MODEL_DIR", "qqq_bot/ml/models")
        cfg = MLConfig(
            enabled=os.getenv("ML_ENABLED", "0") == "1",
            mode=os.getenv("ML_MODE", "advisory"),
            model_dir=model_dir,
            model_path=os.getenv("ML_MODEL_PATH", str(Path(model_dir) / "qqq_signal_filter.joblib")),
            metadata_path=os.getenv("ML_METADATA_PATH", str(Path(model_dir) / "metadata.json")),
            long_threshold=float(os.getenv("ML_LONG_THRESHOLD", "0.62")),
            short_threshold=float(os.getenv("ML_SHORT_THRESHOLD", "0.62")),
            min_bars=int(os.getenv("ML_MIN_BARS", "240")),
            log_path=os.getenv("ML_LOG_PATH", "logs/ml_predictions.jsonl"),
            only_regular_session=os.getenv("ML_ONLY_REGULAR_SESSION", "1") == "1",
            block_weak_signals=os.getenv("ML_BLOCK_WEAK_SIGNALS", "1") == "1",
            block_exits=os.getenv("ML_BLOCK_EXITS", "0") == "1",
            min_edge=float(os.getenv("ML_MIN_EDGE", "0.05")),
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
            debug={"load_error": self.load_error} if self.load_error else None,
        )

    @staticmethod
    def _is_exit_signal(base_signal: str, current_position: str | None) -> bool:
        pos = str(current_position or "").upper()
        if base_signal == "SELL" and pos in {"LONG", "CALL"}:
            return True
        if base_signal == "BUY" and pos in {"SHORT", "PUT"}:
            return True
        return False

    def decide(
        self,
        bars: pd.DataFrame,
        base_signal: str,
        strategy_context: dict[str, Any] | None = None,
        current_position: str | None = None,
    ) -> MLDecision:
        base_signal = str(base_signal).upper()
        if base_signal not in {"BUY", "SELL", "HOLD"}:
            base_signal = "HOLD"

        if not self.config.enabled:
            return self._empty_decision(base_signal, "ml_disabled")
        if self.predictor is None:
            return self._empty_decision(base_signal, "model_not_loaded")
        if bars is None or len(bars) < self.config.min_bars:
            return self._empty_decision(base_signal, "insufficient_bars")

        strategy_context = dict(strategy_context or {})
        strategy_context["base_signal"] = base_signal
        session = str(strategy_context.get("session", "")).lower()
        if self.config.only_regular_session and session and session not in {"regular", "rth"}:
            return self._empty_decision(base_signal, "session_filtered")

        if base_signal == "BUY":
            strategy_context["signal_is_buy"] = 1
            strategy_context["signal_is_sell"] = 0
        elif base_signal == "SELL":
            strategy_context["signal_is_buy"] = 0
            strategy_context["signal_is_sell"] = 1

        pos = str(current_position or "").upper()
        strategy_context["position_is_long"] = int(pos in {"LONG", "CALL"})
        strategy_context["position_is_short"] = int(pos in {"SHORT", "PUT"})

        try:
            pred = self.predictor.predict_latest(bars, strategy_context=strategy_context)
        except Exception as exc:
            return self._empty_decision(base_signal, f"prediction_error:{repr(exc)}")

        long_prob = float(pred.get("long_prob", 0.5))
        short_prob = float(pred.get("short_prob", 0.5))
        long_threshold = float(self.config.long_threshold)
        short_threshold = float(self.config.short_threshold)

        long_edge = long_prob - short_prob
        short_edge = short_prob - long_prob
        allow_long = long_prob >= long_threshold and long_edge >= self.config.min_edge
        allow_short = short_prob >= short_threshold and short_edge >= self.config.min_edge

        final_signal = base_signal
        reason = "advisory_no_override"
        is_exit = self._is_exit_signal(base_signal, current_position)

        if base_signal == "HOLD":
            reason = "hold_base_signal"
            allow_long = False
            allow_short = False
        elif is_exit and not self.config.block_exits:
            reason = "exit_not_blocked_by_ml"
            allow_long = base_signal == "BUY"
            allow_short = base_signal == "SELL"
        elif self.config.mode in {"soft", "hard"} and self.config.block_weak_signals:
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
            features_used=pred.get("feature_columns"),
            debug={
                "thresholds": pred.get("thresholds"),
                "target_mode": pred.get("target_mode"),
                "long_edge": long_edge,
                "short_edge": short_edge,
                "is_exit": is_exit,
            },
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
