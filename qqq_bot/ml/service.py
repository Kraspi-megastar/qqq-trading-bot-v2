from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .features import DEFAULT_FEATURES, latest_feature_row
from .predict import MLPredictor
from .schemas import MLConfig, MLDecision
from .telemetry import append_jsonl


class MLTradingService:
    """
    Live ML layer for the QQQ bot.

    Safe first-stage behavior:
    - advisory mode by default;
    - does not block base strategy signals unless mode is soft/hard and block_weak_signals=1;
    - logs every evaluated closed bar to ML_LOG_PATH, even if no model is loaded yet.
    """

    def __init__(self, config: MLConfig):
        self.config = config
        self.config.validate()
        self.predictor: Optional[MLPredictor] = None
        self.load_error: str | None = None

        if self.config.enabled and Path(self.config.model_path).exists():
            try:
                self.predictor = MLPredictor(
                    model_path=self.config.model_path,
                    metadata_path=self.config.metadata_path if Path(self.config.metadata_path).exists() else None,
                )
            except Exception as exc:
                self.predictor = None
                self.load_error = repr(exc)

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
            only_regular_session=os.getenv("ML_ONLY_REGULAR_SESSION", "0") == "1",
            block_weak_signals=os.getenv("ML_BLOCK_WEAK_SIGNALS", "0") == "1",
        )
        return cls(cfg)

    def _empty_decision(
        self,
        base_signal: str,
        reason: str,
        *,
        debug: dict[str, Any] | None = None,
    ) -> MLDecision:
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
            debug=debug,
        )

    def _latest_bar_payload(self, bars: pd.DataFrame) -> dict[str, Any]:
        if bars is None or len(bars) == 0:
            return {}

        last = bars.iloc[-1]
        ts = None
        if "timestamp" in bars.columns:
            ts = last.get("timestamp")
        elif "ts" in bars.columns:
            ts = last.get("ts")
        elif isinstance(bars.index, pd.DatetimeIndex):
            ts = bars.index[-1]

        payload: dict[str, Any] = {"bar_ts": ts}
        for col in ("open", "high", "low", "close", "volume", "rsi", "vwap", "atr"):
            if col in bars.columns:
                try:
                    val = float(last[col])
                    payload[col] = None if np.isnan(val) or np.isinf(val) else val
                except Exception:
                    payload[col] = None
        return payload

    def _feature_snapshot(
        self,
        bars: pd.DataFrame,
        strategy_context: dict[str, Any],
    ) -> tuple[list[str] | None, dict[str, Any] | None, str | None]:
        try:
            feature_columns = None
            if self.predictor is not None and getattr(self.predictor, "feature_columns", None):
                feature_columns = self.predictor.feature_columns
            row = latest_feature_row(
                bars,
                feature_columns=feature_columns or DEFAULT_FEATURES,
                strategy_context=strategy_context,
            )
            return list(row.columns), row.iloc[0].to_dict(), None
        except Exception as exc:
            return None, None, repr(exc)

    def decide(
        self,
        bars: pd.DataFrame,
        base_signal: str,
        strategy_context: Optional[dict[str, Any]] = None,
        current_position: Optional[str] = None,
    ) -> MLDecision:
        base_signal = str(base_signal or "HOLD").upper()
        strategy_context = dict(strategy_context or {})

        latest_bar = self._latest_bar_payload(bars)
        debug: dict[str, Any] = {
            "model_path": self.config.model_path,
            "model_loaded": self.predictor is not None,
            "load_error": self.load_error,
            "bars": 0 if bars is None else int(len(bars)),
            **latest_bar,
        }

        if not self.config.enabled:
            decision = self._empty_decision(base_signal, "ml_disabled", debug=debug)
            self._log(strategy_context, decision)
            return decision

        if bars is None or len(bars) < self.config.min_bars:
            decision = self._empty_decision(base_signal, "insufficient_bars", debug=debug)
            self._log(strategy_context, decision)
            return decision

        session = str(strategy_context.get("session", "")).lower()
        if self.config.only_regular_session and session not in {"regular", "rth"}:
            decision = self._empty_decision(base_signal, "session_filtered", debug=debug)
            self._log(strategy_context, decision)
            return decision

        if base_signal == "BUY":
            strategy_context["signal_is_buy"] = 1
            strategy_context["signal_is_sell"] = 0
        elif base_signal == "SELL":
            strategy_context["signal_is_buy"] = 0
            strategy_context["signal_is_sell"] = 1
        else:
            strategy_context["signal_is_buy"] = 0
            strategy_context["signal_is_sell"] = 0

        pos = str(current_position or "").upper()
        strategy_context["position_is_long"] = 1 if pos == "LONG" else 0
        strategy_context["position_is_short"] = 1 if pos == "SHORT" else 0

        feature_cols, feature_values, feature_error = self._feature_snapshot(bars, strategy_context)
        if feature_error:
            debug["feature_error"] = feature_error

        if self.predictor is None:
            decision = self._empty_decision(base_signal, "model_not_loaded", debug=debug)
            decision.features_used = feature_cols
            if feature_values is not None:
                decision.debug = {**(decision.debug or {}), "feature_values": feature_values}
            self._log(strategy_context, decision)
            return decision

        pred = self.predictor.predict_latest(bars, strategy_context=strategy_context)
        prob = float(pred["probability"])

        target_mode = str(getattr(self.predictor, "metadata", {}).get("target_mode", "signal_quality"))

        if target_mode == "signal_quality":
            # prob = probability that the current base BUY/SELL signal will work.
            if base_signal == "BUY":
                long_prob = prob
                short_prob = 0.0
            elif base_signal == "SELL":
                long_prob = 0.0
                short_prob = prob
            else:
                long_prob = prob
                short_prob = 0.0

            allow_long = base_signal == "BUY" and long_prob >= self.config.long_threshold
            allow_short = base_signal == "SELL" and short_prob >= self.config.short_threshold
        else:
            # directional model: probability is interpreted as upside probability.
            long_prob = prob
            short_prob = float(1.0 - prob)
            allow_long = long_prob >= self.config.long_threshold
            allow_short = short_prob >= self.config.short_threshold

        final_signal = base_signal
        reason = "advisory_no_override"

        if base_signal == "HOLD":
            final_signal = "HOLD"
            reason = "hold_base_signal"
        elif self.config.mode in {"soft", "hard"} and self.config.block_weak_signals:
            if base_signal == "BUY" and not allow_long:
                final_signal = "HOLD"
                reason = "buy_blocked_by_ml"
            elif base_signal == "SELL" and not allow_short:
                final_signal = "HOLD"
                reason = "sell_blocked_by_ml"
            else:
                reason = "signal_confirmed_by_ml"

        debug.update({
            "threshold": pred.get("threshold"),
            "target_mode": target_mode,
            "feature_values": pred.get("feature_values") or feature_values,
        })

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
            debug=debug,
        )
        self._log(strategy_context, decision)
        return decision

    def _log(self, strategy_context: dict[str, Any], decision: MLDecision) -> None:
        payload = {
            "event_ts": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": strategy_context.get("symbol"),
            "timeframe": strategy_context.get("timeframe"),
            "session": strategy_context.get("session"),
            "strategy_id": strategy_context.get("strategy_id"),
            **decision.to_dict(),
        }
        append_jsonl(self.config.log_path, payload)
