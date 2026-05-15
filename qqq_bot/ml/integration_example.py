from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .service import MLTradingService


@dataclass
class BaseSignal:
    side: str
    buy_score: float = 0.0
    sell_score: float = 0.0
    nearU: bool = False
    nearL: bool = False
    bounceU: bool = False
    bounceL: bool = False
    bb_ok: bool = False
    rsi_ok: bool = False
    ema_up: bool = False
    ema_dn: bool = False


def apply_ml_gate(
    bars_df: pd.DataFrame,
    signal: BaseSignal,
    current_position: str = "FLAT",
    symbol: str = "QQQ.US",
    timeframe: str = "5m",
    session: str = "regular",
) -> dict[str, Any]:
    ml_service = MLTradingService.from_env()

    ml_result = ml_service.decide(
        bars=bars_df,
        base_signal=signal.side,
        strategy_context={
            "buy_score": signal.buy_score,
            "sell_score": signal.sell_score,
            "nearU": signal.nearU,
            "nearL": signal.nearL,
            "bounceU": signal.bounceU,
            "bounceL": signal.bounceL,
            "bb_ok": signal.bb_ok,
            "rsi_ok": signal.rsi_ok,
            "ema_up": signal.ema_up,
            "ema_dn": signal.ema_dn,
            "session": session,
            "symbol": symbol,
            "timeframe": timeframe,
            "timeframe_minutes": 5.0,
        },
        current_position=current_position,
    )

    return {
        "base_signal": signal.side,
        "final_signal": ml_result.final_signal,
        "ml_mode": ml_result.mode,
        "ml_reason": ml_result.reason,
        "ml_long_prob": round(ml_result.long_prob, 4),
        "ml_short_prob": round(ml_result.short_prob, 4),
        "allow_long": ml_result.allow_long,
        "allow_short": ml_result.allow_short,
    }
