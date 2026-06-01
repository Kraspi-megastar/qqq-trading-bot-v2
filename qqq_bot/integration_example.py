"""Minimal integration example for scheduler.py / bot.py.

Do not run this file directly. Copy the relevant lines into the place where your bot
already receives fresh OHLCV bars and builds Telegram messages.
"""

from qqq_bot.ml.outcome_service import MLOutcomeService
from qqq_bot.options_signal import process_options_signal
from qqq_bot.signals import generate_str2_decision

ml_service = MLOutcomeService()


def handle_new_bars(bars, tradernet_client=None, current_position="FLAT"):
    ml_decision = ml_service.predict(bars)
    strategy_decision = generate_str2_decision(
        bars,
        ml_decision=ml_decision,
        current_position=current_position,
    )
    options_decision = process_options_signal(
        strategy_decision,
        tradernet_client=tradernet_client,
    )
    return strategy_decision, options_decision
