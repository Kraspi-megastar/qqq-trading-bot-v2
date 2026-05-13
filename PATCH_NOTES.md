# Куда встраивать в текущий проект

## 1) Инициализация при старте бота
В файле, где создается основной сервис/runner бота, добавь:

```python
from qqq_bot.ml.service import MLTradingService
from qqq_bot.ml.schemas import MLConfig

ml_service = MLTradingService.from_env()
```

## 2) В точке принятия торгового решения
Сейчас у тебя, вероятно, есть что-то вроде:
- получили бары
- посчитали индикаторы
- стратегия выдала BUY/SELL/HOLD

После этого добавь:

```python
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
        "session": "regular",
        "symbol": "QQQ.US",
        "timeframe": "5m",
    },
    current_position=state.position,
)
```

И уже дальше:
- в `advisory` режиме не блокируй сигнал, только логируй
- в `soft/hard` режиме используй `ml_result.final_signal`

## 3) В телеграм-сообщение
Полезно добавить:
- `ML long prob`
- `ML short prob`
- `ML verdict`
- `ML mode`

## 4) Что НЕ делать на старте
- не заменяй стратегию моделью полностью
- не включай hard-режим без paper-периода
- не обучай модель на future-leaking признаках
