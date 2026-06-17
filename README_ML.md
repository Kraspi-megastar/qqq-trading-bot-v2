# ML bundle for QQQ trading bot

Этот набор файлов добавляет в существующий бот отдельный ML-слой, не заменяя базовую стратегию.

## Что входит
- `qqq_bot/ml/features.py` — расчет признаков
- `qqq_bot/ml/labels.py` — target/label generation
- `qqq_bot/ml/dataset.py` — подготовка train/inference dataset
- `qqq_bot/ml/train.py` — обучение модели
- `qqq_bot/ml/predict.py` — загрузка модели и прогноз вероятности
- `qqq_bot/ml/service.py` — рантайм-сервис для advisory / soft / hard режима
- `qqq_bot/ml/backtest.py` — walk-forward backtest
- `qqq_bot/ml/telemetry.py` — логирование предсказаний
- `qqq_bot/ml/schemas.py` — типы данных/конфиг
- `qqq_bot/ml/integration_example.py` — пример интеграции в текущую сигнальную логику
- `.env.ml.example` — переменные окружения
- `PATCH_NOTES.md` — куда встроить в существующий проект

## Рекомендуемый порядок внедрения
1. Установить зависимости из `requirements-ml.txt`
2. Добавить переменные из `.env.ml.example` в свой `.env`
3. Подключить `MLTradingService` в том месте, где формируется BUY/SELL/HOLD
4. Запустить в режиме `advisory`
5. Накопить предсказания и фактические результаты
6. Обучить модель командой из `train.py`
7. После валидации перевести в `soft`, затем при необходимости в `hard`

## Структура входных данных
Ожидается DataFrame баров с колонками:
- `timestamp` или DatetimeIndex
- `open`, `high`, `low`, `close`, `volume`

Опционально, если уже считаются в боте:
- `rsi`, `ema9`, `ema21`, `bb_lower`, `bb_mid`, `bb_upper`, `atr`, `vwap`

Если индикаторов нет, модуль рассчитает основные сам.

## Как использовать в проде
`MLTradingService.decide(...)` принимает:
- последние бары
- базовый сигнал стратегии
- словарь `strategy_context` с доп. полями (`buy_score`, `sell_score`, `nearU`, `nearL`, ...)
- опционально `current_position`

Возвращает:
- вероятность long / short
- окончательный вердикт ML
- режим работы
- причину решения

## Команды
### Обучение
```bash
python -m qqq_bot.ml.train --input data/qqq_bars.parquet --model-dir qqq_bot/ml/models
```

### Бэктест
```bash
python -m qqq_bot.ml.backtest --input data/qqq_bars.parquet --model-dir qqq_bot/ml/models
```
