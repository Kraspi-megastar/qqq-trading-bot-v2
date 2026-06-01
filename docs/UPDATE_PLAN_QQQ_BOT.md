# План обновления QQQ Trading Bot — STR#2 + options + outcome ML

## Что изменено

- STR#2 генерирует сигналы только внутри regular-session NYSE/Nasdaq, но индикаторы считаются по полной истории баров, включая extended hours.
- Введены явные типы сигналов: `OPEN_LONG`, `CLOSE_LONG`, `OPEN_SHORT`, `CLOSE_SHORT`, `HOLD`.
- Legacy `BUY/SELL/HOLD` сохранены только для графика и Telegram-совместимости.
- `SELL` больше не означает автоматическую покупку PUT. PUT открывается только при `OPEN_SHORT`.
- Разделены режимы входа: `breakout` и `pullback`.
- 0DTE не отключены, но разрешены только при сильном движении (`breakout`) и подтверждении outcome ML.
- Добавлено логирование опционных событий и котировок в `logs/options_events.jsonl`, `logs/options_quotes.jsonl`, `logs/options_quotes_missing.jsonl`.
- Cooldown для новой логики не используется: в `.env` поставь `COOLDOWN_SECONDS=0`.

## Файлы, которые обновлены

```text
qqq_bot/signals.py
qqq_bot/scheduler.py
qqq_bot/bot.py
qqq_bot/handlers.py
qqq_bot/cache.py
qqq_bot/state_store.py
qqq_bot/option_quotes.py
.env.example
requirements.txt
```

## Файлы, которые добавлены

```text
qqq_bot/signal_types.py
qqq_bot/market_session.py
qqq_bot/options_signal.py
qqq_bot/option_quotes.py
qqq_bot/ml/__init__.py
qqq_bot/ml/features_outcome.py
qqq_bot/ml/outcome_labels.py
qqq_bot/ml/outcome_dataset.py
qqq_bot/ml/outcome_train.py
qqq_bot/ml/outcome_service.py
requirements-ml-outcome.txt
```

## Что удалено как лишнее из архива

```text
.git/
.idea/
__pycache__/
qqq_bot_patch_20260601/
logs/  # пустая папка; бот создаст ее сам
```

## Обновление на Windows

```powershell
cd D:\qqq_trading_bot

New-Item -ItemType Directory -Force backup_before_update
Copy-Item qqq_bot backup_before_update\qqq_bot -Recurse -Force
Copy-Item requirements.txt backup_before_update\requirements.txt -Force
Copy-Item .env backup_before_update\.env -Force -ErrorAction SilentlyContinue

.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install -r requirements-ml-outcome.txt

.\.venv\Scripts\python.exe -m py_compile qqq_bot\*.py qqq_bot\ml\*.py
```

В `.env` обязательно добавь или проверь:

```env
STRATEGY_ID=2
COOLDOWN_SECONDS=0
STR2_REGULAR_ONLY=true
STR2_ALLOW_SHORT_ENTRIES=true
ML_ENABLED=true
ML_OUTCOME_MODEL_DIR=qqq_bot/ml/models
OPTIONS_ENABLE_0DTE=true
OPTIONS_REQUIRE_STRONG_MOVE_FOR_0DTE=true
OPTIONS_REQUIRE_ML_FOR_0DTE=true
OPTIONS_LOG_QUOTES=true
OPTIONS_LOGS_DIR=logs
```

Запуск:

```powershell
.\.venv\Scripts\python.exe run.py
```

## Обновление на VPS

```bash
cd ~/apps/qqq-bot/repo
cp -a qqq_bot qqq_bot.bak_$(date +%Y%m%d_%H%M)
cp requirements.txt requirements.txt.bak_$(date +%Y%m%d_%H%M)

~/apps/qqq-bot/venv/bin/pip install -r requirements.txt
~/apps/qqq-bot/venv/bin/pip install -r requirements-ml-outcome.txt

~/apps/qqq-bot/venv/bin/python -m py_compile qqq_bot/*.py qqq_bot/ml/*.py

sudo systemctl restart qqq-bot-staging
sudo systemctl status qqq-bot-staging --no-pager
journalctl -u qqq-bot-staging -n 100 --no-pager
```

## Обучение outcome ML

Нужен CSV с 5m-барами:

```text
ts,open,high,low,close,volume
```

Собрать датасет:

```powershell
.\.venv\Scripts\python.exe -m qqq_bot.ml.outcome_dataset --input data\qqq_5m_history.csv --output data\ml_outcome_dataset.csv --horizon 12
```

Обучить модели:

```powershell
.\.venv\Scripts\python.exe -m qqq_bot.ml.outcome_train --dataset data\ml_outcome_dataset.csv --model-dir qqq_bot\ml\models --horizon 12
```

После обучения должны появиться:

```text
qqq_bot/ml/models/outcome_long_05atr.joblib
qqq_bot/ml/models/outcome_long_10atr.joblib
qqq_bot/ml/models/outcome_short_05atr.joblib
qqq_bot/ml/models/outcome_short_10atr.joblib
qqq_bot/ml/models/outcome_metadata.json
```

Пока модели не обучены/не загружены, обычные STR#2 сигналы будут работать, но 0DTE входы будут блокироваться из-за требования ML-подтверждения.

## Проверка логики после запуска

В Telegram проверь:

```text
/status
/chart
/stats
```

В логах проверь:

```bash
ls -lah logs
cat logs/options_events.jsonl | tail -20
cat logs/options_quotes_missing.jsonl | tail -20
```

Если `options_quotes_missing.jsonl` заполняется, значит бот выбирает опционные тикеры, но источник реальных bid/ask/greeks еще не подключен. Для этого задай `OPTION_QUOTE_URL` или добавь в TraderNetClient синхронный метод `get_option_quote()`.

## Проверка котировок опционов TraderNet

После запуска в regular-session проверьте, что создается файл:

```text
logs/options_quotes.jsonl
```

Если вместо него заполняется:

```text
logs/options_quotes_missing.jsonl
```

значит TraderNet не вернул данные по конкретному опционному тикеру или формат тикера отличается от ожидаемого. В этом случае нужно взять фактический тикер опциона из терминала TraderNet и сравнить его с тикером, который генерирует бот.
