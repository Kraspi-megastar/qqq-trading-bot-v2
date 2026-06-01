# CHANGELOG_UPDATE_20260601

## Обновлено

- `qqq_bot/signals.py` — совместимый `compute_signal()`, STR#2 только regular-session, явные типы сигналов, breakout/pullback, outcome ML approval.
- `qqq_bot/scheduler.py` — подключен outcome ML в live loop, история сигналов хранит semantic signal type/mode, STR#2 runtime поддерживает LONG/SHORT.
- `qqq_bot/bot.py` — Telegram-сообщение показывает signal type/mode/ML probabilities и опционный блок; убран сломанный вызов `state_file_path`.
- `qqq_bot/handlers.py` — `/status` показывает semantic signal type и корректную STR#2 позицию из runtime.
- `qqq_bot/cache.py` — добавлены поля `last_signal_type`, `last_signal_mode`, `last_signal_price`.
- `qqq_bot/state_store.py` — runtime STR#2 теперь допускает `SHORT`.
- `qqq_bot/option_quotes.py` — безопасно пропускает async quote methods в sync provider; реальный quote источник подключается через `OPTION_QUOTE_URL` или sync `get_option_quote()`.
- `.env.example` — добавлены параметры STR#2, ML outcome и options; `TRADERNET_SID` очищен; `COOLDOWN_SECONDS=0`.
- `requirements.txt` — исправлен перенос строки и добавлены ML зависимости.

## Добавлено

- `qqq_bot/signal_types.py`
- `qqq_bot/market_session.py`
- `qqq_bot/options_signal.py`
- `qqq_bot/option_quotes.py`
- `qqq_bot/ml/__init__.py`
- `qqq_bot/ml/features_outcome.py`
- `qqq_bot/ml/outcome_labels.py`
- `qqq_bot/ml/outcome_dataset.py`
- `qqq_bot/ml/outcome_train.py`
- `qqq_bot/ml/outcome_service.py`
- `requirements-ml-outcome.txt`
- `docs/UPDATE_PLAN_QQQ_BOT.md`

## Удалено из архива как лишнее

- `.git/`
- `.idea/`
- `__pycache__/`
- `*.pyc`
- `qqq_bot_patch_20260601/`
- пустая `logs/`

## Проверка

Выполнено:

```bash
python -m py_compile qqq_bot/*.py qqq_bot/ml/*.py ml/*.py
```

Проверены импорты:

```text
qqq_bot.signals OK
qqq_bot.scheduler OK
qqq_bot.options_signal OK
qqq_bot.ml.outcome_service OK
ml.service OK
```

## TraderNet option quotes integration

Added direct best-effort quote extraction for option tickers from TraderNet `securities/export`:

- `TraderNetClient.get_option_quote()` now requests `ltp,ltt,bbp,bap,bbs,bas,vol,vlt,oi,iv,delta,gamma,theta,vega`.
- `DefaultOptionQuoteProvider` can synchronously query `client.quotes_url`, so the existing synchronous options signal path can log bid/ask/last without awaiting inside Telegram send flow.
- `bot.py` now passes `TRADERNET_SID` and `TRADERNET_TIMEOUT_SECONDS` into `TraderNetClient`.
- If TraderNet does not return greeks/IV for US options, logs still store the raw quote fields and bid/ask/ltp when available.
