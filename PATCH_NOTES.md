# Patch Notes

## v3.0 (рефакторинг + опционные сигналы)

### Новое
- **Опционные сигналы в стратегии #1**: при каждом BUY/SELL сигнал теперь включает
  рекомендованный опционный контракт (тикер вида `QQQ 250718C480`).
  - BUY → CALL, SELL → PUT
  - Страйк: ATM (настраивается через OPTION_STRIKE_OFFSET)
  - Экспирация: ближайшая пятница ≥ OPTION_MIN_DTE дней
  - Сначала пробуем получить реальный тикер из TraderNet getOptionChain;
    если API не поддерживает — строим тикер аналитически
- **Новая команда /options** — показывает текущий опционный контракт по последнему сигналу
- **Новая команда /config** — показывает все параметры включая опционные настройки
- **JSONL-лог решений** — каждое решение пишется в `{CACHE_DIR}/logs/signals.jsonl`
  для последующего бэктестинга

### Архитектурные изменения
- **pipeline.py** — новый модуль с `bars_to_df`, `add_indicators`, `min_bars_for_indicators`.
  Устранено дублирование между scheduler.py и handlers.py.
- **compute_signal() — чистая функция**: больше не мутирует внешний объект.
  Новое состояние Strategy#2 возвращается в `decision.new_state` и применяется
  явно в scheduler. Это устраняет баг с двойной мутацией при /status.
- **Strategy2Runtime → Strategy2State** (из signals.py). Единый dataclass вместо двух.
- **state_store.py**: атомарная запись `state.json` через tmp→replace
  (как в cache.py). Аварийное завершение больше не оставляет пустой файл.
- Retry с экспоненциальным backoff для `get_quote_ltp` (3 попытки).

### Исправленные баги
- Дублирование `_bars_to_df`/`_add_indicators` между scheduler и handlers.
- `vol_ma` теперь явно выставляется в NaN при нулевом объёме — vol_filter
  корректно деактивируется вместо ложного срабатывания.
- Синтаксическая ошибка в requirements.txt (склейка с pyproject.toml) исправлена.
- /status больше не мутирует app.strategy2 при пересчёте сигнала.

## v2.0
- Стратегия #2: MACD + VWAP + RSI + Supertrend + ATR-stop
- Персистентное хранение позиции (state.json)
- ML-слой (опциональный, отдельный модуль)

## v1.0
- Стратегия #1: BB + EMA + RSI
- TraderNet API
- Telegram-бот (aiogram)
