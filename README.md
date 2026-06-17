# QQQ Trading Signal Bot (TraderNet realtime + Telegram)

## Что делает
- Подключается к TraderNet WebSocket и получает realtime котировки по SYMBOL.
- Агрегирует тики в бары TIMEFRAME_MIN.
- Считает RSI/EMA/Bollinger Bands.
- Генерирует сигналы BUY/SELL по логике из ТЗ.
- Рисует график (без разрывов вне торговых часов — ось X это индекс баров).
- Отправляет сигнал + картинку в Telegram-канал.

## Установка
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
