"""
paper_s2.py — БУМАЖНАЯ (виртуальная) стратегия #2, работающая параллельно.

Не торгует реальными деньгами. Каждый бар считает сигнал стратегии #2,
ведёт виртуальную позицию с теми же RTH-правилами, что и реал:
  - входы только в окне RTH,
  - force-close в конце дня (не держим ночь),
и оценивает виртуальный P&L (грубо, через движение базового актива × дельту),
чтобы можно было ОБКАТАТЬ #2 на живых данных до перевода на реальные деньги.

Состояние держится в AppState.paper_s2_state (переживает рестарт через персист).
Сообщения шлются в закрытый канал с пометкой "📄 БУМАЖНАЯ #2".
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


# Грубая дельта для оценки P&L опциона по движению базового актива.
# Реальный опцион ~0.375 по дизайну; берём 0.4 как в бэктесте.
_PAPER_DELTA = 0.4


@dataclass
class PaperPosition:
    direction: str        # "CALL" | "PUT"
    entry_underlying: float
    entry_ts: str
    entry_signal_price: float


@dataclass
class PaperS2State:
    position: Optional[dict] = None      # PaperPosition как dict (для персиста)
    day_pnl: float = 0.0                 # накопленный виртуальный P&L за день ($/1 контракт)
    day_trades: int = 0
    day_wins: int = 0
    session_date: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def _est_pnl(direction: str, entry_u: float, exit_u: float) -> float:
    """Грубая оценка P&L опциона на 1 контракт через движение базового × дельта."""
    move = exit_u - entry_u
    if direction == "PUT":
        move = -move
    return move * _PAPER_DELTA * 100.0


def reset_day_if_needed(st: PaperS2State, session_date: str) -> None:
    if st.session_date != session_date:
        st.session_date = session_date
        st.day_pnl = 0.0
        st.day_trades = 0
        st.day_wins = 0


def process_paper_signal(st: PaperS2State, action: str, underlying: float,
                         ts: str, can_open: bool, force_close: bool
                         ) -> Optional[str]:
    """
    Обновляет виртуальную позицию #2 по сигналу.

    action: "BUY"/"SELL"/"HOLD" — сигнал стратегии #2 на этом баре.
    underlying: текущая цена QQQ.
    can_open: можно ли открывать (окно RTH).
    force_close: конец дня — закрыть виртуальную позицию.

    Возвращает текст сообщения для канала, если было событие (открытие/закрытие),
    иначе None.
    """
    pos = st.position

    # 1) Принудительное закрытие в конце дня (не держим ночь)
    if force_close and pos is not None:
        pnl = _est_pnl(pos["direction"], pos["entry_underlying"], underlying)
        st.day_pnl += pnl
        st.day_trades += 1
        if pnl > 0:
            st.day_wins += 1
        d = pos["direction"]
        st.position = None
        return (f"📄 <b>БУМАЖНАЯ #2</b>: 🔚 закрытие {d} в конце дня\n"
                f"QQQ {pos['entry_underlying']:.2f} → {underlying:.2f} | "
                f"~P&L ${pnl:+.0f} (оценка)")

    if action not in ("BUY", "SELL"):
        return None

    want = "CALL" if action == "BUY" else "PUT"

    # 2) Разворот: если позиция в другую сторону — закрываем и (если можно) открываем
    if pos is not None and pos["direction"] != want:
        pnl = _est_pnl(pos["direction"], pos["entry_underlying"], underlying)
        st.day_pnl += pnl
        st.day_trades += 1
        if pnl > 0:
            st.day_wins += 1
        old = pos["direction"]
        st.position = None
        msg_close = (f"📄 <b>БУМАЖНАЯ #2</b>: 🔄 разворот, закрытие {old}\n"
                     f"QQQ {pos['entry_underlying']:.2f} → {underlying:.2f} | "
                     f"~P&L ${pnl:+.0f} (оценка)")
        # открываем новую в окне RTH
        if can_open:
            st.position = PaperPosition(want, underlying, ts, underlying).__dict__
            return msg_close + f"\n📄 БУМАЖНАЯ #2: 📈 открытие {want} @ QQQ {underlying:.2f}"
        return msg_close

    # 3) Нет позиции — открываем (в окне RTH)
    if pos is None and can_open:
        st.position = PaperPosition(want, underlying, ts, underlying).__dict__
        return (f"📄 <b>БУМАЖНАЯ #2</b>: 📈 открытие {want} @ QQQ {underlying:.2f}")

    # 4) Позиция в ту же сторону — держим, ничего не шлём
    return None


def format_paper_day_summary(st: PaperS2State) -> str:
    """Итог бумажной #2 за день — для конца сессии."""
    wr = (st.day_wins / st.day_trades * 100) if st.day_trades else 0.0
    sign = "🟢" if st.day_pnl >= 0 else "🔴"
    return (f"📄 <b>Бумажная #2 — итог дня</b>\n"
            f"Сделок: {st.day_trades} | винрейт {wr:.0f}%\n"
            f"{sign} Виртуальный P&L (на 1 контракт, оценка): ${st.day_pnl:+.0f}")
