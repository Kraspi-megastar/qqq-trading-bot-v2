"""
⚠️ DORMANT / НЕ ИСПОЛЬЗУЕТСЯ (отключено при оптимизации).

Консенсус-голосование трёх источников (#1, #2, ML) было advisory-only —
оно НЕ влияло на торговые решения, только отображалось в сообщениях.
Удалено из scheduler/bot при чистке. Код сохранён на случай возврата.
Бот торгует по стратегии #1 с фильтром тренда 1h.
"""

"""
consensus.py — ансамбль из трёх источников с окном согласия.

Источники (каждый голосует +1 / 0 / -1):
  - Стратегия #1 (BB+EMA+RSI)
  - Стратегия #2 (MACD+VWAP+Supertrend)
  - ML-блок (long_prob/short_prob → edge → голос)

Окно согласия: голос источника считается "активным" N баров после его прихода.
Итоговый score = взвешенная сумма активных голосов (по умолчанию веса 1.0).

Консенсус НЕ управляет размером позиции и НЕ открывает опционы — это делает
только стратегия #1. Консенсус используется для:
  - показа "силы сигнала" (разбивка голосов) при сигнале #1,
  - предупреждения о конфликте против открытой позиции,
  - досрочного закрытия, если ОБА других источника единогласно против позиции.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsensusConfig:
    enabled: bool = True
    agree_window_bars: int = 12       # окно согласия (баров)
    weight_s1: float = 1.0
    weight_s2: float = 1.0
    weight_ml: float = 1.0
    ml_min_edge: float = 0.05         # порог |long_prob - short_prob| для ML-голоса
    # порог "сильного" консенсуса против позиции (для авто-закрытия участвует
    # отдельная проверка единогласия #2 и ML, не только score)
    conflict_score_threshold: float = 1.0


# ────────────────────────────────────────────────────────────────────────────
# Голос одного источника с временем жизни
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceVote:
    vote: int = 0                     # +1 / 0 / -1
    bar_index: int = -1               # индекс бара, когда голос установлен
    detail: str = ""                  # текстовое пояснение (RSI, prob и т.д.)

    def is_active(self, current_bar: int, window: int) -> bool:
        if self.vote == 0 or self.bar_index < 0:
            return False
        return (current_bar - self.bar_index) < window


# ────────────────────────────────────────────────────────────────────────────
# Состояние консенсуса (хранится в AppState)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsensusState:
    s1: SourceVote = field(default_factory=SourceVote)
    s2: SourceVote = field(default_factory=SourceVote)
    ml: SourceVote = field(default_factory=SourceVote)
    bar_counter: int = 0              # монотонный счётчик баров

    def tick(self) -> None:
        self.bar_counter += 1

    def set_vote(self, source: str, vote: int, detail: str = "") -> None:
        sv = SourceVote(vote=vote, bar_index=self.bar_counter, detail=detail)
        if source == "s1":
            self.s1 = sv
        elif source == "s2":
            self.s2 = sv
        elif source == "ml":
            self.ml = sv

    def clear_vote(self, source: str) -> None:
        """Сбрасывает голос источника в нейтральный (для непрерывных источников,
        когда текущая оценка стала нейтральной — старый голос не должен висеть)."""
        empty = SourceVote()
        if source == "s1":
            self.s1 = empty
        elif source == "s2":
            self.s2 = empty
        elif source == "ml":
            self.ml = empty


# ────────────────────────────────────────────────────────────────────────────
# Результат расчёта консенсуса
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsensusResult:
    score: float                      # итоговый взвешенный балл
    s1_vote: int
    s2_vote: int
    ml_vote: int
    s1_active: bool
    s2_active: bool
    ml_active: bool
    s1_detail: str
    s2_detail: str
    ml_detail: str
    agreement: str                    # "STRONG_BULL"/"BULL"/"MIXED"/"BEAR"/"STRONG_BEAR"/"NEUTRAL"

    def votes_against(self, position_type: Optional[str]) -> tuple[bool, bool]:
        """
        Возвращает (s2_against, ml_against) для открытой позиции.
        position_type: "CALL" (лонг) или "PUT" (шорт).
        """
        if position_type == "CALL":
            # против лонга = медвежий голос
            return (self.s2_active and self.s2_vote < 0,
                    self.ml_active and self.ml_vote < 0)
        elif position_type == "PUT":
            return (self.s2_active and self.s2_vote > 0,
                    self.ml_active and self.ml_vote > 0)
        return (False, False)


# ────────────────────────────────────────────────────────────────────────────
# Преобразование сигналов в голоса
# ────────────────────────────────────────────────────────────────────────────

def signal_to_vote(action: str) -> int:
    a = str(action).upper()
    if a == "BUY":
        return 1
    if a == "SELL":
        return -1
    return 0


def ml_to_vote(long_prob: float, short_prob: float, min_edge: float) -> tuple[int, str]:
    """ML-вероятности → голос. edge = long_prob - short_prob."""
    edge = float(long_prob) - float(short_prob)
    if edge >= min_edge:
        return 1, f"L={long_prob:.2f} S={short_prob:.2f} edge=+{edge:.2f}"
    if edge <= -min_edge:
        return -1, f"L={long_prob:.2f} S={short_prob:.2f} edge={edge:.2f}"
    return 0, f"L={long_prob:.2f} S={short_prob:.2f} edge={edge:.2f} (нейтр.)"


def strategy2_regime_vote(row, min_agree: int = 3) -> tuple[int, str]:
    """
    Непрерывная оценка "режима" по логике стратегии #2 — голос каждый бар.

    В отличие от точечного входа #2 (нужны все 5 условий разом), здесь
    считаем бычьи/медвежьи признаки по 4 индикаторам и голосуем по большинству:
      - close vs VWAP
      - MACD vs сигнальная линия
      - Supertrend направление
      - RSI vs 50

    Сумма признаков от -4 до +4. Голос:
      +1 если бычьих признаков >= min_agree (по умолчанию 3 из 4)
      -1 если медвежьих признаков >= min_agree
       0 иначе (индикаторы противоречат друг другу)

    row — строка DataFrame (df.iloc[-1]) с колонками close, vwap, macd,
    macd_signal, supertrend_dir, rsi.
    Возвращает (vote, detail).
    """
    def _f(key):
        try:
            v = row[key]
            return float(v) if v is not None else None
        except (KeyError, TypeError, ValueError):
            return None

    close = _f("close")
    vwap_v = _f("vwap")
    macd_v = _f("macd")
    macd_sig = _f("macd_signal")
    st_dir = _f("supertrend_dir")
    rsi_v = _f("rsi")

    signals = []  # +1 бычий / -1 медвежий по каждому признаку
    parts = []

    if close is not None and vwap_v is not None:
        s = 1 if close > vwap_v else -1
        signals.append(s)
        parts.append(f"VWAP{'+' if s > 0 else '-'}")

    if macd_v is not None and macd_sig is not None:
        s = 1 if macd_v > macd_sig else -1
        signals.append(s)
        parts.append(f"MACD{'+' if s > 0 else '-'}")

    if st_dir is not None:
        s = 1 if st_dir > 0 else -1
        signals.append(s)
        parts.append(f"ST{'+' if s > 0 else '-'}")

    if rsi_v is not None:
        s = 1 if rsi_v > 50 else -1
        signals.append(s)
        parts.append(f"RSI{'+' if s > 0 else '-'}")

    if not signals:
        return 0, "нет данных"

    bulls = sum(1 for s in signals if s > 0)
    bears = sum(1 for s in signals if s < 0)
    detail = " ".join(parts) + f" ({bulls}↑/{bears}↓)"

    if bulls >= min_agree:
        return 1, detail
    if bears >= min_agree:
        return -1, detail
    return 0, detail


# ────────────────────────────────────────────────────────────────────────────
# Расчёт консенсуса
# ────────────────────────────────────────────────────────────────────────────

def _agreement_label(score: float) -> str:
    if score >= 2.0:
        return "STRONG_BULL"
    if score >= 1.0:
        return "BULL"
    if score <= -2.0:
        return "STRONG_BEAR"
    if score <= -1.0:
        return "BEAR"
    if score == 0.0:
        return "NEUTRAL"
    return "MIXED"


def compute_consensus(state: ConsensusState, cfg: ConsensusConfig) -> ConsensusResult:
    w = cfg.agree_window_bars
    cur = state.bar_counter

    s1_active = state.s1.is_active(cur, w)
    s2_active = state.s2.is_active(cur, w)
    ml_active = state.ml.is_active(cur, w)

    s1_v = state.s1.vote if s1_active else 0
    s2_v = state.s2.vote if s2_active else 0
    ml_v = state.ml.vote if ml_active else 0

    score = (cfg.weight_s1 * s1_v +
             cfg.weight_s2 * s2_v +
             cfg.weight_ml * ml_v)

    return ConsensusResult(
        score=round(score, 2),
        s1_vote=s1_v, s2_vote=s2_v, ml_vote=ml_v,
        s1_active=s1_active, s2_active=s2_active, ml_active=ml_active,
        s1_detail=state.s1.detail, s2_detail=state.s2.detail, ml_detail=state.ml.detail,
        agreement=_agreement_label(score),
    )


# ────────────────────────────────────────────────────────────────────────────
# Форматирование для Telegram
# ────────────────────────────────────────────────────────────────────────────

_AGREEMENT_RU = {
    "STRONG_BULL": "сильный бычий консенсус",
    "BULL": "бычий консенсус",
    "MIXED": "смешанный",
    "NEUTRAL": "нейтральный",
    "BEAR": "медвежий консенсус",
    "STRONG_BEAR": "сильный медвежий консенсус",
}


def _vote_symbol(vote: int, active: bool) -> str:
    if not active:
        return "○ нет"
    if vote > 0:
        return "🟢 +1 (BUY)"
    if vote < 0:
        return "🔴 −1 (SELL)"
    return "⚪ 0"


def format_consensus(res: ConsensusResult) -> str:
    """Разбивка голосов для сообщения при сигнале #1."""
    lines = [
        "📊 <b>Консенсус источников</b>",
        f"Итоговый балл: <b>{res.score:+.1f}</b>  ({_AGREEMENT_RU.get(res.agreement, res.agreement)})",
        "",
        f"#1 (BB+RSI):    {_vote_symbol(res.s1_vote, res.s1_active)}",
        f"#2 (MACD+VWAP): {_vote_symbol(res.s2_vote, res.s2_active)}",
        f"ML-блок:        {_vote_symbol(res.ml_vote, res.ml_active)}",
    ]
    if res.ml_active and res.ml_detail:
        lines.append(f"    ML: {res.ml_detail}")
    return "\n".join(lines)


def format_consensus_public(res: ConsensusResult) -> str:
    """
    Компактная разбивка для публичного канала.
    Источники обезличены: #1 / #2 / #3, ML назван "Predict".
    """
    lines = [
        "📊 <b>Консенсус источников</b>",
        f"Итоговый балл: <b>{res.score:+.1f}</b> ({_AGREEMENT_RU.get(res.agreement, res.agreement)})",
        "",
        f"#1:  {_vote_symbol(res.s1_vote, res.s1_active)}",
        f"#2:  {_vote_symbol(res.s2_vote, res.s2_active)}",
        f"#3:  {_vote_symbol(res.ml_vote, res.ml_active)}",
    ]
    if res.ml_active and res.ml_detail:
        lines.append(f"(Predict): {res.ml_detail}")
    return "\n".join(lines)


def format_conflict_warning(res: ConsensusResult, position_type: str, will_close: bool) -> str:
    """Предупреждение о конфликте против открытой позиции."""
    pos_ru = "CALL (лонг)" if position_type == "CALL" else "PUT (шорт)"
    head = "⚠️ <b>Конфликт консенсуса против позиции</b>"
    lines = [
        head,
        f"Открыта: {pos_ru}",
        f"Консенсус: {res.score:+.1f} ({_AGREEMENT_RU.get(res.agreement, res.agreement)})",
        "",
        f"#2: {_vote_symbol(res.s2_vote, res.s2_active)}",
        f"ML: {_vote_symbol(res.ml_vote, res.ml_active)}",
    ]
    if will_close:
        lines += ["", "🔻 <b>Оба источника единогласно против → досрочное закрытие</b>"]
    else:
        lines += ["", "ℹ️ Рекомендация: рассмотреть сокращение/закрытие вручную."]
    return "\n".join(lines)
