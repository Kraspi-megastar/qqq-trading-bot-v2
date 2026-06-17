"""
state_store.py — персистентное хранение состояния бота между перезапусками.

Хранит:
  - strategy_id
  - Strategy2State (позиция, ATR-стоп, и т.д.)
  - OptionPosition (открытая опционная позиция для стратегии #1)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ────────────────────────────────────────────────────────────────────────────
# Datetime helpers
# ────────────────────────────────────────────────────────────────────────────

def _utc_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _date_iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d is not None else None


def _parse_date_iso(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────
# BotState — то что сериализуется в state.json
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class BotState:
    session_id: Optional[str] = None      # NY-date ISO (yyyy-mm-dd)
    strategy_id: int = 1

    # Strategy #2 runtime
    s2_position: str = "FLAT"
    s2_entry_price: Optional[float] = None
    s2_entry_ts: Optional[str] = None     # UTC ISO
    s2_atr_stop: Optional[float] = None

    # Опционная позиция (стратегия #1)
    opt_type: Optional[str] = None        # "CALL" | "PUT" | None
    opt_ticker: Optional[str] = None
    opt_tn_ticker: Optional[str] = None   # TraderNet format: QQQ.17JUN2026.C749
    opt_strike: Optional[float] = None
    opt_expiry: Optional[str] = None      # ISO date string
    opt_entry_underlying: Optional[float] = None
    opt_entry_date: Optional[str] = None  # ISO date string


# ────────────────────────────────────────────────────────────────────────────
# Файл
# ────────────────────────────────────────────────────────────────────────────

def state_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "state.json"


def load_state(cache_dir: Path) -> Optional[BotState]:
    p = state_path(cache_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        st = BotState(**{k: data.get(k) for k in BotState().__dict__.keys()})
        return st
    except Exception:
        return None


def save_state(cache_dir: Path, st: BotState) -> None:
    """Атомарная запись: tmp → replace."""
    p = state_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(asdict(st), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(p)


# ────────────────────────────────────────────────────────────────────────────
# Применение / сборка состояния
# ────────────────────────────────────────────────────────────────────────────

def apply_state_if_same_session(
    *,
    app: Any,
    current_session_id: str,
) -> bool:
    """
    Загружает state.json и применяет только если session_id совпадает.
    Возвращает True если применили.
    """
    from .signals import Strategy2State
    from .options import OptionPosition

    st = load_state(app.cfg.cache_dir)
    if st is None:
        return False
    if st.session_id != current_session_id:
        return False

    # strategy id
    if int(st.strategy_id) in (1, 2):
        app.strategy_id = int(st.strategy_id)

    # Strategy2 runtime
    app.strategy2 = Strategy2State(
        position=st.s2_position or "FLAT",
        entry_price=st.s2_entry_price,
        atr_stop=st.s2_atr_stop,
        entry_ts=_parse_utc_iso(st.s2_entry_ts),
    )

    # Опционная позиция
    if st.opt_type and st.opt_ticker and st.opt_strike is not None and st.opt_expiry:
        from .options import tradernet_option_ticker as _tn_fmt
        tn_tick = st.opt_tn_ticker or _tn_fmt(
            st.opt_type,
            st.opt_strike or 0.0,
            _parse_date_iso(st.opt_expiry) or date.today(),
        )
        app.option_position = OptionPosition(
            option_type=st.opt_type,
            ticker=st.opt_ticker,
            tn_ticker=tn_tick,
            strike=st.opt_strike,
            expiry=_parse_date_iso(st.opt_expiry) or date.today(),
            entry_underlying=st.opt_entry_underlying or 0.0,
            entry_date=_parse_date_iso(st.opt_entry_date) or date.today(),
        )
    else:
        app.option_position = None

    return True


def build_state_from_app(*, app: Any, session_id: str) -> BotState:
    pos = getattr(app, "option_position", None)
    return BotState(
        session_id=session_id,
        strategy_id=int(getattr(app, "strategy_id", 1)),
        s2_position=getattr(app.strategy2, "position", "FLAT"),
        s2_entry_price=getattr(app.strategy2, "entry_price", None),
        s2_entry_ts=_utc_iso(getattr(app.strategy2, "entry_ts", None)),
        s2_atr_stop=getattr(app.strategy2, "atr_stop", None),
        # опционная позиция
        opt_type=pos.option_type if pos else None,
        opt_ticker=pos.ticker if pos else None,
        opt_tn_ticker=pos.tn_ticker if pos else None,
        opt_strike=pos.strike if pos else None,
        opt_expiry=_date_iso(pos.expiry) if pos else None,
        opt_entry_underlying=pos.entry_underlying if pos else None,
        opt_entry_date=_date_iso(pos.entry_date) if pos else None,
    )
