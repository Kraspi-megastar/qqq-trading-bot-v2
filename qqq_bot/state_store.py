from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass
class BotState:
    session_id: str | None = None          # NY-date ISO (yyyy-mm-dd)
    strategy_id: int = 1

    # Strategy #2 runtime
    s2_position: str = "FLAT"              # "FLAT" | "LONG"
    s2_entry_price: float | None = None
    s2_entry_ts: str | None = None         # UTC ISO
    s2_atr_stop: float | None = None


def state_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "state.json"


def load_state(cache_dir: Path) -> BotState | None:
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
    p = state_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(st), ensure_ascii=False, indent=2), encoding="utf-8")


def apply_state_if_same_session(
    *,
    app: Any,
    current_session_id: str,
) -> bool:
    """
    Загружаем state.json и применяем, только если session_id совпадает (чтобы не переносить позицию на новый день).
    Возвращает True если применили.
    """
    st = load_state(app.cfg.cache_dir)
    if st is None:
        return False
    if st.session_id != current_session_id:
        return False

    # strategy
    if int(st.strategy_id) in (1, 2):
        app.strategy_id = int(st.strategy_id)

    # strategy2 runtime
    app.strategy2.position = st.s2_position or "FLAT"
    app.strategy2.entry_price = st.s2_entry_price
    app.strategy2.entry_ts = _parse_utc_iso(st.s2_entry_ts)
    app.strategy2.atr_stop = st.s2_atr_stop

    return True


def build_state_from_app(
    *,
    app: Any,
    session_id: str,
) -> BotState:
    return BotState(
        session_id=session_id,
        strategy_id=int(getattr(app, "strategy_id", 1)),
        s2_position=getattr(app.strategy2, "position", "FLAT"),
        s2_entry_price=getattr(app.strategy2, "entry_price", None),
        s2_entry_ts=_utc_iso(getattr(app.strategy2, "entry_ts", None)),
        s2_atr_stop=getattr(app.strategy2, "atr_stop", None),
    )
