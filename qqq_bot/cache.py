from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterable
from collections import deque
from datetime import datetime, timezone

from .models import Bar


def _safe_name(s: str) -> str:
    # "QQQ.US" ok, но на всякий случай чистим запрещённые символы для Windows
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".", "@") else "_" for ch in s)


def cache_file_path(cache_dir: Path, symbol: str, timeframe_minutes: int) -> Path:
    return (cache_dir / f"bars_{_safe_name(symbol)}_{int(timeframe_minutes)}m.json").resolve()


def _dt_to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dt_from_iso_z(s: str) -> datetime:
    # поддержка "Z" и "+00:00"
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class Stats:
    ticks: int = 0
    bars_real: int = 0
    bars_synth: int = 0

    signals_buy: int = 0
    signals_sell: int = 0

    last_signal: str | None = None
    last_signal_ts: datetime | None = None           # timestamp бара (UTC)
    last_signal_type: str | None = None              # OPEN_LONG/CLOSE_LONG/OPEN_SHORT/CLOSE_SHORT
    last_signal_mode: str | None = None              # breakout/pullback/exit
    last_signal_price: float | None = None
    last_signal_sent_at: datetime | None = None      # wall-clock UTC (когда отправили)

    # последние сигналы (пересчитанные по истории/кешу или полученные в real-time)
    # список из (action, bar_ts_utc), максимум 3 элемента
    signal_history: list[tuple[str, datetime]] = field(default_factory=list)

    cooldown_skips: int = 0
    last_error: str | None = None

    # диагностика бустрапа/кеша
    bootstrap_attempt: str | None = None
    bootstrap_result: str | None = None
    cache_file: str | None = None
    cache_load: str | None = None
    cache_save: str | None = None

    # сессия/тайминг
    session_state: str | None = None        # "OPEN"/"CLOSED"
    now_utc: datetime | None = None


@dataclass
class BarCache:
    timeframe_minutes: int
    maxlen: int
    bars: Deque[Bar] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not self.bars:
            self.bars = deque(maxlen=self.maxlen)
        else:
            self.bars = deque(self.bars, maxlen=self.maxlen)

    def __len__(self) -> int:
        return len(self.bars)

    def clear(self) -> None:
        self.bars.clear()

    def to_list(self) -> list[Bar]:
        return list(self.bars)

    def last(self) -> Bar | None:
        return self.bars[-1] if self.bars else None

    def append(self, bar: Bar) -> None:
        self.bars.append(bar)

    def extend(self, bars: Iterable[Bar]) -> None:
        for b in bars:
            self.bars.append(b)

    def replace_last(self, bar: Bar) -> None:
        if not self.bars:
            self.bars.append(bar)
        else:
            self.bars[-1] = bar

    # ------------------------
    # Persistence (JSON file)
    # ------------------------
    def load_from_file(self, path: Path) -> int:
        if not path.exists():
            return 0
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            return 0

        loaded: list[Bar] = []
        for it in data:
            if not isinstance(it, dict):
                continue
            try:
                ts = _dt_from_iso_z(str(it["ts"]))
                loaded.append(
                    Bar(
                        ts=ts,
                        open=float(it["open"]),
                        high=float(it["high"]),
                        low=float(it["low"]),
                        close=float(it["close"]),
                        volume=float(it.get("volume", 0.0)),
                        synthetic=bool(it.get("synthetic", False)),
                    )
                )
            except Exception:
                continue

        loaded.sort(key=lambda b: b.ts)

        self.clear()
        self.extend(loaded[-self.maxlen:])
        return len(self.bars)

    def save_to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for b in self.to_list():
            data.append(
                {
                    "ts": _dt_to_iso_z(b.ts),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "synthetic": bool(b.synthetic),
                }
            )

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
