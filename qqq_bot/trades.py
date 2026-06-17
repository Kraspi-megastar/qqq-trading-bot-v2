"""
trades.py — учёт опционных сделок с P&L.

Каждая сделка: открытие (OPEN) → закрытие (CLOSE).
Цены опционов берутся из TraderNet в момент сигнала.
Хранение: {CACHE_DIR}/trades.jsonl (одна JSON-строка на запись).

Формат тикера TraderNet: QQQ.17JUN2026.C749
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Запись о сделке
# (формат тикера TraderNet см. options.tradernet_option_ticker: +QQQ.31JUL2026.C732)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    trade_id: str                       # уникальный ID
    session_date: str                   # NY-дата открытия (YYYY-MM-DD)
    option_type: str                    # CALL | PUT
    ticker: str                         # QQQ.17JUN2026.C749
    strike: float
    expiry: str                         # ISO date string
    dte_at_entry: int                   # дней до экспирации при открытии
    entry_price: Optional[float]        # цена опциона при входе
    entry_underlying: float             # цена QQQ при входе
    entry_ts: str                       # UTC ISO
    exit_price: Optional[float] = None  # цена опциона при выходе
    exit_underlying: Optional[float] = None
    exit_ts: Optional[str] = None
    contracts: int = 1                  # количество контрактов
    status: str = "open"                # open | closed

    def pnl(self) -> Optional[float]:
        """P&L в долларах. 1 контракт = 100 акций."""
        if self.entry_price is None or self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) * 100 * self.contracts

    def pnl_pct(self) -> Optional[float]:
        """P&L в процентах."""
        if self.entry_price is None or self.entry_price == 0 or self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    def pnl_str(self) -> str:
        p = self.pnl()
        pct = self.pnl_pct()
        if p is None:
            return "n/a (n/a)"
        sign = "+" if p >= 0 else ""
        pct_str = f"{sign}{pct:.1f}%" if pct is not None else "n/a"
        return f"{sign}${p:.2f} ({pct_str})"

    def entry_ts_dt(self) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(self.entry_ts.replace("Z", "+00:00"))
        except Exception:
            return None

    def exit_ts_dt(self) -> Optional[datetime]:
        if not self.exit_ts:
            return None
        try:
            return datetime.fromisoformat(self.exit_ts.replace("Z", "+00:00"))
        except Exception:
            return None


# ────────────────────────────────────────────────────────────────────────────
# Журнал сделок
# ────────────────────────────────────────────────────────────────────────────

class TradeJournal:
    def __init__(self, cache_dir: Path) -> None:
        self._path = Path(cache_dir) / "trades.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._trades: list[TradeRecord] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        self._trades.append(TradeRecord(**d))
                    except Exception:
                        pass
        except Exception:
            pass

    def _append(self, trade: TradeRecord) -> None:
        """Дописывает/обновляет запись атомарно."""
        # Перезаписываем весь файл (сделок обычно немного — десятки/сотни)
        tmp = self._path.with_suffix(".jsonl.tmp")
        # Обновляем в памяти
        for i, t in enumerate(self._trades):
            if t.trade_id == trade.trade_id:
                self._trades[i] = trade
                break
        else:
            self._trades.append(trade)
        # Записываем
        with open(tmp, "w", encoding="utf-8") as f:
            for t in self._trades:
                f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
        tmp.replace(self._path)

    # ── Публичный API ────────────────────────────────────────────────────────

    def open_trade(
        self,
        *,
        session_date: str,
        option_type: str,
        ticker: str,
        strike: float,
        expiry: date,
        dte_at_entry: int,
        entry_price: Optional[float],
        entry_underlying: float,
        entry_ts: datetime,
        contracts: int = 1,
    ) -> TradeRecord:
        trade = TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            session_date=session_date,
            option_type=option_type,
            ticker=ticker,
            strike=strike,
            expiry=expiry.isoformat(),
            dte_at_entry=dte_at_entry,
            entry_price=entry_price,
            entry_underlying=entry_underlying,
            entry_ts=entry_ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            contracts=contracts,
            status="open",
        )
        self._append(trade)
        return trade

    def close_trade(
        self,
        *,
        ticker: str,
        exit_price: Optional[float],
        exit_underlying: float,
        exit_ts: datetime,
    ) -> Optional[TradeRecord]:
        """Закрывает последнюю открытую сделку с данным тикером."""
        for i in range(len(self._trades) - 1, -1, -1):
            t = self._trades[i]
            if t.ticker == ticker and t.status == "open":
                t.exit_price = exit_price
                t.exit_underlying = exit_underlying
                t.exit_ts = exit_ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                t.status = "closed"
                self._append(t)
                return t
        return None

    def open_trade_for_ticker(self, ticker: str) -> Optional[TradeRecord]:
        for t in reversed(self._trades):
            if t.ticker == ticker and t.status == "open":
                return t
        return None

    def any_open(self) -> Optional[TradeRecord]:
        for t in reversed(self._trades):
            if t.status == "open":
                return t
        return None

    def closed_trades(self, session_date: Optional[str] = None, limit: int = 20) -> list[TradeRecord]:
        result = [t for t in self._trades if t.status == "closed"]
        if session_date:
            result = [t for t in result if t.session_date == session_date]
        return result[-limit:]

    def all_trades(self, session_date: Optional[str] = None) -> list[TradeRecord]:
        if session_date:
            return [t for t in self._trades if t.session_date == session_date]
        return list(self._trades)
