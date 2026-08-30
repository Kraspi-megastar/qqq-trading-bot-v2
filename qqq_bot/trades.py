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
    account_id: str = "signal"          # счёт исполнения (ffa/tfos/...) или "signal" (сигнальный уровень)
    entry_order_id: Optional[int] = None  # id ордера входа у брокера
    exit_order_id: Optional[int] = None   # id ордера выхода у брокера

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
        account_id: str = "signal",
        entry_order_id: Optional[int] = None,
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
            account_id=account_id,
            entry_order_id=entry_order_id,
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
        account_id: str = "signal",
        exit_order_id: Optional[int] = None,
    ) -> Optional[TradeRecord]:
        """Закрывает последнюю открытую сделку с данным тикером НА ДАННОМ СЧЁТЕ."""
        for i in range(len(self._trades) - 1, -1, -1):
            t = self._trades[i]
            if (t.ticker == ticker and t.status == "open"
                    and getattr(t, "account_id", "signal") == account_id):
                t.exit_price = exit_price
                t.exit_underlying = exit_underlying
                t.exit_ts = exit_ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                t.status = "closed"
                t.exit_order_id = exit_order_id
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

    def closed_trades(self, session_date: Optional[str] = None, limit: int = 20,
                      account_id: Optional[str] = None) -> list[TradeRecord]:
        result = [t for t in self._trades if t.status == "closed"]
        if session_date:
            result = [t for t in result if t.session_date == session_date]
        if account_id is not None:
            result = [t for t in result if getattr(t, "account_id", "signal") == account_id]
        return result[-limit:]

    def all_trades(self, session_date: Optional[str] = None,
                   account_id: Optional[str] = None) -> list[TradeRecord]:
        result = list(self._trades)
        if session_date:
            result = [t for t in result if t.session_date == session_date]
        if account_id is not None:
            result = [t for t in result if getattr(t, "account_id", "signal") == account_id]
        return result

    def accounts_seen(self) -> list[str]:
        """Список счетов, встречающихся в журнале (кроме сигнального уровня)."""
        seen = []
        for t in self._trades:
            acc = getattr(t, "account_id", "signal")
            if acc not in seen:
                seen.append(acc)
        return seen


def build_day_report(app, public: bool = False) -> str:
    """
    Дневной отчёт по опционным сделкам за сегодня.
    public=False → полный, С РАЗБИВКОЙ ПО СЧЕТАМ (для закрытого канала).
    public=True  → на 1 контракт, по СИГНАЛУ (для открытого канала).

    Важно: одна сделка журналируется и на сигнальном уровне ("signal"),
    и по каждому счёту (tfos/ostrov/...). Поэтому:
      - публичный отчёт берёт ТОЛЬКО signal-записи (одна на сигнал, без дублей);
      - закрытый отчёт группирует по РЕАЛЬНЫМ счетам (signal исключаем).
    """
    import html as _html
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if app.trade_journal is None:
        return "Журнал сделок недоступен."

    tz = ZoneInfo(app.cfg.display_tz)
    today_str = datetime.now(tz=tz).date().isoformat()
    journal = app.trade_journal

    if public:
        # ── ПУБЛИЧНЫЙ: только сигнальный уровень, на 1 контракт ──
        sig = [t for t in journal.closed_trades(session_date=today_str, limit=500,
                                                account_id="signal")
               if t.pnl() is not None]
        wins = [t for t in sig if (t.pnl() or 0) > 0]
        losses = [t for t in sig if (t.pnl() or 0) <= 0]
        wr = f"{len(wins)/len(sig)*100:.0f}%" if sig else "n/a"
        lines = [f"📋 <b>Итоги дня — {today_str}</b>", ""]
        lines.append(f"Сигналов: {len(sig)} | ✅ {len(wins)} / ❌ {len(losses)} | винрейт {wr}")
        if sig:
            total_per1 = 0.0; pct_list = []
            for t in sig:
                n = getattr(t, "contracts", 1) or 1
                total_per1 += (t.pnl() or 0.0) / n
                pct = t.pnl_pct() if hasattr(t, "pnl_pct") else None
                if pct is not None:
                    pct_list.append(pct)
            avg_pct = sum(pct_list) / len(pct_list) if pct_list else None
            sign_tot = "🟢" if total_per1 >= 0 else "🔴"
            lines.append("")
            lines.append(f"{sign_tot} <b>Итог за день (на 1 контракт): ${total_per1:+.0f}</b>")
            if avg_pct is not None:
                lines.append(f"Средняя доходность по сделкам: {avg_pct:+.1f}%")
            lines.append("")
            for t in sig:
                n = getattr(t, "contracts", 1) or 1
                per1 = (t.pnl() or 0.0) / n
                pct = t.pnl_pct() if hasattr(t, "pnl_pct") else None
                pct_str = f" ({pct:+.0f}%)" if pct is not None else ""
                sign = "🟢" if (t.pnl() or 0) > 0 else "🔴"
                lines.append(f"{sign} {t.option_type} ${per1:+.0f}{pct_str}")
        else:
            lines.append("")
            lines.append("Сигналов с закрытием за день нет.")
        return "\n".join(lines)

    # ── ЗАКРЫТЫЙ: разбивка по РЕАЛЬНЫМ счетам ──
    accounts = [a for a in journal.accounts_seen() if a != "signal"]
    lines = [f"📋 <b>Дневной отчёт QQQ — {today_str}</b>", ""]

    if not accounts:
        # реальных сделок по счетам ещё нет — покажем сигнальный уровень как справку
        sig = [t for t in journal.closed_trades(session_date=today_str, limit=500,
                                                account_id="signal") if t.pnl() is not None]
        if not sig:
            lines.append("Сделок за день нет.")
            return "\n".join(lines)
        total = sum(t.pnl() or 0 for t in sig)
        w = sum(1 for t in sig if (t.pnl() or 0) > 0)
        lines.append("<i>Реальных сделок по счетам нет — показан сигнальный уровень.</i>")
        lines.append(f"Сигналов: {len(sig)} | ✅ {w} / ❌ {len(sig)-w}")
        lines.append(f"Итог (сигнал, на позицию): ${total:+.2f}")
        return "\n".join(lines)

    grand_total = 0.0; grand_n = 0
    for acc in accounts:
        trades = [t for t in journal.closed_trades(session_date=today_str, limit=500,
                                                   account_id=acc) if t.pnl() is not None]
        if not trades:
            continue
        wins = [t for t in trades if (t.pnl() or 0) > 0]
        losses = [t for t in trades if (t.pnl() or 0) <= 0]
        total = sum(t.pnl() or 0 for t in trades)
        wr = f"{len(wins)/len(trades)*100:.0f}%" if trades else "n/a"
        grand_total += total; grand_n += len(trades)
        sign = "🟢" if total >= 0 else "🔴"
        lines.append(f"{sign} <b>[{acc}]</b> — {len(trades)} сд. | ✅{len(wins)}/❌{len(losses)} | {wr}")
        lines.append(f"    P/L счёта: <b>${total:+.2f}</b>")
        for t in trades:
            pnl = t.pnl() or 0.0
            pct = t.pnl_pct() if hasattr(t, "pnl_pct") else None
            pct_str = f" ({pct:+.0f}%)" if pct is not None else ""
            s2 = "🟢" if pnl > 0 else "🔴"
            n = getattr(t, "contracts", 1) or 1
            lines.append(f"    {s2} {t.option_type} {n}× → ${pnl:+.2f}{pct_str}")
        lines.append("")

    if grand_n > 0:
        gsign = "🟢" if grand_total >= 0 else "🔴"
        lines.append(f"{gsign} <b>ИТОГО по всем счетам: ${grand_total:+.2f}</b> ({grand_n} сделок)")
    else:
        lines.append("Реальных закрытых сделок по счетам за день нет.")
    return "\n".join(lines)



def build_day_report_public(app) -> str:
    return build_day_report(app, public=True)


def build_account_stats(app, session_date: Optional[str] = None) -> str:
    """
    Статистика P&L по КАЖДОМУ СЧЁТУ отдельно (реальные сделки).
    session_date=None → за всю историю; иначе за конкретный день.
    """
    import html as _html

    if app.trade_journal is None:
        return "Журнал сделок недоступен."

    journal = app.trade_journal
    # какие счета реально исполняли (исключаем сигнальный уровень)
    accounts = [a for a in journal.accounts_seen() if a != "signal"]

    period = session_date if session_date else "вся история"
    lines = [f"📊 <b>Статистика по счетам — {period}</b>", ""]

    if not accounts:
        lines.append("Реальных сделок по счетам пока нет.")
        lines.append("")
        lines.append("<i>Сделки исполняются при подтверждении ордера на счёте. "
                     "Как только пройдут реальные сделки — здесь появится разбивка "
                     "P&L по каждому счёту (FFA, TFOS и т.д.).</i>")
        return "\n".join(lines)

    grand_pnl = 0.0
    grand_n = 0
    for acc in accounts:
        closed = journal.closed_trades(session_date=session_date, limit=1000, account_id=acc)
        with_pnl = [t for t in closed if t.pnl() is not None]
        if not with_pnl:
            lines.append(f"<b>[{acc}]</b> — закрытых сделок нет")
            lines.append("")
            continue
        wins = [t for t in with_pnl if (t.pnl() or 0) > 0]
        losses = [t for t in with_pnl if (t.pnl() or 0) <= 0]
        total = sum(t.pnl() or 0.0 for t in with_pnl)
        wr = len(wins) / len(with_pnl) * 100 if with_pnl else 0
        avg = total / len(with_pnl) if with_pnl else 0
        avg_win = (sum(t.pnl() or 0 for t in wins) / len(wins)) if wins else 0
        avg_loss = (sum(t.pnl() or 0 for t in losses) / len(losses)) if losses else 0
        grand_pnl += total
        grand_n += len(with_pnl)
        sign = "🟢" if total >= 0 else "🔴"
        lines.append(f"{sign} <b>[{acc}]</b>")
        lines.append(f"  P&L: ${total:+.0f} | сделок {len(with_pnl)} | винрейт {wr:.0f}%")
        lines.append(f"  ✅ {len(wins)} (ср. ${avg_win:+.0f}) / ❌ {len(losses)} (ср. ${avg_loss:+.0f})")
        lines.append(f"  ср. сделка: ${avg:+.1f}")
        lines.append("")

    if len(accounts) > 1 and grand_n > 0:
        gsign = "🟢" if grand_pnl >= 0 else "🔴"
        lines.append(f"{gsign} <b>ИТОГО по всем счетам: ${grand_pnl:+.0f}</b> ({grand_n} сделок)")

    return "\n".join(lines)


def parse_tn_ticker(tn_ticker: str) -> Optional[dict]:
    """
    Разбирает тикер TraderNet '+QQQ.28AUG2026.C710' на компоненты.
    Возвращает {option_type, strike, expiry(date)} или None при неудаче.
    """
    from datetime import datetime as _dt
    try:
        s = tn_ticker.lstrip("+")
        parts = s.split(".")
        if len(parts) < 3:
            return None
        # parts[-1] = C710 / P705 ; parts[-2] = 28AUG2026
        cp = parts[-1]
        opt = "CALL" if cp[0].upper() == "C" else ("PUT" if cp[0].upper() == "P" else None)
        if opt is None:
            return None
        strike = float(cp[1:])
        expiry = _dt.strptime(parts[-2], "%d%b%Y").date()
        return {"option_type": opt, "strike": strike, "expiry": expiry}
    except Exception:
        return None
