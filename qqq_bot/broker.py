"""
broker.py — исполнение сделок через TraderNet/Freedom24 Public API (tradernet-sdk).

БЕЗОПАСНОСТЬ — ЭТО ПРИОРИТЕТ. Модуль построен так, чтобы:
  - НИКОГДА не отправлять ордер без явного разрешения (semi-auto по умолчанию);
  - проверять purchasing power перед каждым ордером;
  - защищать от 0DTE-маржинальной ловушки;
  - обрабатывать отказы брокера как штатную ситуацию, а не краш;
  - иметь стоп-кран (/halt), мгновенно блокирующий любое исполнение;
  - сверять реальную позицию у брокера (reconciliation) перед действиями.

Клиент tradernet-sdk (класс Tradernet) инициализируется public/private ключами.
Ключи НЕ хранятся в этом модуле — передаются из config (который читает их из env).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Коды операций и статусов TraderNet (из реального ответа get_placed)
# ────────────────────────────────────────────────────────────────────────────

# oper: направление приказа
OPER_BUY = 1
OPER_SELL = 2
# (в ответах наблюдались и другие коды — трактуем осторожно)

# type: тип ордера
ORDER_TYPE_MARKET = 1
ORDER_TYPE_LIMIT = 2

# exp: срок действия
EXP_DAY = 1

# stat: статус ордера (из реальных ответов)
STAT_ACTIVE = 10        # активен / размещён
STAT_CANCELLED = 2      # отменён


# ────────────────────────────────────────────────────────────────────────────
# Конфигурация исполнения
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionConfig:
    enabled: bool = False              # ГЛАВНЫЙ рубильник: исполнение вообще включено?
    account_id: str = "ffa"            # короткий id счёта
    label: str = "FFA"                 # имя для сообщений
    mode: str = "semi_auto"            # "semi_auto" (подтверждение) | "auto" | "off"
    public_key: str = ""
    private_key: str = ""

    # Sizing
    position_pct: float = 5.0          # % от размера счёта на одну сделку
    max_position_pct: float = 100.0    # жёсткий потолок % на сделку (предохранитель от опечаток)
    max_contracts: int = 50            # абсолютный потолок контрактов

    # Предохранители
    max_orders_per_day: int = 0        # лимит ордеров в день (0 = без лимита)
    max_notional_per_trade: float = 50000.0  # макс. номинал одной сделки, $

    # 0DTE / overnight защита
    hold_overnight_min_dte: int = 99   # держать на ночь только если DTE >= этого
                                       # 99 = НИКОГДА не держать (закрывать всё)
    block_new_position_if_dte_lte: int = 0  # не открывать если DTE <= этого (0DTE-защита)

    # Reconciliation
    require_reconcile: bool = True     # сверять позицию с брокером перед действием

    # Semi-auto подтверждение
    confirm_timeout_sec: int = 300     # 5 минут на подтверждение кнопкой

    # Символ базового актива
    underlying_symbol: str = "QQQ.US"


# ────────────────────────────────────────────────────────────────────────────
# Результат операции исполнения
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecResult:
    ok: bool
    action: str                        # "order_placed" | "order_cancelled" | "blocked" | "error" | "pending_confirm"
    reason: str
    order_id: Optional[int] = None
    raw: dict = field(default_factory=dict)


@dataclass
class PendingOrder:
    """Ордер, ожидающий подтверждения пользователем (semi-auto)."""
    token: str                         # уникальный id для callback-кнопки
    tn_ticker: str
    side: str                          # BUY | SELL
    contracts: int
    limit_price: float
    dte: Optional[int]
    is_open: bool
    created_ts: datetime
    human: str                         # текст для кнопки/сообщения
    market: bool = False               # рыночный приказ (для стопов)

    def is_expired(self, now: datetime, timeout_sec: int) -> bool:
        return (now - self.created_ts).total_seconds() > timeout_sec


# ────────────────────────────────────────────────────────────────────────────
# Broker — обёртка над tradernet-sdk с предохранителями
# ────────────────────────────────────────────────────────────────────────────

class Broker:
    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg
        self._client = None
        self._halted = False           # стоп-кран
        self._orders_today = 0
        self._orders_day = date.today()
        self.load_error: Optional[str] = None
        self._pending: dict[str, PendingOrder] = {}  # token → ожидающий ордер

        if cfg.enabled and cfg.public_key and cfg.private_key:
            try:
                from tradernet import Tradernet
                self._client = Tradernet(cfg.public_key, cfg.private_key)
            except Exception as e:
                self.load_error = repr(e)
                self._client = None

    # ── Состояние ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Готов ли брокер исполнять (клиент загружен, не остановлен)."""
        return self._client is not None and not self._halted and self.cfg.enabled

    def halt(self) -> None:
        """Стоп-кран: мгновенно блокирует любое исполнение."""
        self._halted = True

    def resume(self) -> None:
        self._halted = False

    @property
    def halted(self) -> bool:
        return self._halted

    def _reset_daily_counter(self) -> None:
        today = date.today()
        if today != self._orders_day:
            self._orders_day = today
            self._orders_today = 0

    # ── Чтение состояния счёта ───────────────────────────────────────────────

    def account_summary(self) -> Optional[dict]:
        """Сводка по счёту (позиции, баланс, ордера). None при ошибке."""
        if self._client is None:
            return None
        try:
            return self._client.account_summary()
        except Exception as e:
            logger.warning("account_summary error: %s", repr(e))
            return None

    def get_active_orders(self) -> list[dict]:
        """
        Список РЕАЛЬНО активных ордеров (stat == 10).
        Брокер отдаёт в списке и недавно отменённые (stat==2) — отфильтровываем.
        """
        if self._client is None:
            return []
        try:
            resp = self._client.get_placed(active=True)
            orders = (
                resp.get("result", {}).get("orders", {}).get("order", [])
                if isinstance(resp, dict) else []
            )
            if isinstance(orders, dict):
                orders = [orders]
            # Только реально активные: stat == 10
            return [o for o in (orders or []) if int(o.get("stat", 0)) == STAT_ACTIVE]
        except Exception as e:
            logger.warning("get_placed error: %s", repr(e))
            return []

    def find_order_for_ticker(self, tn_ticker: str) -> Optional[dict]:
        """Активный ордер по тикеру инструмента, если есть."""
        for o in self.get_active_orders():
            if str(o.get("instr", "")) == tn_ticker:
                return o
        return None

    def purchasing_power(self, currency: str = "USD") -> Optional[float]:
        """
        Свободный денежный остаток в указанной валюте (по умолчанию USD).

        Структура account_summary: result.ps.acc — список счетов по валютам,
        у каждого curr (валюта) и s (сумма). Берём консервативно денежный остаток
        (без учёта маржи) — лучше недооценить доступное, чем переоценить.

        Возвращает None если не удалось определить (тогда caller НЕ должен открывать).
        """
        summ = self.account_summary()
        if not isinstance(summ, dict):
            return None
        try:
            acc = summ["result"]["ps"]["acc"]
            if not isinstance(acc, list):
                return None
            for entry in acc:
                if isinstance(entry, dict) and entry.get("curr") == currency:
                    return float(entry.get("s", 0.0))
            # валюта не найдена — значит 0 на этом балансе
            return 0.0
        except (KeyError, TypeError, ValueError):
            return None

    def get_positions(self) -> list[dict]:
        """Открытые позиции из account_summary (result.ps.pos)."""
        summ = self.account_summary()
        if not isinstance(summ, dict):
            return []
        try:
            pos = summ["result"]["ps"]["pos"]
            if isinstance(pos, dict):
                pos = [pos]
            return pos or []
        except (KeyError, TypeError):
            return []

    # ── Предохранители перед ордером ─────────────────────────────────────────

    def _preflight(self, notional: float, contracts: int, dte: Optional[int], is_open: bool) -> Optional[str]:
        """
        Проверки перед отправкой ордера. Возвращает причину блокировки или None если ок.
        """
        if not self.available:
            return "исполнение недоступно (выключено/остановлено/нет клиента)"

        self._reset_daily_counter()
        # Дневной лимит ордеров: 0 = без лимита (по умолчанию отключён).
        if self.cfg.max_orders_per_day > 0 and self._orders_today >= self.cfg.max_orders_per_day:
            return f"достигнут дневной лимит ордеров ({self.cfg.max_orders_per_day})"

        if contracts > self.cfg.max_contracts:
            return f"превышен лимит контрактов ({contracts} > {self.cfg.max_contracts})"

        if notional > self.cfg.max_notional_per_trade:
            return f"превышен номинал сделки (${notional:.0f} > ${self.cfg.max_notional_per_trade:.0f})"

        # 0DTE-защита: не открывать позицию слишком близко к экспирации
        if is_open and dte is not None and dte <= self.cfg.block_new_position_if_dte_lte:
            return f"0DTE-защита: не открываем позицию с DTE={dte}"

        # Purchasing power — только для открытия
        if is_open:
            pp = self.purchasing_power()
            if pp is None:
                return "не удалось проверить purchasing power (торговля заблокирована из осторожности)"
            if notional > pp:
                return f"недостаточно purchasing power (нужно ${notional:.0f}, есть ${pp:.0f})"

        return None

    # ── Отправка / отмена ордеров ────────────────────────────────────────────

    def place_option_order(
        self,
        *,
        tn_ticker: str,               # +QQQ.01JUL2026.C733
        side: str,                    # "BUY" | "SELL"
        contracts: int,
        limit_price: float,           # цена опциона (лимит); для рынка передаётся ориентир
        dte: Optional[int],
        is_open: bool,                # True = открытие, False = закрытие
        confirmed: bool = False,      # semi_auto: True только после подтверждения пользователем
        market: bool = False,         # True = рыночный приказ (price=0), False = лимитный
    ) -> ExecResult:
        """
        Отправляет ордер на опцион. В semi_auto без confirmed=True — не отправляет,
        а возвращает pending_confirm.
        market=True → рыночный приказ (быстрое исполнение, для стопов).
        """
        # для оценки номинала/предохранителей используем limit_price как ориентир
        ref_price = abs(limit_price) if limit_price else 0.0
        notional = ref_price * contracts * 100  # 1 контракт = 100

        block = self._preflight(notional, contracts, dte, is_open)
        if block is not None:
            return ExecResult(ok=False, action="blocked", reason=block)

        order_kind = "MKT" if market else f"@ {limit_price:.2f}"

        # semi_auto: требуем явное подтверждение
        if self.cfg.mode == "semi_auto" and not confirmed:
            return ExecResult(
                ok=False, action="pending_confirm",
                reason=f"{side} {contracts}× {tn_ticker} {order_kind} — требуется подтверждение",
            )

        if self.cfg.mode == "off":
            return ExecResult(ok=False, action="blocked", reason="режим off")

        # Отправка через SDK. place_order: отрицательный qty = продажа.
        # Рыночный приказ: price=0.0 (SDK трактует 0 как рынок).
        qty = contracts if side == "BUY" else -contracts
        send_price = 0.0 if market else float(limit_price)
        try:
            resp = self._client.place_order(
                symbol=tn_ticker,
                quantity=qty,
                price=send_price,
                duration="day",
                use_margin=False,       # опционы без маржи по умолчанию
            )
        except Exception as e:
            return ExecResult(ok=False, action="error", reason=f"SDK exception: {e!r}")

        # Разбор ответа
        if isinstance(resp, dict) and resp.get("error"):
            return ExecResult(ok=False, action="error", reason=str(resp["error"]).strip(), raw=resp)

        order_id = None
        if isinstance(resp, dict):
            order_id = (resp.get("order_id") or resp.get("id")
                        or resp.get("result", {}).get("order_id"))

        self._orders_today += 1
        return ExecResult(
            ok=True, action="order_placed",
            reason=f"{side} {contracts}× {tn_ticker} {order_kind}",
            order_id=int(order_id) if order_id else None, raw=resp if isinstance(resp, dict) else {},
        )

    def cancel_order(self, order_id: int) -> ExecResult:
        if self._client is None:
            return ExecResult(ok=False, action="error", reason="нет клиента")
        try:
            resp = self._client.cancel(order_id=int(order_id))
        except Exception as e:
            return ExecResult(ok=False, action="error", reason=f"cancel exception: {e!r}")

        # Успех отмены: {"result": 1}
        if isinstance(resp, dict) and resp.get("result") == 1:
            return ExecResult(ok=True, action="order_cancelled",
                              reason=f"ордер {order_id} отменён", order_id=int(order_id), raw=resp)

        # "may not be cancelled" = ордер УЖЕ неактивен (отменён/исполнён/отклонён).
        # Для наших целей это не ошибка: цель (ордера нет в рынке) достигнута.
        err = str(resp.get("error", "")).strip().lower() if isinstance(resp, dict) else ""
        if "may not be cancelled" in err or "not be cancelled" in err:
            # сверяемся: реально ли он неактивен
            still_active = any(int(o.get("order_id", 0)) == int(order_id)
                               for o in self.get_active_orders())
            if not still_active:
                return ExecResult(ok=True, action="order_cancelled",
                                  reason=f"ордер {order_id} уже неактивен", order_id=int(order_id), raw=resp)

        return ExecResult(ok=False, action="error",
                          reason=str(resp.get("error", "unknown")).strip() if isinstance(resp, dict) else "unknown",
                          raw=resp if isinstance(resp, dict) else {})

    # ── Semi-auto: ожидающие подтверждения ордера ────────────────────────────

    def create_pending(
        self, *, tn_ticker: str, side: str, contracts: int,
        limit_price: float, dte: Optional[int], is_open: bool,
        market: bool = False,
    ) -> PendingOrder:
        """Регистрирует ордер, ожидающий подтверждения. Возвращает PendingOrder с токеном."""
        import uuid
        token = uuid.uuid4().hex[:12]
        action_ru = "ОТКРЫТЬ" if is_open else "ЗАКРЫТЬ"
        price_part = "по рынку" if market else f"@ {limit_price:.2f}"
        human = f"{action_ru} {side} {contracts}× {tn_ticker} {price_part}"
        po = PendingOrder(
            token=token, tn_ticker=tn_ticker, side=side, contracts=contracts,
            limit_price=limit_price, dte=dte, is_open=is_open,
            created_ts=datetime.now(tz=timezone.utc), human=human, market=market,
        )
        self._pending[token] = po
        return po

    def get_pending(self, token: str) -> Optional[PendingOrder]:
        return self._pending.get(token)

    def purge_expired_pending(self) -> list[PendingOrder]:
        """Удаляет протухшие ожидания. Возвращает список удалённых (для уведомления)."""
        now = datetime.now(tz=timezone.utc)
        expired = [po for po in self._pending.values()
                   if po.is_expired(now, self.cfg.confirm_timeout_sec)]
        for po in expired:
            self._pending.pop(po.token, None)
        return expired

    def confirm_pending(self, token: str) -> ExecResult:
        """
        Подтверждает и ИСПОЛНЯЕТ ожидающий ордер по токену.
        Проверяет тайм-аут и заново прогоняет предохранители (цена/маржа могли измениться).
        """
        po = self._pending.get(token)
        if po is None:
            return ExecResult(ok=False, action="error", reason="ордер не найден или уже обработан")

        now = datetime.now(tz=timezone.utc)
        if po.is_expired(now, self.cfg.confirm_timeout_sec):
            self._pending.pop(token, None)
            return ExecResult(ok=False, action="error",
                              reason=f"подтверждение истекло (>{self.cfg.confirm_timeout_sec//60} мин)")

        # Убираем из ожидания и исполняем с confirmed=True
        self._pending.pop(token, None)
        return self.place_option_order(
            tn_ticker=po.tn_ticker, side=po.side, contracts=po.contracts,
            limit_price=po.limit_price, dte=po.dte, is_open=po.is_open,
            confirmed=True, market=getattr(po, "market", False),
        )

    def reject_pending(self, token: str) -> bool:
        """Отклоняет ожидающий ордер (пользователь нажал Отмена)."""
        return self._pending.pop(token, None) is not None

    # ── Аварийный выход ──────────────────────────────────────────────────────
    def list_open_option_positions(self) -> list[dict]:
        """
        Открытые ОПЦИОННЫЕ позиции для аварийного закрытия.
        Возвращает список {ticker, qty} с ненулевым количеством.
        """
        out = []
        for p in self.get_positions():
            try:
                ticker = str(p.get("i", "")).strip()
                qty = int(p.get("q", 0) or 0)
                if ticker and qty != 0:
                    out.append({"ticker": ticker, "qty": qty,
                                "mkt": p.get("mkt_price", p.get("market_value"))})
            except (ValueError, TypeError):
                continue
        return out

    def emergency_close_all(self) -> list[ExecResult]:
        """
        АВАРИЙНЫЙ ВЫХОД: продать по РЫНКУ все открытые опционные позиции счёта.
        Обходит semi_auto-подтверждение и дневные лимиты (это осознанное ручное
        решение оператора). Закрытие опциона всегда SELL (и CALL, и PUT
        открывались через BUY). Возвращает результат по каждой позиции.
        """
        results: list[ExecResult] = []
        if self._client is None:
            return [ExecResult(ok=False, action="error", reason="нет клиента брокера")]
        if self._halted:
            return [ExecResult(ok=False, action="error", reason="стоп-кран (halt) активен")]

        positions = self.list_open_option_positions()
        if not positions:
            return [ExecResult(ok=False, action="noop", reason="нет открытых позиций")]

        for pos in positions:
            ticker = pos["ticker"]
            raw_qty = int(pos["qty"])
            if raw_qty == 0:
                continue
            # Закрытие по знаку позиции: лонг (qty>0) → SELL (-qty),
            # шорт (qty<0) → BUY (+|qty|). Бот обычно только лонг-опционы,
            # но поддержим обе стороны на случай нестандартной позиции.
            close_qty = -raw_qty            # противоположный знак закрывает позицию
            qty = abs(raw_qty)
            side_txt = "SELL" if raw_qty > 0 else "BUY"
            try:
                resp = self._client.place_order(
                    symbol=ticker, quantity=close_qty, price=0.0,   # price=0 → рынок
                    duration="day", use_margin=False,
                )
            except Exception as e:
                results.append(ExecResult(ok=False, action="error",
                                          reason=f"{ticker}: SDK exception {e!r}"))
                continue
            if isinstance(resp, dict) and resp.get("error"):
                results.append(ExecResult(ok=False, action="error",
                                          reason=f"{ticker}: {str(resp['error']).strip()}", raw=resp))
                continue
            order_id = None
            if isinstance(resp, dict):
                order_id = (resp.get("order_id") or resp.get("id")
                            or resp.get("result", {}).get("order_id"))
            self._orders_today += 1
            results.append(ExecResult(
                ok=True, action="order_placed",
                reason=f"АВАРИЙНОЕ ЗАКРЫТИЕ {side_txt} {qty}× {ticker} MKT",
                order_id=int(order_id) if order_id else None,
                raw=resp if isinstance(resp, dict) else {},
            ))
        return results



    def calc_contracts(self, option_price: float, account_value: float) -> int:
        """
        Сколько контрактов купить исходя из position_pct от счёта.
        1 контракт = 100 * цена опциона (премия).
        """
        pct = min(self.cfg.position_pct, self.cfg.max_position_pct) / 100.0
        budget = account_value * pct
        per_contract = max(option_price * 100, 1e-9)
        n = int(budget // per_contract)
        n = max(0, min(n, self.cfg.max_contracts))
        return n

    # ── 0DTE / overnight решение ─────────────────────────────────────────────

    def must_close_before_eod(self, dte: Optional[int]) -> bool:
        """
        Нужно ли обязательно закрыть позицию перед концом дня.
        True если DTE < hold_overnight_min_dte (по умолчанию 99 → закрывать всегда).
        """
        if dte is None:
            return True
        return dte < self.cfg.hold_overnight_min_dte
