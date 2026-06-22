from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp
import asyncio
from zoneinfo import ZoneInfo

from .models import Quote, Bar
from .utils_time import safe_float, floor_time


_TN_TZ = ZoneInfo("Europe/Moscow")  # сервер трактует строки DD.MM.YYYY HH:MM как MSK (по диагностике)


def _dt_to_tn_str(dt_utc: datetime) -> str:
    """
    TraderNet getHloc ожидает строку 'DD.MM.YYYY HH:MM' без TZ.
    Практически сервер интерпретирует её как MSK (UTC+3),
    поэтому конвертируем из UTC -> MSK и форматируем.
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_msk = dt_utc.astimezone(_TN_TZ)
    return dt_msk.strftime("%d.%m.%Y %H:%M")


@dataclass
class TraderNetClient:
    api_url: str               # e.g. https://tradernet.ru/api/
    quotes_url: str            # e.g. https://tradernet.ru/securities/export
    session: aiohttp.ClientSession

    # optional auth
    sid: str | None = None

    # network
    timeout_seconds: int = 20
    alt_api_urls: tuple[str, ...] = field(
        default_factory=lambda: (
            "https://tradernet.ru/api/",
            "https://tradernet.com/api/",
            "https://tradernet.global/api/",
        )
    )

    async def get_quote_ltp(self, symbol: str) -> float:
        """
        Реалтайм: берем последнюю цену (ltp) через securities/export.
        """
        params = {"params": "ltp", "tickers": symbol}
        async with self.session.get(self.quotes_url, params=params, timeout=10) as r:
            r.raise_for_status()
            txt = await r.text()
            data = json.loads(txt)

        if not isinstance(data, list) or not data or "ltp" not in data[0]:
            raise RuntimeError(f"Quote missing ltp: {txt[:200]}")
        return float(data[0]["ltp"])

    async def get_option_quote(self, option_ticker: str) -> dict | None:
        """
        Запрашивает котировку опциона (ltp, bid, ask, delta если есть).
        Возвращает dict или None если опцион не торгуется / не существует.

        option_ticker — формат TraderNet, напр. QQQ.19JUN2026.C480
        """
        params = {"params": "ltp,bid,ask,delta,oi,vol", "tickers": option_ticker}
        try:
            async with self.session.get(self.quotes_url, params=params, timeout=10) as r:
                if r.status != 200:
                    return None
                data = json.loads(await r.text())
        except Exception:
            return None

        if not isinstance(data, list) or not data:
            return None

        row = data[0]
        # Опцион считается существующим только если есть хоть какая-то цена
        ltp = safe_float(row.get("ltp"))
        bid = safe_float(row.get("bid"))
        ask = safe_float(row.get("ask"))
        if ltp is None and bid is None and ask is None:
            return None

        return {
            "ticker": option_ticker,
            "ltp": ltp,
            "bid": bid,
            "ask": ask,
            "delta": safe_float(row.get("delta")),
            "oi": safe_float(row.get("oi")),
            "vol": safe_float(row.get("vol")),
        }

    async def get_option_quote_raw(self, option_ticker: str) -> str:
        """Диагностика: возвращает сырой текст ответа quotes_url для опционного тикера."""
        params = {"params": "ltp,bid,ask,delta,oi,vol", "tickers": option_ticker}
        try:
            async with self.session.get(self.quotes_url, params=params, timeout=10) as r:
                return f"HTTP {r.status}\n" + (await r.text())[:800]
        except Exception as e:
            return f"ERROR: {e!r}"

    async def option_exists(self, option_ticker: str) -> bool:
        """True если по опциону есть рыночная котировка (значит он реально торгуется)."""
        q = await self.get_option_quote(option_ticker)
        return q is not None

    async def get_option_chain(self, underlying: str, option_type: str) -> list[dict]:
        """
        Запрашивает опционную цепочку через API getOptionChain.
        Возвращает список контрактов с полями strike, expiry, ticker, delta (если есть).
        Пустой список если API недоступен.
        """
        payload = {
            "cmd": "getOptionChain",
            "params": {"id": underlying, "type": option_type.lower()},
        }
        if self.sid:
            payload["SID"] = self.sid

        try:
            async with self.session.post(
                self.api_url,
                data={"q": json.dumps(payload, ensure_ascii=False)},
                timeout=self.timeout_seconds,
                headers={"User-Agent": "qqq_trading_bot/3.0"},
                cookies={"SID": self.sid} if self.sid else None,
            ) as r:
                if r.status != 200:
                    return []
                data = json.loads(await r.text())
        except Exception:
            return []

        contracts = None
        if isinstance(data, list):
            contracts = data
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list) and v:
                    contracts = v
                    break
        if not contracts:
            return []

        result = []
        for c in contracts:
            if not isinstance(c, dict):
                continue
            result.append({
                "strike": safe_float(c.get("strike") or c.get("exercise_price")),
                "expiry": c.get("expiry") or c.get("expiration") or c.get("exp_date"),
                "ticker": c.get("ticker") or c.get("symbol") or c.get("id"),
                "delta": safe_float(c.get("delta")),
            })
        return result

    async def get_hloc(
        self,
        symbol: str,
        timeframe_minutes: int,
        date_from_utc: datetime,
        date_to_utc: datetime,
        count: int,
        interval_mode: str = "ClosedRay",
        user_id: int | None = None,
    ) -> list[Bar]:
        """
        История getHloc (candlesticks).
        В документации TraderNet методы часто требуют SID (авторизация) —
        поддерживаем его в payload и как Cookie, если задан. (SID может передаваться
        cookie-значением SID либо параметром запроса.)
        """
        payload = {
            "cmd": "getHloc",
            "params": {
                "userId": user_id,
                "id": symbol,
                "count": int(count),
                "timeframe": int(timeframe_minutes),
                "date_from": _dt_to_tn_str(date_from_utc),
                "date_to": _dt_to_tn_str(date_to_utc),
                "intervalMode": interval_mode,
            },
        }
        if self.sid:
            payload["SID"] = self.sid

        # URL rotation: сначала основной, затем альтернативы (на случай блокировок/миграций домена).
        urls = [self.api_url] + [u for u in self.alt_api_urls if u != self.api_url]

        last_exc: Exception | None = None
        timeout = aiohttp.ClientTimeout(total=float(max(5, int(self.timeout_seconds))))

        headers = {"User-Agent": "qqq_trading_bot/1.0"}
        cookies = None
        if self.sid:
            cookies = {"SID": self.sid}

        for url in urls:
            # 2 попытки на URL (часто помогает при сетевой нестабильности)
            for attempt in (1, 2):
                try:
                    async with self.session.post(
                        url,
                        data={"q": json.dumps(payload, ensure_ascii=False)},
                        timeout=timeout,
                        headers=headers,
                        cookies=cookies,
                    ) as r:
                        r.raise_for_status()
                        txt = await r.text()

                    data = json.loads(txt)
                    if not isinstance(data, dict):
                        return []

                    if "hloc" not in data or symbol not in data.get("hloc", {}):
                        return []

                    hloc = data["hloc"][symbol]  # [[H, L, O, C], ...]
                    xs = data.get("xSeries", {}).get(symbol, [])
                    vl = data.get("vl", {}).get(symbol, [])

                    if not isinstance(hloc, list) or not isinstance(xs, list):
                        return []

                    bars: list[Bar] = []
                    n = min(len(hloc), len(xs))
                    for i in range(n):
                        row = hloc[i]
                        if not (isinstance(row, list) and len(row) >= 4):
                            continue
                        high, low, open_, close = row[0], row[1], row[2], row[3]

                        ts = datetime.fromtimestamp(int(xs[i]), tz=timezone.utc)
                        ts = floor_time(ts, timeframe_minutes)  # open time bucket
                        volume = float(vl[i]) if isinstance(vl, list) and i < len(vl) else 0.0

                        bars.append(
                            Bar(
                                ts=ts,
                                open=float(open_),
                                high=float(high),
                                low=float(low),
                                close=float(close),
                                volume=volume,
                                synthetic=False,
                            )
                        )

                    bars.sort(key=lambda b: b.ts)
                    return bars

                except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError) as e:  # type: ignore[name-defined]
                    last_exc = e
                    # короткий backoff
                    if attempt == 1:
                        await asyncio.sleep(0.4)
                    continue
                except Exception as e:
                    last_exc = e
                    break

        if last_exc is not None:
            raise last_exc
        return []

    async def get_quote(self, symbol: str) -> Quote:
        """
        Расширенный quote через securities/export:
        ltp — last traded price
        ltt — last traded time (может приходить без TZ)
        """
        params = {"params": "ltp,ltt", "tickers": symbol}
        async with self.session.get(self.quotes_url, params=params, timeout=10) as r:
            r.raise_for_status()
            txt = await r.text()
            data = json.loads(txt)

        if not isinstance(data, list) or not data:
            return Quote(symbol=symbol, ltp=None, ltt=None)

        row = data[0] if isinstance(data[0], dict) else {}
        ltp = safe_float(row.get("ltp"))
        # ltt у вас иногда без tz — поэтому тут не парсим, чтобы не вводить в заблуждение
        return Quote(symbol=symbol, ltp=ltp, ltt=None)
