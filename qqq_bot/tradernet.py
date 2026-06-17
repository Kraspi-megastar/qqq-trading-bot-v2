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
