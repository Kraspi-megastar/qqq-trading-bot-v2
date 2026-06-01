from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp
import asyncio
from zoneinfo import ZoneInfo

from .models import Quote, Bar
from .utils_time import safe_float, floor_time


_TN_TZ = ZoneInfo("Europe/Moscow")  # СЃРµСЂРІРµСЂ С‚СЂР°РєС‚СѓРµС‚ СЃС‚СЂРѕРєРё DD.MM.YYYY HH:MM РєР°Рє MSK (РїРѕ РґРёР°РіРЅРѕСЃС‚РёРєРµ)


def _dt_to_tn_str(dt_utc: datetime) -> str:
    """
    TraderNet getHloc РѕР¶РёРґР°РµС‚ СЃС‚СЂРѕРєСѓ 'DD.MM.YYYY HH:MM' Р±РµР· TZ.
    РџСЂР°РєС‚РёС‡РµСЃРєРё СЃРµСЂРІРµСЂ РёРЅС‚РµСЂРїСЂРµС‚РёСЂСѓРµС‚ РµС‘ РєР°Рє MSK (UTC+3),
    РїРѕСЌС‚РѕРјСѓ РєРѕРЅРІРµСЂС‚РёСЂСѓРµРј РёР· UTC -> MSK Рё С„РѕСЂРјР°С‚РёСЂСѓРµРј.
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
        Р РµР°Р»С‚Р°Р№Рј: Р±РµСЂРµРј РїРѕСЃР»РµРґРЅСЋСЋ С†РµРЅСѓ (ltp) С‡РµСЂРµР· securities/export.
        """
        params = {"fields": "ltp", "tickers": symbol}
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
        РСЃС‚РѕСЂРёСЏ getHloc (candlesticks).
        Р’ РґРѕРєСѓРјРµРЅС‚Р°С†РёРё TraderNet РјРµС‚РѕРґС‹ С‡Р°СЃС‚Рѕ С‚СЂРµР±СѓСЋС‚ SID (Р°РІС‚РѕСЂРёР·Р°С†РёСЏ) вЂ”
        РїРѕРґРґРµСЂР¶РёРІР°РµРј РµРіРѕ РІ payload Рё РєР°Рє Cookie, РµСЃР»Рё Р·Р°РґР°РЅ. (SID РјРѕР¶РµС‚ РїРµСЂРµРґР°РІР°С‚СЊСЃСЏ
        cookie-Р·РЅР°С‡РµРЅРёРµРј SID Р»РёР±Рѕ РїР°СЂР°РјРµС‚СЂРѕРј Р·Р°РїСЂРѕСЃР°.)
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

        # URL rotation: СЃРЅР°С‡Р°Р»Р° РѕСЃРЅРѕРІРЅРѕР№, Р·Р°С‚РµРј Р°Р»СЊС‚РµСЂРЅР°С‚РёРІС‹ (РЅР° СЃР»СѓС‡Р°Р№ Р±Р»РѕРєРёСЂРѕРІРѕРє/РјРёРіСЂР°С†РёР№ РґРѕРјРµРЅР°).
        urls = [self.api_url] + [u for u in self.alt_api_urls if u != self.api_url]

        last_exc: Exception | None = None
        timeout = aiohttp.ClientTimeout(total=float(max(5, int(self.timeout_seconds))))

        headers = {"User-Agent": "qqq_trading_bot/1.0"}
        cookies = None
        if self.sid:
            cookies = {"SID": self.sid}

        for url in urls:
            # 2 РїРѕРїС‹С‚РєРё РЅР° URL (С‡Р°СЃС‚Рѕ РїРѕРјРѕРіР°РµС‚ РїСЂРё СЃРµС‚РµРІРѕР№ РЅРµСЃС‚Р°Р±РёР»СЊРЅРѕСЃС‚Рё)
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
                    # РєРѕСЂРѕС‚РєРёР№ backoff
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
        Р Р°СЃС€РёСЂРµРЅРЅС‹Р№ quote С‡РµСЂРµР· securities/export:
        ltp вЂ” last traded price
        ltt вЂ” last traded time (РјРѕР¶РµС‚ РїСЂРёС…РѕРґРёС‚СЊ Р±РµР· TZ)
        """
        params = {"fields": "ltp,ltt", "tickers": symbol}
        async with self.session.get(self.quotes_url, params=params, timeout=10) as r:
            r.raise_for_status()
            txt = await r.text()
            data = json.loads(txt)

        if not isinstance(data, list) or not data:
            return Quote(symbol=symbol, ltp=None, ltt=None)

        row = data[0] if isinstance(data[0], dict) else {}
        ltp = safe_float(row.get("ltp"))
        # ltt Сѓ РІР°СЃ РёРЅРѕРіРґР° Р±РµР· tz вЂ” РїРѕСЌС‚РѕРјСѓ С‚СѓС‚ РЅРµ РїР°СЂСЃРёРј, С‡С‚РѕР±С‹ РЅРµ РІРІРѕРґРёС‚СЊ РІ Р·Р°Р±Р»СѓР¶РґРµРЅРёРµ
        return Quote(symbol=symbol, ltp=ltp, ltt=None)

    async def get_option_quote(self, ticker: str) -> dict:
        """Return raw option quote fields from TraderNet securities/export.

        TraderNet uses compact field names in securities/export. For bid/ask the
        public documentation names bbp/bap as best bid/best ask; ltp is last
        traded price. Greeks/IV are requested as best-effort because availability
        depends on market/account/data entitlement and may be absent for US options.
        """
        fields = [
            "ltp", "ltt",
            "bbp", "bap", "bbs", "bas",
            "vol", "vlt", "oi",
            "iv", "delta", "gamma", "theta", "vega",
        ]
        params = {"fields": ",".join(fields), "tickers": ticker}
        headers = {"User-Agent": "qqq_trading_bot/1.0"}
        cookies = {"SID": self.sid} if self.sid else None
        async with self.session.get(self.quotes_url, params=params, timeout=10, headers=headers, cookies=cookies) as r:
            r.raise_for_status()
            txt = await r.text()
            data = json.loads(txt)

        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            return {"ticker": ticker, "raw": data}

        row = dict(data[0])
        row["ticker"] = ticker
        return row

