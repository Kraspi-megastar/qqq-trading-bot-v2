from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Protocol


@dataclass
class OptionQuote:
    ticker: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    mark: Optional[float] = None
    mid: Optional[float] = None
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    ts: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.mid is None and self.bid is not None and self.ask is not None:
            self.mid = (self.bid + self.ask) / 2.0
        if self.mark is None:
            self.mark = self.mid if self.mid is not None else self.last
        if self.ts is None:
            self.ts = datetime.now(timezone.utc).isoformat()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mark": self.mark,
            "mid": self.mid,
            "iv": self.iv,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "ts": self.ts,
            "raw": self.raw,
        }


class QuoteProvider(Protocol):
    def get_option_quote(self, ticker: str) -> Optional[OptionQuote]:
        ...


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def quote_from_mapping(ticker: str, data: Dict[str, Any]) -> OptionQuote:
    # Accept common key names from broker/export APIs.
    lower = {str(k).lower(): v for k, v in data.items()}
    def get(*keys: str):
        for key in keys:
            if key in data:
                return data[key]
            if key.lower() in lower:
                return lower[key.lower()]
        return None

    return OptionQuote(
        ticker=ticker,
        # TraderNet securities/export uses compact names:
        #   bbp/bap = best bid/best ask price, bbs/bas = sizes, ltp = last traded price.
        bid=_to_float(get("bid", "b", "bbp", "best_bid", "best_bid_price")),
        ask=_to_float(get("ask", "a", "bap", "best_ask", "best_ask_price")),
        last=_to_float(get("last", "lp", "ltp", "price", "close")),
        mark=_to_float(get("mark", "theo", "theoretical")),
        iv=_to_float(get("iv", "implied_volatility", "volatility")),
        delta=_to_float(get("delta")),
        gamma=_to_float(get("gamma")),
        theta=_to_float(get("theta")),
        vega=_to_float(get("vega")),
        volume=_to_float(get("volume", "vol", "vlt")),
        open_interest=_to_float(get("open_interest", "oi", "openinterest")),
        ts=str(get("ts", "time", "timestamp") or datetime.now(timezone.utc).isoformat()),
        raw=data,
    )


class DefaultOptionQuoteProvider:
    """Best-effort option quote adapter.

    1) If a project tradernet client is passed, this provider first tries TraderNet
       securities/export directly using client.quotes_url. This keeps the options
       layer synchronous and avoids awaiting inside the Telegram send path.
    2) If the client also exposes synchronous get_option_quote/get_quote/get_quotes,
       this provider can use those methods. Async methods are intentionally skipped
       here because process_options_signal is synchronous.
    3) If OPTION_QUOTE_URL is configured, it calls that HTTP endpoint with ?ticker=...
    4) Otherwise returns None. The options layer will still log the missing quote event.
    """

    def __init__(self, client: Any = None, quote_url: Optional[str] = None, timeout: float = 5.0):
        self.client = client
        self.quote_url = quote_url or os.getenv("OPTION_QUOTE_URL")
        self.timeout = timeout

    def get_option_quote(self, ticker: str) -> Optional[OptionQuote]:
        q = self._from_tradernet_export(ticker)
        if q is not None:
            return q
        q = self._from_client(ticker)
        if q is not None:
            return q
        return self._from_http(ticker)

    def _from_tradernet_export(self, ticker: str) -> Optional[OptionQuote]:
        """Read option quote from TraderNet securities/export if client.quotes_url is available."""
        if self.client is None:
            return None
        quotes_url = getattr(self.client, "quotes_url", None)
        if not quotes_url:
            return None
        fields = [
            "ltp", "ltt",
            "bbp", "bap", "bbs", "bas",
            "vol", "vlt", "oi",
            # Best-effort; some TraderNet accounts/markets may not return greeks/IV.
            "iv", "delta", "gamma", "theta", "vega",
        ]
        sep = "&" if "?" in quotes_url else "?"
        url = f"{quotes_url}{sep}{urllib.parse.urlencode({'fields': ','.join(fields), 'tickers': ticker})}"
        headers = {"User-Agent": "qqq_trading_bot/1.0"}
        sid = getattr(self.client, "sid", None)
        if sid:
            headers["Cookie"] = f"SID={sid}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8")
            data = json.loads(raw)
            if isinstance(data, list) and data:
                item = data[0]
                if isinstance(item, dict):
                    return quote_from_mapping(ticker, item)
            if isinstance(data, dict):
                payload = data.get("data", data)
                if isinstance(payload, list) and payload:
                    payload = payload[0]
                if isinstance(payload, dict):
                    return quote_from_mapping(ticker, payload)
        except Exception:
            return None
        return None

    def _from_client(self, ticker: str) -> Optional[OptionQuote]:
        if self.client is None:
            return None
        for name in ("get_option_quote", "get_quote"):
            fn = getattr(self.client, name, None)
            if callable(fn):
                data = fn(ticker)
                if inspect.isawaitable(data):
                    # Runtime bot loop is async; this sync provider cannot await here.
                    # Use OPTION_QUOTE_URL or add a synchronous get_option_quote adapter for real bid/ask/greeks.
                    continue
                if isinstance(data, OptionQuote):
                    return data
                if isinstance(data, dict):
                    return quote_from_mapping(ticker, data)
        fn = getattr(self.client, "get_quotes", None)
        if callable(fn):
            data = fn([ticker])
            if inspect.isawaitable(data):
                return None
            if isinstance(data, dict):
                item = data.get(ticker) or data.get("data") or data
                if isinstance(item, list) and item:
                    item = item[0]
                if isinstance(item, dict):
                    return quote_from_mapping(ticker, item)
            if isinstance(data, list) and data:
                item = data[0]
                if isinstance(item, dict):
                    return quote_from_mapping(ticker, item)
        return None

    def _from_http(self, ticker: str) -> Optional[OptionQuote]:
        if not self.quote_url:
            return None
        url = self.quote_url
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urllib.parse.urlencode({'ticker': ticker})}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                payload = data.get("data", data)
                if isinstance(payload, list) and payload:
                    payload = payload[0]
                if isinstance(payload, dict):
                    return quote_from_mapping(ticker, payload)
        except Exception:
            return None
        return None

