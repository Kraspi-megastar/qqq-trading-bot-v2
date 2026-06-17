# qqq_bot/diagnostics.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_API_URL = "https://tradernet.ru/api/"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "diagnostics.log"
JSONL_FILE = LOG_DIR / "diagnostics.jsonl"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("diagnostics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def jsonl_append(obj: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def dt_to_api_str(dt_: datetime) -> str:
    # TraderNet обычно принимает формат: "DD.MM.YYYY HH:MM"
    return dt_.strftime("%d.%m.%Y %H:%M")


def post_form(url: str, form: Dict[str, str], headers: Dict[str, str], timeout: int = 20) -> Tuple[int, str, bytes]:
    body = urlencode(form).encode("utf-8")
    req = Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "qqq_trading_bot/diagnostics",
            **headers,
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 0) or 0)
        content_type = resp.headers.get("Content-Type", "") or ""
        raw = resp.read()
    return status, content_type, raw


def safe_json_load(raw: bytes) -> Tuple[Optional[Any], Optional[str]]:
    try:
        txt = raw.decode("utf-8", errors="replace")
        return json.loads(txt), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def extract_symbol_series(payload: Any, symbol: str) -> Tuple[Optional[list], Optional[list], Optional[list]]:
    """
    Ожидаем структуру вида:
    {
      "hloc": {"QQQ.US": [[H,L,O,C], ...]},
      "vl":   {"QQQ.US": [v1, v2, ...]},
      "xSeries": {"QQQ.US": [unix_ts, unix_ts, ...]}
    }
    """
    if not isinstance(payload, dict):
        return None, None, None

    hloc = payload.get("hloc")
    vl = payload.get("vl")
    xs = payload.get("xSeries")

    if isinstance(hloc, dict):
        hloc = hloc.get(symbol)
    if isinstance(vl, dict):
        vl = vl.get(symbol)
    if isinstance(xs, dict):
        xs = xs.get(symbol)

    return xs if isinstance(xs, list) else None, hloc if isinstance(hloc, list) else None, vl if isinstance(vl, list) else None


def unix_to_dt_utc(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


@dataclass
class CycleResult:
    cycle: int
    cycles_total: int
    now_utc: str
    tz_offset_hours: int
    date_from: str
    date_to: str
    timeframe_min: int
    count_param: int
    interval_mode: str

    http_status: Optional[int] = None
    content_type: Optional[str] = None
    raw_len: Optional[int] = None
    parse_error: Optional[str] = None

    bars_count: Optional[int] = None
    max_ts_utc: Optional[str] = None
    staleness_sec: Optional[int] = None
    last_close: Optional[float] = None

    raw_head: Optional[str] = None
    top_keys: Optional[list] = None


async_mode_hint = False  # keep file import-friendly; script is sync.


def run() -> int:
    logger = setup_logging()

    p = argparse.ArgumentParser(description="Diagnostics for TraderNet getHloc (candles).")
    p.add_argument("--api-url", default=DEFAULT_API_URL)
    p.add_argument("--symbol", default="QQQ.US")
    p.add_argument("--timeframe", type=int, default=5, help="Candle timeframe in minutes.")
    p.add_argument("--lookback", type=int, default=240, help="Lookback window in minutes for date_from/date_to.")
    p.add_argument("--cycles", type=int, default=60)
    p.add_argument("--interval", type=int, default=10, help="Seconds between cycles.")
    p.add_argument("--interval-mode", default="ClosedRay", help="Usually ClosedRay.")
    p.add_argument(
        "--count",
        type=int,
        default=-48,
        help="Use 0 when using date_from/date_to. Negative may mean 'extra bars before interval' in some TN setups.",
    )
    p.add_argument(
        "--tz-offset",
        type=int,
        default=-5,
        help="Hours offset applied to UTC for date_from/date_to strings. For New York winter usually -5, summer -4.",
    )
    p.add_argument("--timeout", type=int, default=20)

    args = p.parse_args()

    sid = os.getenv("TRADERNET_SID", "").strip()
    if not sid:
        logger.info("TRADERNET_SID not set. Some setups return limited/delayed data without auth.")
    else:
        logger.info("TRADERNET_SID is set (length=%s).", len(sid))

    headers: Dict[str, str] = {}
    # В разных инсталляциях TraderNet SID может ожидаться как cookie.
    # Самый частый вариант: Cookie: SID=<value>
    if sid:
        headers["Cookie"] = f"SID={sid}"

    logger.info(
        "Starting diagnostics | symbol=%s timeframe=%sm lookback=%sm cycles=%s interval=%ss tz_offset=%s count=%s",
        args.symbol,
        args.timeframe,
        args.lookback,
        args.cycles,
        args.interval,
        args.tz_offset,
        args.count,
    )
    logger.info("Logs: %s and %s", LOG_FILE.as_posix(), JSONL_FILE.as_posix())

    for i in range(1, args.cycles + 1):
        now_utc = datetime.now(timezone.utc)
        tz = timezone(timedelta(hours=args.tz_offset))
        now_tz = now_utc.astimezone(tz)

        date_to_dt = now_tz
        date_from_dt = now_tz - timedelta(minutes=args.lookback)

        date_to = dt_to_api_str(date_to_dt)
        date_from = dt_to_api_str(date_from_dt)

        expected = max(1, args.lookback // max(1, args.timeframe))

        logger.info(
            "Cycle %s/%s | now_utc=%s | date_from=%s | date_to=%s | expected~%s bars",
            i,
            args.cycles,
            now_utc.isoformat(),
            date_from,
            date_to,
            expected,
        )

        params = {
            "id": args.symbol,
            "timeframe": args.timeframe,
            "date_from": date_from,
            "date_to": date_to,
            "intervalMode": args.interval_mode,
            "count": args.count,
        }
        q = {"cmd": "getHloc", "params": params}
        form = {"q": json.dumps(q, ensure_ascii=False)}

        cr = CycleResult(
            cycle=i,
            cycles_total=args.cycles,
            now_utc=now_utc.isoformat(),
            tz_offset_hours=args.tz_offset,
            date_from=date_from,
            date_to=date_to,
            timeframe_min=args.timeframe,
            count_param=args.count,
            interval_mode=args.interval_mode,
        )

        try:
            status, ctype, raw = post_form(args.api_url, form=form, headers=headers, timeout=args.timeout)
            cr.http_status = status
            cr.content_type = ctype
            cr.raw_len = len(raw)

            payload, perr = safe_json_load(raw)
            cr.parse_error = perr

            if perr:
                head = raw.decode("utf-8", errors="replace")[:200]
                cr.raw_head = head
                logger.warning("Bars | HTTP=%s | parse_error=%s | head=%r", status, perr, head)
                jsonl_append(asdict(cr))
            else:
                if isinstance(payload, dict):
                    cr.top_keys = list(payload.keys())[:50]

                xs, hloc, vl = extract_symbol_series(payload, args.symbol)

                logger.info(
                    "RAW lens | xSeries=%s hloc=%s vl=%s",
                    len(xs) if isinstance(xs, list) else None,
                    len(hloc) if isinstance(hloc, list) else None,
                    len(vl) if isinstance(vl, list) else None,
                )

                if not xs or not hloc:
                    # Может прийти ошибка в виде dict без нужных ключей
                    head = raw.decode("utf-8", errors="replace")[:200]
                    cr.raw_head = head
                    logger.warning("Bars | EMPTY/UNPARSEABLE | HTTP=%s | keys=%s | head=%r", status, cr.top_keys, head)
                    jsonl_append(asdict(cr))
                else:
                    cr.bars_count = len(xs)
                    max_ts = unix_to_dt_utc(xs[-1])
                    cr.max_ts_utc = max_ts.isoformat()
                    cr.staleness_sec = int((now_utc - max_ts).total_seconds())
                    try:
                        last = hloc[-1]
                        # ожидаем [H, L, O, C]
                        cr.last_close = float(last[3]) if isinstance(last, (list, tuple)) and len(last) >= 4 else None
                    except Exception:
                        cr.last_close = None

                    logger.info(
                        "Bars | count=%s | max_ts=%s | staleness_sec=%s | last_close=%s",
                        cr.bars_count,
                        cr.max_ts_utc,
                        cr.staleness_sec,
                        cr.last_close,
                    )
                    jsonl_append(asdict(cr))

        except Exception as e:
            logger.exception("Cycle failed: %s", e)
            cr.parse_error = f"{type(e).__name__}: {e}"
            jsonl_append(asdict(cr))

        time.sleep(max(1, args.interval))

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
