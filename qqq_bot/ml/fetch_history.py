from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from qqq_bot.cache import cache_file_path
from qqq_bot.config import load_config
from qqq_bot.models import Bar
from qqq_bot.tradernet import TraderNetClient
from qqq_bot.utils_time import floor_time, utc_now


def _dt_to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Bad datetime: {value!r}")
    return ts.to_pydatetime().astimezone(timezone.utc)


def _bar_to_dict(b: Bar) -> dict[str, Any]:
    return {
        "ts": _dt_to_iso_z(b.ts),
        "open": float(b.open),
        "high": float(b.high),
        "low": float(b.low),
        "close": float(b.close),
        "volume": float(getattr(b, "volume", 0.0)),
        "synthetic": bool(getattr(b, "synthetic", False)),
    }


def _dict_to_bar(row: dict[str, Any]) -> Bar | None:
    try:
        ts = pd.to_datetime(row.get("ts") or row.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return Bar(
            ts=ts.to_pydatetime().astimezone(timezone.utc),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0) or 0.0),
            synthetic=bool(row.get("synthetic", False)),
        )
    except Exception:
        return None


def _load_existing(path: Path) -> list[Bar]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        bars = [_dict_to_bar(x) for x in data if isinstance(x, dict)]
        return sorted([b for b in bars if b is not None], key=lambda b: b.ts)
    except Exception:
        return []


def _write_bars(path: Path, bars: list[Bar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bars = sorted(bars, key=lambda b: b.ts)
    data = [_bar_to_dict(b) for b in bars]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _merge_bars(*chunks: list[Bar]) -> list[Bar]:
    uniq: dict[datetime, Bar] = {}
    for bars in chunks:
        for b in bars:
            if getattr(b, "synthetic", False):
                continue
            ts = b.ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            b.ts = ts.astimezone(timezone.utc)
            uniq[b.ts] = b
    return sorted(uniq.values(), key=lambda b: b.ts)


async def fetch_history(
    output: Path,
    lookback_days: int,
    chunk_days: int,
    max_count: int,
    sleep_seconds: float,
    date_to: datetime | None = None,
    date_from: datetime | None = None,
    merge_existing: bool = True,
) -> list[Bar]:
    cfg = load_config()
    tf = int(cfg.timeframe_minutes)

    end = date_to or floor_time(utc_now(), tf)
    start = date_from or (end - timedelta(days=int(lookback_days)))

    existing = _load_existing(output) if merge_existing else []

    async with aiohttp.ClientSession() as session:
        tn = TraderNetClient(
            api_url=cfg.tradernet_api_url,
            quotes_url=cfg.tradernet_quotes_url,
            session=session,
            sid=cfg.tradernet_sid,
            timeout_seconds=cfg.tradernet_timeout_seconds,
        )

        chunks: list[list[Bar]] = []
        cur_to = end
        i = 0
        while cur_to > start:
            cur_from = max(start, cur_to - timedelta(days=int(chunk_days)))
            # TraderNet often behaves better with count not too large. Use negative count
            # to ask for the latest bars inside the chunk.
            theoretical = int((cur_to - cur_from).total_seconds() // (tf * 60)) + 10
            count = min(int(max_count), max(50, theoretical))

            i += 1
            print(
                f"[{i}] fetch {cfg.symbol} {tf}m "
                f"{cur_from.isoformat()} -> {cur_to.isoformat()} count=-{count}",
                flush=True,
            )

            try:
                bars = await tn.get_hloc(
                    symbol=cfg.symbol,
                    timeframe_minutes=tf,
                    date_from_utc=cur_from,
                    date_to_utc=cur_to,
                    count=-int(count),
                )
                print(f"    got {len(bars)} bars", flush=True)
                chunks.append(bars)
            except Exception as exc:
                print(f"    ERROR: {exc!r}", flush=True)

            cur_to = cur_from
            if sleep_seconds > 0:
                await asyncio.sleep(float(sleep_seconds))

    merged = _merge_bars(existing, *chunks)
    _write_bars(output, merged)
    return merged


def main() -> None:
    cfg = load_config()
    default_output = Path("data/history") / f"bars_{cfg.symbol}_{cfg.timeframe_minutes}m_history.json"

    parser = argparse.ArgumentParser(description="Fetch longer historical bar file for ML training.")
    parser.add_argument("--output", default=str(default_output), help="Output JSON bars file for training.")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--chunk-days", type=int, default=2)
    parser.add_argument("--max-count", type=int, default=800)
    parser.add_argument("--sleep", type=float, default=0.5, help="Pause between API chunks, seconds.")
    parser.add_argument("--date-from", default=None, help="Optional ISO UTC date, e.g. 2025-01-01T00:00:00Z")
    parser.add_argument("--date-to", default=None, help="Optional ISO UTC date, e.g. 2026-05-15T20:00:00Z")
    parser.add_argument("--no-merge", action="store_true", help="Do not merge with existing output file.")
    args = parser.parse_args()

    output = Path(args.output)
    date_from = _parse_dt(args.date_from)
    date_to = _parse_dt(args.date_to)

    bars = asyncio.run(
        fetch_history(
            output=output,
            lookback_days=args.lookback_days,
            chunk_days=args.chunk_days,
            max_count=args.max_count,
            sleep_seconds=args.sleep,
            date_to=date_to,
            date_from=date_from,
            merge_existing=not args.no_merge,
        )
    )

    print(f"Wrote: {output}")
    print(f"Bars: {len(bars)}")
    if bars:
        print(f"Range: {_dt_to_iso_z(bars[0].ts)} -> {_dt_to_iso_z(bars[-1].ts)}")
    print("Next:")
    print(
        f"python -m qqq_bot.ml.build_training_data --input {output} "
        "--output data/ml/qqq_s2_training_long.parquet --strategy 2"
    )


if __name__ == "__main__":
    main()
