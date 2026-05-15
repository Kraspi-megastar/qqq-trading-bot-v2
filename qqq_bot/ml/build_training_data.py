from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_bot.cache import cache_file_path
from qqq_bot.config import load_config
from qqq_bot.indicators import atr, bollinger, ema, macd, rsi, supertrend, vwap
from qqq_bot.scheduler import Strategy2Runtime
from qqq_bot.signals import SignalDecision, compute_signal



RESERVED_FRAME_COLUMNS = {
    "timestamp", "ts", "open", "high", "low", "close", "volume", "synthetic",
    "ema_fast", "ema_slow", "ema9", "ema21", "ema_trend",
    "rsi", "bb_mid", "bb_upper", "bb_lower",
    "macd", "macd_signal", "macd_hist",
    "vwap", "atr", "supertrend", "supertrend_dir", "vol_ma",
}

MODEL_DETAIL_COLUMNS = {
    # Strategy #1 diagnostics
    "buy_score", "sell_score", "nearU", "nearL", "bounceU", "bounceL",
    "bb_ok", "rsi_ok", "ema_up", "ema_dn",

    # Strategy #2 diagnostics
    "macd_cross_up", "above_vwap", "st_up", "trend_ok", "vol_ok",
    "macd_zero_exit", "atr_exit", "atr_stop_mult",

    # Generic ML context
    "position_is_long", "position_is_short", "bars_since_last_signal",
    "timeframe_minutes",
}


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Final safety guard: parquet/csv cannot be reliably written with duplicated names."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df

def _parse_ts(value: Any) -> datetime:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Bad timestamp: {value!r}")
    return ts.to_pydatetime().astimezone(timezone.utc)


def _load_bars(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported input format: {path}")

    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df["volume"] = df["volume"].fillna(0.0)

    return df.reset_index(drop=True)


def _add_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = out["timestamp"]

    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    s = cfg.signal

    out["ema_fast"] = ema(close, s.ema_fast)
    out["ema_slow"] = ema(close, s.ema_slow)
    out["ema9"] = out["ema_fast"] if int(s.ema_fast) == 9 else ema(close, 9)
    out["ema21"] = out["ema_slow"] if int(s.ema_slow) == 21 else ema(close, 21)
    out["rsi"] = rsi(close, s.rsi_period)

    mid, upper, lower = bollinger(close, s.bb_period, s.bb_std)
    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower

    out["ema_trend"] = ema(close, getattr(s, "ema_trend_period", 200))

    m_line, m_sig, m_hist = macd(
        close,
        fast=getattr(s, "macd_fast", 12),
        slow=getattr(s, "macd_slow", 26),
        signal=getattr(s, "macd_signal", 9),
    )
    out["macd"] = m_line
    out["macd_signal"] = m_sig
    out["macd_hist"] = m_hist

    out["vwap"] = vwap(
        out,
        tz_name=cfg.display_tz,
        reset_daily=True,
        price_mode=getattr(s, "vwap_price_mode", "typical"),
    )
    out["atr"] = atr(out, period=getattr(s, "atr_period", 14))

    st_line, st_dir = supertrend(
        out,
        period=getattr(s, "supertrend_period", 10),
        multiplier=getattr(s, "supertrend_mult", 3.0),
    )
    out["supertrend"] = st_line
    out["supertrend_dir"] = st_dir

    vol = pd.to_numeric(out.get("volume", 0.0), errors="coerce").fillna(0.0)
    out["vol_ma"] = vol.rolling(int(getattr(s, "vol_ma_period", 20))).mean()

    return out


def _min_bars_for_indicators(cfg) -> int:
    s = cfg.signal
    return max(
        int(getattr(s, "bb_period", 20)),
        int(getattr(s, "ema_slow", 21)),
        int(getattr(s, "rsi_period", 14)),
        int(getattr(s, "ema_trend_period", 200)),
        int(getattr(s, "atr_period", 14)),
        int(getattr(s, "supertrend_period", 10)),
        int(getattr(s, "macd_slow", 26)),
    ) + 5


def _safe_detail_value(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, str):
        return value
    return None


def _flatten_decision(decision: SignalDecision) -> dict[str, Any]:
    row: dict[str, Any] = {
        "base_signal": decision.action,
        "decision_reason": decision.reason,
    }

    details = getattr(decision, "details", {}) or {}

    for key, value in details.items():
        key = str(key)
        safe = _safe_detail_value(value)

        # Keep row-level strategy diagnostics under their original names because
        # dataset.py can use them as model features.
        if key in MODEL_DETAIL_COLUMNS:
            row[key] = safe
            continue

        # Never let decision.details overwrite OHLCV / indicator columns.
        # Duplicated column names break pyarrow.to_parquet and can silently
        # confuse training. Keep these only as diagnostics.
        if key in RESERVED_FRAME_COLUMNS:
            row[f"detail_{key}"] = safe
            continue

        if isinstance(safe, str):
            row[f"detail_{key}"] = safe
        else:
            row[key] = safe

    return row


def _replay_strategy(df: pd.DataFrame, cfg, strategy_id: int, min_bars: int) -> pd.DataFrame:
    out = df.copy()
    runtime = Strategy2Runtime()

    decisions: list[dict[str, Any]] = []
    for i in range(len(out)):
        if i + 1 < min_bars:
            dec = SignalDecision("HOLD", "warmup", {"strategy": strategy_id})
        else:
            window = out.iloc[: i + 1].copy()
            dec = compute_signal(
                window,
                cfg.signal,
                strategy_id=strategy_id,
                runtime_state=runtime if strategy_id == 2 else None,
            )
        decisions.append(_flatten_decision(dec))

    dec_df = pd.DataFrame(decisions)
    result = pd.concat([out.reset_index(drop=True), dec_df.reset_index(drop=True)], axis=1)

    # Convenience normalized columns for dataset/features.
    result["base_signal"] = result["base_signal"].astype(str).str.upper().fillna("HOLD")
    result["signal_is_buy"] = result["base_signal"].eq("BUY").astype(int)
    result["signal_is_sell"] = result["base_signal"].eq("SELL").astype(int)

    return _dedupe_columns(result)


def _write_frame(df: pd.DataFrame, output: str | Path) -> None:
    df = _dedupe_columns(df)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        df.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    else:
        raise ValueError("Output must end with .parquet or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML training frame from QQQ bot cached bars.")
    parser.add_argument("--input", default=None, help="Input bars file: json/csv/parquet. Default: cache file from config.")
    parser.add_argument("--output", default="data/ml/qqq_s2_training.parquet")
    parser.add_argument("--strategy", type=int, default=2, choices=[1, 2])
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--min-bars", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    input_path = args.input or str(cache_file_path(cfg.cache_dir, cfg.symbol, cfg.timeframe_minutes))

    raw = _load_bars(input_path)
    if not args.include_synthetic and "synthetic" in raw.columns:
        raw = raw[~raw["synthetic"].astype(bool)].copy()

    with_ind = _add_indicators(raw, cfg)
    min_bars = int(args.min_bars or _min_bars_for_indicators(cfg))
    frame = _replay_strategy(with_ind, cfg=cfg, strategy_id=args.strategy, min_bars=min_bars)

    # Put timestamp first and keep ts for compatibility with strategy/chart code.
    if "timestamp" in frame.columns:
        cols = ["timestamp"] + [c for c in frame.columns if c != "timestamp"]
        frame = frame[cols]

    _write_frame(frame, args.output)

    signals = frame["base_signal"].value_counts(dropna=False).to_dict()
    print(f"Wrote: {args.output}")
    print(f"Rows: {len(frame)}")
    print(f"Input: {input_path}")
    print(f"Strategy: #{args.strategy}; min_bars={min_bars}")
    print(f"Signals: {signals}")
    if len(frame) > 0:
        print(f"Range: {frame['timestamp'].iloc[0]} -> {frame['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
