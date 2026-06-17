"""
pipeline.py — единственное место для _bars_to_df, _add_indicators, _min_bars_for_indicators.
Импортируется и из scheduler.py, и из handlers.py, чтобы не было дублирования.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .indicators import ema, rsi, bollinger, macd, vwap, atr, supertrend
from .models import Bar

if TYPE_CHECKING:
    from .config import AppConfig


def bars_to_df(bars: list[Bar]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "ts": b.ts,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "synth": bool(getattr(b, "synthetic", False)),
            }
            for b in bars
        ]
    )
    if not df.empty:
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame, cfg: "AppConfig") -> pd.DataFrame:
    if df.empty:
        return df

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    # strategy #1
    df["ema_fast"] = ema(close, cfg.signal.ema_fast)
    df["ema_slow"] = ema(close, cfg.signal.ema_slow)
    df["rsi"] = rsi(close, cfg.signal.rsi_period)
    mid, upper, lower = bollinger(close, cfg.signal.bb_period, cfg.signal.bb_std)
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower

    # strategy #2 extras
    df["ema_trend"] = ema(close, getattr(cfg.signal, "ema_trend_period", 200))
    m_line, m_sig, m_hist = macd(
        close,
        fast=getattr(cfg.signal, "macd_fast", 12),
        slow=getattr(cfg.signal, "macd_slow", 26),
        signal=getattr(cfg.signal, "macd_signal", 9),
    )
    df["macd"] = m_line
    df["macd_signal"] = m_sig
    df["macd_hist"] = m_hist

    df["vwap"] = vwap(
        df,
        tz_name=cfg.display_tz,
        reset_daily=True,
        price_mode=getattr(cfg.signal, "vwap_price_mode", "typical"),
    )

    df["atr"] = atr(df, period=getattr(cfg.signal, "atr_period", 14))

    st_line, st_dir = supertrend(
        df,
        period=getattr(cfg.signal, "supertrend_period", 10),
        multiplier=getattr(cfg.signal, "supertrend_mult", 3.0),
    )
    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir

    vol_ma_period = int(getattr(cfg.signal, "vol_ma_period", 20))
    vol_raw = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)

    # Если у всего датафрейма volume == 0 (типично для realtime-баров без данных объёма),
    # обнуляем vol_ma — сигнальный слой увидит NaN и отключит vol_filter.
    if float(vol_raw.sum()) > 0:
        df["vol_ma"] = vol_raw.rolling(vol_ma_period).mean()
    else:
        df["vol_ma"] = float("nan")

    return df


def min_bars_for_indicators(cfg: "AppConfig") -> int:
    s = cfg.signal
    need = max(
        int(getattr(s, "ema_slow", 21)),
        int(getattr(s, "bb_period", 20)),
        int(getattr(s, "rsi_period", 14)),
        int(getattr(s, "ema_trend_period", 200)),
        int(getattr(s, "atr_period", 14)),
        int(getattr(s, "supertrend_period", 10)),
        int(getattr(s, "macd_slow", 26)),
    )
    return need + 5
