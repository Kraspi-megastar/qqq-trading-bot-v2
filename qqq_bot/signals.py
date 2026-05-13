from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import SignalConfig


@dataclass(frozen=True)
class SignalDecision:
    action: str  # "BUY" | "SELL" | "HOLD"
    reason: str
    details: dict[str, Any]


def _bool(v: Any) -> bool:
    return bool(v) if v is not None else False


def _cross_up(prev_a: float, prev_b: float, a: float, b: float) -> bool:
    return (prev_a <= prev_b) and (a > b)


# -------- STRATEGY #1 (как было) --------
def _compute_strategy_1(df: pd.DataFrame, cfg: SignalConfig) -> SignalDecision:
    if df is None or len(df) == 0:
        return SignalDecision("HOLD", "Нет данных.", {})

    last = df.iloc[-1]

    price = float(last.get("close"))
    rsi_v = float(last.get("rsi")) if pd.notna(last.get("rsi")) else float("nan")
    ema_f = float(last.get("ema_fast")) if pd.notna(last.get("ema_fast")) else float("nan")
    ema_s = float(last.get("ema_slow")) if pd.notna(last.get("ema_slow")) else float("nan")
    bb_l = float(last.get("bb_lower")) if pd.notna(last.get("bb_lower")) else float("nan")
    bb_m = float(last.get("bb_mid")) if pd.notna(last.get("bb_mid")) else float("nan")
    bb_u = float(last.get("bb_upper")) if pd.notna(last.get("bb_upper")) else float("nan")

    ema_up = ema_f > ema_s
    ema_dn = ema_f < ema_s

    thr = price * float(cfg.near_bb_tol)
    nearL = ((price - bb_l) <= thr) and (price <= bb_m)
    nearU = ((bb_u - price) <= thr) and (price >= bb_m)

    n = int(cfg.bounce_lookback)
    bounceL = False
    bounceU = False
    if len(df) >= n + 1 and all(col in df.columns for col in ["close", "bb_lower", "bb_upper"]):
        prev = df.iloc[-(n + 1):-1]
        prev_close = prev["close"].astype(float)
        prev_l = prev["bb_lower"].astype(float)
        prev_u = prev["bb_upper"].astype(float)
        bounceL = ((prev_close < prev_l).any()) and (price > bb_l)
        bounceU = ((prev_close > prev_u).any()) and (price < bb_u)

    bb_buy_ok = nearL or bounceL
    bb_sell_ok = nearU or bounceU

    rsi_buy_ok = (nearL and rsi_v <= float(cfg.rsi_buy)) or (bounceL and rsi_v <= float(cfg.rsi_buy_bounce))
    rsi_sell_ok = (nearU and rsi_v >= float(cfg.rsi_sell)) or (bounceU and rsi_v >= float(cfg.rsi_sell_bounce))

    buy_score = int(ema_up) + int(bb_buy_ok) + int(rsi_buy_ok)
    sell_score = int(ema_dn) + int(bb_sell_ok) + int(rsi_sell_ok)

    buy = (buy_score >= 2) and rsi_buy_ok
    sell = (sell_score >= 2) and rsi_sell_ok

    details = {
        "strategy": 1,
        "rsi": rsi_v,
        "price": price,
        "nearL": nearL,
        "nearU": nearU,
        "bounceL": bounceL,
        "bounceU": bounceU,
        "ema_up": ema_up,
        "ema_dn": ema_dn,
        "buy_score": buy_score,
        "sell_score": sell_score,
    }

    if buy and not sell:
        return SignalDecision("BUY", f"BUY: 2/3 подтверждений (RSI обязателен). buy_score={buy_score}", details)
    if sell and not buy:
        return SignalDecision("SELL", f"SELL: 2/3 подтверждений (RSI обязателен). sell_score={sell_score}", details)

    return SignalDecision("HOLD", f"HOLD: buy_score={buy_score}, sell_score={sell_score}", details)


# -------- STRATEGY #2 (MACD + VWAP + RSI + Supertrend) --------
def _compute_strategy_2(df: pd.DataFrame, cfg: SignalConfig, runtime_state: Any | None) -> SignalDecision:
    if df is None or len(df) < 2:
        return SignalDecision("HOLD", "STR#2: недостаточно данных.", {"strategy": 2})

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # timestamp последнего закрытого бара
    bar_ts = last.get("ts", None)

    close = float(last.get("close"))
    vwap_v = float(last.get("vwap")) if pd.notna(last.get("vwap")) else float("nan")
    rsi_v = float(last.get("rsi")) if pd.notna(last.get("rsi")) else float("nan")

    macd_v = float(last.get("macd")) if pd.notna(last.get("macd")) else float("nan")
    macd_sig = float(last.get("macd_signal")) if pd.notna(last.get("macd_signal")) else float("nan")
    prev_macd_v = float(prev.get("macd")) if pd.notna(prev.get("macd")) else float("nan")
    prev_macd_sig = float(prev.get("macd_signal")) if pd.notna(prev.get("macd_signal")) else float("nan")

    st_dir = int(last.get("supertrend_dir")) if pd.notna(last.get("supertrend_dir")) else 0
    ema200 = float(last.get("ema_trend")) if pd.notna(last.get("ema_trend")) else float("nan")
    atr_v = float(last.get("atr")) if pd.notna(last.get("atr")) else float("nan")

    vol = float(last.get("volume")) if pd.notna(last.get("volume")) else 0.0
    vol_ma = float(last.get("vol_ma")) if pd.notna(last.get("vol_ma")) else float("nan")

    pos = getattr(runtime_state, "position", "FLAT") if runtime_state is not None else "FLAT"
    atr_stop = getattr(runtime_state, "atr_stop", None) if runtime_state is not None else None

    macd_cross_up = _cross_up(prev_macd_v, prev_macd_sig, macd_v, macd_sig)
    above_vwap = close > vwap_v if pd.notna(vwap_v) else False
    rsi_ok = rsi_v > 50.0 if pd.notna(rsi_v) else False
    st_up = (st_dir == 1)

    trend_ok = close > ema200 if pd.notna(ema200) else True
    vol_filter_enabled = (pd.notna(vol_ma) and vol_ma > 0 and vol > 0)
    vol_ok = (vol > vol_ma) if vol_filter_enabled else True

    stop_mult = float(getattr(cfg, "atr_stop_mult", 3.0))
    new_stop_candidate = close - stop_mult * atr_v if (pd.notna(atr_v) and atr_v > 0) else None
    if pos == "LONG" and atr_stop is not None and new_stop_candidate is not None:
        new_atr_stop = max(float(atr_stop), float(new_stop_candidate))
    else:
        new_atr_stop = float(new_stop_candidate) if new_stop_candidate is not None else None

    # exit: MACD below 0-line
    macd_zero_exit = (prev_macd_v >= 0.0) and (macd_v < 0.0)
    atr_exit = (pos == "LONG") and (new_atr_stop is not None) and (close < new_atr_stop)

    details = {
        "strategy": 2,
        "position": pos,
        "close": close,
        "vwap": vwap_v,
        "rsi": rsi_v,
        "macd": macd_v,
        "macd_signal": macd_sig,
        "macd_cross_up": macd_cross_up,
        "above_vwap": above_vwap,
        "rsi_ok": rsi_ok,
        "supertrend_dir": st_dir,
        "st_up": st_up,
        "ema200": ema200,
        "trend_ok": trend_ok,
        "vol": vol,
        "vol_ma": vol_ma,
        "vol_ok": vol_ok,
        "atr": atr_v,
        "atr_stop": new_atr_stop,
        "macd_zero_exit": macd_zero_exit,
        "atr_exit": atr_exit,
        "atr_stop_mult": stop_mult,
    }

    # EXIT
    if pos == "LONG" and (macd_zero_exit or atr_exit):
        if runtime_state is not None:
            runtime_state.position = "FLAT"
            runtime_state.atr_stop = None
            runtime_state.entry_price = None
            runtime_state.entry_ts = None
        return SignalDecision("SELL", "STR#2 EXIT: macd_zero_exit or ATR-stop", details)

    # ENTRY
    entry_ok = macd_cross_up and above_vwap and rsi_ok and st_up and vol_ok and trend_ok
    if pos != "LONG" and entry_ok:
        if runtime_state is not None:
            runtime_state.position = "LONG"
            runtime_state.entry_price = close
            runtime_state.atr_stop = new_atr_stop
            runtime_state.entry_ts = bar_ts  # ключевое для persistence
        return SignalDecision("BUY", "STR#2 BUY: MACD↑ + close>VWAP + RSI>50 + ST↑ + filters OK", details)

    # UPDATE STOP in HOLD when long
    if runtime_state is not None and pos == "LONG":
        runtime_state.atr_stop = new_atr_stop

    return SignalDecision("HOLD", "STR#2 HOLD", details)


def compute_signal(
    df: pd.DataFrame,
    cfg: SignalConfig,
    strategy_id: int = 1,
    runtime_state: Any | None = None,
) -> SignalDecision:
    return _compute_strategy_2(df, cfg, runtime_state) if int(strategy_id) == 2 else _compute_strategy_1(df, cfg)
