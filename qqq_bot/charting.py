from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def plot_chart(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str = "",
    current_action: str | None = None,
    signal_history: list[tuple[str, object]] | None = None,  # [(BUY/SELL, ts), ...]
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df is None or len(df) == 0:
        fig = plt.figure(figsize=(12, 6))
        plt.title(title or "chart")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    d = df.copy()

    ts = pd.to_datetime(d.get("ts"), utc=True, errors="coerce")
    ts_iso = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
    x = np.arange(len(d))

    close = pd.to_numeric(d.get("close"), errors="coerce")
    ema_fast = pd.to_numeric(d.get("ema_fast"), errors="coerce")
    ema_slow = pd.to_numeric(d.get("ema_slow"), errors="coerce")
    ema_trend = pd.to_numeric(d.get("ema_trend"), errors="coerce")  # EMA200 для страт#2

    bb_u = pd.to_numeric(d.get("bb_upper"), errors="coerce")
    bb_m = pd.to_numeric(d.get("bb_mid"), errors="coerce")
    bb_l = pd.to_numeric(d.get("bb_lower"), errors="coerce")

    vwap = pd.to_numeric(d.get("vwap"), errors="coerce")
    st = pd.to_numeric(d.get("supertrend"), errors="coerce")

    rsi = pd.to_numeric(d.get("rsi"), errors="coerce")

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_rsi = fig.add_subplot(gs[1, 0], sharex=ax)

    # PRICE
    ax.plot(x, close, linewidth=1.2, label="Price")
    if ema_fast.notna().any():
        ax.plot(x, ema_fast, linewidth=1.0, label="EMA fast")
    if ema_slow.notna().any():
        ax.plot(x, ema_slow, linewidth=1.0, label="EMA slow")
    if ema_trend.notna().any():
        ax.plot(x, ema_trend, linewidth=1.0, label="EMA trend")

    if bb_u.notna().any():
        ax.plot(x, bb_u, linewidth=0.9, label="BB upper")
    if bb_m.notna().any():
        ax.plot(x, bb_m, linewidth=0.9, label="BB mid")
    if bb_l.notna().any():
        ax.plot(x, bb_l, linewidth=0.9, label="BB lower")

    if vwap.notna().any():
        ax.plot(x, vwap, linewidth=1.0, label="VWAP")
    if st.notna().any():
        ax.plot(x, st, linewidth=1.0, label="Supertrend")

    # SIGNAL MARKERS (triangles)
    if signal_history:
        ts_to_idx = {ts_iso.iloc[i]: i for i in range(len(ts_iso)) if ts_iso.iloc[i]}
        # для fallback: числовой timestamp каждого бара (для поиска ближайшего)
        # наносекунды с эпохи для каждого бара (для поиска ближайшего).
        try:
            ts_epoch = ts.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()
        except Exception:
            ts_epoch = ts.astype("int64").to_numpy()
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        seen = set()

        for item in signal_history:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue

            action = item[0]
            ts_obj = item[1]
            price_hint = item[2] if len(item) >= 3 else None

            if action not in ("BUY", "SELL"):
                continue

            try:
                ts_dt = pd.to_datetime(ts_obj, utc=True, errors="coerce")
                ts_key = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                ts_dt = None
                ts_key = ""

            if not ts_key:
                continue

            # 1) точное совпадение по строке timestamp
            idx = ts_to_idx.get(ts_key)

            # 2) fallback: ближайший бар в пределах одного таймфрейма.
            #    Раньше при малейшем расхождении формата треугольник ПРОПАДАЛ
            #    (idx is None → continue). Теперь ищем ближайший бар.
            if idx is None and ts_dt is not None:
                try:
                    _t = pd.Timestamp(ts_dt)
                    if _t.tz is not None:
                        _t = _t.tz_convert("UTC").tz_localize(None)
                    target = int(_t.value)  # наносекунды
                    diffs = np.abs(ts_epoch - target)
                    nearest = int(np.argmin(diffs))
                    # принимаем только если ближайший бар в пределах ~1 бара (10 мин)
                    if diffs[nearest] <= 10 * 60 * 1_000_000_000:
                        idx = nearest
                except Exception:
                    idx = None

            if idx is None:
                continue

            if isinstance(price_hint, (int, float)):
                yv = float(price_hint)
            else:
                yv = close.iloc[idx]
                if pd.isna(yv):
                    continue
                yv = float(yv)

            k = (action, idx)
            if k in seen:
                continue
            seen.add(k)

            if action == "BUY":
                buy_x.append(idx)
                buy_y.append(yv)
            else:
                sell_x.append(idx)
                sell_y.append(yv)

        if buy_x:
            ax.scatter(buy_x, buy_y, marker="^", s=90, color="green", zorder=10)
        if sell_x:
            ax.scatter(sell_x, sell_y, marker="v", s=90, color="red", zorder=10)

    # RSI
    if rsi.notna().any():
        ax_rsi.plot(x, rsi, linewidth=1.0)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.axhline(30, linewidth=0.8)
        ax_rsi.axhline(50, linewidth=0.8)
        ax_rsi.axhline(70, linewidth=0.8)
        ax_rsi.set_ylabel("RSI", fontsize=9)

    # X labels
    n_ticks = 10
    if len(x) > 1:
        tick_pos = np.linspace(0, len(x) - 1, min(n_ticks, len(x))).astype(int)
        tick_lbl = [ts_iso.iloc[i][5:16] if ts_iso.iloc[i] else "" for i in tick_pos]
        ax_rsi.set_xticks(tick_pos)
        ax_rsi.set_xticklabels(tick_lbl, rotation=0, fontsize=8)

    bar_index = len(d) - 1
    ax.set_title(f"{title} (bar-index {bar_index}, no session gaps)", fontsize=10)

    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax_rsi.grid(True, alpha=0.2)
    plt.setp(ax.get_xticklabels(), visible=False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
