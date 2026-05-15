from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_chart(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    last_signal: str | None = None,
    signal_history: list[tuple[str, object]] | None = None,  # [(BUY/SELL, ts), ...]
    max_signals: int = 6,
) -> None:
    if df is None or len(df) == 0:
        raise ValueError("plot_chart: empty dataframe")

    out = df.copy()

    if "ts" not in out.columns:
        raise ValueError("plot_chart: dataframe must contain 'ts' column")

    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).reset_index(drop=True)

    if len(out) == 0:
        raise ValueError("plot_chart: dataframe has no valid ts values")

    # X-axis as bar index (so there are no visual session gaps)
    out["_x"] = range(len(out))

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # --- main price panel ---
    ax1.plot(out["_x"], out["close"], label="Price")

    if "ema_fast" in out.columns:
        ax1.plot(out["_x"], out["ema_fast"], label="EMA fast")
    if "ema_slow" in out.columns:
        ax1.plot(out["_x"], out["ema_slow"], label="EMA slow")
    if "ema_trend" in out.columns:
        ax1.plot(out["_x"], out["ema_trend"], label="EMA trend")

    if "bb_upper" in out.columns:
        ax1.plot(out["_x"], out["bb_upper"], label="BB upper")
    if "bb_mid" in out.columns:
        ax1.plot(out["_x"], out["bb_mid"], label="BB mid")
    if "bb_lower" in out.columns:
        ax1.plot(out["_x"], out["bb_lower"], label="BB lower")

    if "vwap" in out.columns:
        ax1.plot(out["_x"], out["vwap"], label="VWAP")
    if "supertrend" in out.columns:
        ax1.plot(out["_x"], out["supertrend"], label="Supertrend")

    # --- signal markers ---
    if signal_history:
        normalized = []

        for item in signal_history:
            try:
                if not item or len(item) < 2:
                    continue
                action = str(item[0]).upper()
                ts = pd.to_datetime(item[1], utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                if action not in ("BUY", "SELL"):
                    continue
                normalized.append((action, ts))
            except Exception:
                continue

        # keep only signals inside visible time window
        ts_min = out["ts"].iloc[0]
        ts_max = out["ts"].iloc[-1]

        normalized = [
            (action, ts)
            for action, ts in normalized
            if ts_min <= ts <= ts_max
        ]

        # keep only last N signals
        if max_signals > 0:
            normalized = normalized[-max_signals:]

        # draw markers by nearest bar
        for action, sig_ts in normalized:
            idx = (out["ts"] - sig_ts).abs().idxmin()
            x = out.loc[idx, "_x"]
            y = out.loc[idx, "close"]

            if action == "BUY":
                ax1.scatter(x, y, marker="^", s=220, label=None)
            else:
                ax1.scatter(x, y, marker="v", s=220, label=None)

    ax1.set_title(f"{title} (bar-index {len(out)-1}, no session gaps)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- RSI panel ---
    if "rsi" in out.columns:
        ax2.plot(out["_x"], out["rsi"], label="RSI")
        ax2.axhline(70, linewidth=1)
        ax2.axhline(50, linewidth=1)
        ax2.axhline(30, linewidth=1)

    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # X labels from timestamps, but not every single bar
    step = max(1, len(out) // 10)
    tick_idx = list(range(0, len(out), step))
    if tick_idx[-1] != len(out) - 1:
        tick_idx.append(len(out) - 1)

    tick_labels = [
        out.loc[i, "ts"].strftime("%m-%dT%H:%M")
        for i in tick_idx
    ]

    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_labels, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)