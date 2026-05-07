# candlestick trade plots from trade_sample_*.json (one figure per timeframe × lookback)

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


def _parse_tf_minutes(tf_dir_name: str) -> Optional[int]:
    m = re.match(r"^(\d+)m$", tf_dir_name)
    return int(m.group(1)) if m else None


def _parse_lookback(lb_dir_name: str) -> Optional[int]:
    m = re.match(r"^lookback_(\d+)$", lb_dir_name)
    return int(m.group(1)) if m else None


def _parse_atr(atr_dir_name: str) -> Optional[int]:
    m = re.match(r"^atr_(\d+)$", atr_dir_name)
    return int(m.group(1)) if m else None


def _hs_from_samples(atr_path: str, direction: str) -> set[int]:
    pat = os.path.join(atr_path, f"trade_sample_{direction}_H*.json")
    hs: set[int] = set()
    for path in glob.glob(pat):
        m = re.search(r"_H(\d+)\.json$", path)
        if m:
            hs.add(int(m.group(1)))
    return hs


def _load_sample(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _bar_x(d: dict[str, Any], bar: int) -> pd.Timestamp:
    j = int(bar) - int(d["slice_start_bar"])
    u = d["unix_time"]
    j = max(0, min(j, len(u) - 1))
    return pd.to_datetime(float(u[j]), unit="s", utc=True)


def _bar_xnum(d: dict[str, Any], bar: int) -> float:
    return mdates.date2num(_bar_x(d, bar).tz_convert("America/New_York"))


def _plot_candles(ax, d: dict[str, Any], title_suffix: str, atr_guess: int) -> None:
    t = pd.to_datetime(np.asarray(d["unix_time"], dtype=np.float64), unit="s", utc=True)
    x = mdates.date2num(t.tz_convert("America/New_York"))
    o = np.asarray(d["open"], dtype=np.float64)
    h = np.asarray(d["high"], dtype=np.float64)
    l = np.asarray(d["low"], dtype=np.float64)
    c = np.asarray(d["close"], dtype=np.float64)

    dx = (x[-1] - x[0]) / max(len(x), 2) * 0.55 if len(x) > 1 else 0.002

    for i in range(len(o)):
        color = "#26a69a" if c[i] >= o[i] else "#ef5350"
        ax.plot([x[i], x[i]], [l[i], h[i]], color=color, linewidth=1.0, solid_capstyle="round")
        body_lo, body_hi = min(o[i], c[i]), max(o[i], c[i])
        ax.add_patch(
            Rectangle(
                (x[i] - dx / 2, body_lo),
                dx,
                max(body_hi - body_lo, 1e-9),
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
            )
        )

    sf, sl = int(d["setup_first_bar"]), int(d["setup_last_bar"])
    eb, xb = int(d["entry_bar"]), int(d["exit_bar"])
    hb = eb + int(d["meta"].get("horizon", 0)) - 1

    x_setup_first = _bar_xnum(d, sf)
    x_setup_last = _bar_xnum(d, sl)
    x_entry = _bar_xnum(d, eb)
    x_exit = _bar_xnum(d, xb)
    x_horizon = _bar_xnum(d, hb)

    ax.axvline(x_setup_first, color="#90caf9", linestyle="--", linewidth=1.0, alpha=0.9, label="setup first")
    ax.axvline(x_setup_last, color="#64b5f6", linestyle="--", linewidth=1.0, alpha=0.9, label="setup last")
    ax.axvline(x_horizon, color="#ab47bc", linestyle="-", linewidth=1.1, alpha=0.95, label="horizon end")
    ax.axvline(x_exit, color="#ffb74d", linestyle="-", linewidth=1.2, alpha=0.95, label="exit close bar")

    ep = float(d["entry_price"])
    tp = float(d["target_price"])
    sp = float(d["stop_price"])
    x_right = x[-1] + dx

    # solid rays from entry forward
    ax.hlines(ep, x_entry, x_right, color="w", linestyle="-", linewidth=1.1, alpha=0.9, label="entry")
    ax.hlines(tp, x_entry, x_right, color="#81c784", linestyle="-", linewidth=1.1, alpha=0.9, label="target")
    ax.hlines(sp, x_entry, x_right, color="#e57373", linestyle="-", linewidth=1.1, alpha=0.9, label="stop")

    # dashed segment across setup -> entry gap
    ax.hlines(ep, x_setup_last, x_entry, color="w", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.hlines(tp, x_setup_last, x_entry, color="#81c784", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.hlines(sp, x_setup_last, x_entry, color="#e57373", linestyle="--", linewidth=1.0, alpha=0.9)

    # setup-only delta bounds (pink)
    dl = float(d.get("delta_left_price", np.nan))
    dr = float(d.get("delta_right_price", np.nan))
    if np.isfinite(dl):
        ax.hlines(
            dl,
            x_setup_first,
            x_setup_last,
            color="#ff66cc",
            linestyle="-",
            linewidth=1.1,
            alpha=0.95,
            label="delta left",
        )
    if np.isfinite(dr):
        ax.hlines(
            dr,
            x_setup_first,
            x_setup_last,
            color="#ff99dd",
            linestyle="-",
            linewidth=1.1,
            alpha=0.95,
            label="delta right",
        )

    my = float(d["marker"]["price"])
    mk = d["marker"]["kind"]
    ax.plot(
        x_exit,
        my,
        "o",
        markersize=9,
        markeredgecolor="w",
        markeredgewidth=1.0,
        color={"stop": "#e57373", "target": "#81c784", "timeout": "#ffd54f"}.get(mk, "#fff"),
        label=f"fill ({mk})",
        zorder=5,
    )

    meta = d["meta"]
    out_lbl = {-1: "stop", 0: "timeout", 1: "target"}.get(int(d["outcome"]), str(d["outcome"]))
    ax.set_title(
        f"{meta.get('symbol', '?')}  {meta.get('timeframe_minutes', '?')}m  "
        f"lookback={meta.get('lookback', '?')}  atr={atr_guess}  {title_suffix}\n"
        f"{meta.get('direction', '?')}  H={meta.get('horizon', '?')}  "
        f"mt={meta.get('target_mult')}  ms={meta.get('stop_mult')}  outcome={out_lbl}",
        fontsize=10,
    )
    ax.set_xlim(x[0] - dx, x[-1] + dx)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.5f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz="America/New_York"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.35)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot first-setup trade candles (random ATR per TF x lookback).")
    p.add_argument("run_root", help="Main output folder, e.g. output/EUR_USD_2023-01-01_2024-12-31")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None, help="Default: run_root/plots")
    args = p.parse_args()

    run_root = os.path.abspath(args.run_root)
    out_dir = args.out_dir or os.path.join(run_root, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.style.use("dark_background")
    rng = random.Random(args.seed)

    subdirs = sorted(
        d
        for d in os.listdir(run_root)
        if _parse_tf_minutes(d) is not None and os.path.isdir(os.path.join(run_root, d))
    )

    for tf_name in subdirs:
        tf = _parse_tf_minutes(tf_name)
        if tf is None:
            continue
        tf_path = os.path.join(run_root, tf_name)
        lb_dirs = sorted(
            d
            for d in os.listdir(tf_path)
            if d.startswith("lookback_") and os.path.isdir(os.path.join(tf_path, d))
        )
        for lb_name in lb_dirs:
            lb = _parse_lookback(lb_name)
            if lb is None:
                continue
            lb_path = os.path.join(tf_path, lb_name)
            atr_dirs = sorted(
                os.path.join(lb_path, d)
                for d in os.listdir(lb_path)
                if d.startswith("atr_") and os.path.isdir(os.path.join(lb_path, d))
            )
            if not atr_dirs:
                continue

            atr_path = rng.choice(atr_dirs)
            atr_val = _parse_atr(os.path.basename(atr_path)) or -1

            h_long = _hs_from_samples(atr_path, "long")
            h_short = _hs_from_samples(atr_path, "short")
            common = sorted(h_long & h_short)
            if not common:
                continue
            horizon = rng.choice(common)

            path_long = os.path.join(atr_path, f"trade_sample_long_H{horizon}.json")
            path_short = os.path.join(atr_path, f"trade_sample_short_H{horizon}.json")
            if not (os.path.isfile(path_long) and os.path.isfile(path_short)):
                continue

            dl = _load_sample(path_long)
            ds = _load_sample(path_short)

            fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=False)
            fig.patch.set_facecolor("#1e1e1e")
            _plot_candles(axes[0], dl, "long (1st setup)", atr_val)
            _plot_candles(axes[1], ds, "short (1st setup)", atr_val)
            fig.suptitle(f"Trade samples  |  random ATR={atr_val}  |  seed={args.seed}", fontsize=11, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            fname = f"candles_{tf}m_lookback_{lb}_H{horizon}_atr{atr_val}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=120, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"wrote {fname}")


if __name__ == "__main__":
    main()
