# Per joint bin_id: outer grid window_index (x) × horizon (y); each cell = target × stop heatmap.
# Optional --spread-bins / --accel-bins to limit which joint bins are rendered.
# Writes PNGs under run_root (sibling to Sample CSV).

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from heatmap_edge_combine import (
        build_merged_edge_payload,
        evaluate_long_short_edges,
        load_edge_dict,
    )
except ImportError:
    from analysis.heatmap_edge_combine import (
        build_merged_edge_payload,
        evaluate_long_short_edges,
        load_edge_dict,
    )


def _tf_folder_name(minutes: int) -> str:
    return f"{int(minutes)}m"


def _parse_int_list(s: str | None) -> set[int] | None:
    if s is None or not str(s).strip():
        return None
    return {int(x.strip()) for x in str(s).split(",") if x.strip()}


def _load_parquet_in_atr_dir(atr_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(atr_dir, "pnl_summaries_forward_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet summary in {atr_dir}")
    return pd.read_parquet(paths[0])


def _load_k_from_edge(atr_dir: str, direction: str) -> int:
    name = "long_edge_data.json" if direction == "long" else "short_edge_data.json"
    path = os.path.join(atr_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing edge JSON: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return int(data["k"])


def _pivot_metric(cell_df: pd.DataFrame, targets: np.ndarray, stops: np.ndarray, col: str) -> np.ndarray:
    out = np.full((targets.size, stops.size), np.nan, dtype=np.float64)
    if cell_df.empty:
        return out
    tmp = cell_df.pivot_table(index="target_mult", columns="stop_mult", values=col, aggfunc="first")
    for ti, t in enumerate(targets):
        if t not in tmp.index:
            continue
        for si, s in enumerate(stops):
            if s in tmp.columns:
                val = tmp.loc[t, s]
                out[ti, si] = float(val) if pd.notna(val) else np.nan
    return out


def _plot_bin_grid(
    df_bin: pd.DataFrame,
    bin_id: int,
    spread_i: int,
    accel_j: int,
    windows: np.ndarray,
    horizons: np.ndarray,
    targets: np.ndarray,
    stops: np.ndarray,
    min_count: int,
    vmin: float,
    vmax: float,
    out_path: str,
    title_prefix: str,
) -> None:
    nH, nW = len(horizons), len(windows)
    cell_w_in = max(2.0, min(3.4, 23.0 / max(nW, 1)))
    cell_h_in = max(2.0, min(3.4, 28.0 / max(nH, 1)))
    fig_w = max(8.0, cell_w_in * nW + 2.6)
    fig_h = max(6.5, cell_h_in * nH + 2.3)
    inner_fs = max(7.0, min(12.5, 2.65 * min(cell_w_in, cell_h_in)))
    tick_fs = inner_fs + 1.0
    coltitle_fs = inner_fs + 1.2
    supt_fs = min(13.5, inner_fs + 2.5)

    fig, axes = plt.subplots(nH, nW, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    fig.patch.set_facecolor("#111111")

    if nH == 1 and nW == 1:
        axes = np.array([[axes]])
    elif nH == 1:
        axes = np.array([axes])
    elif nW == 1:
        axes = np.array([[a] for a in axes])

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#555555")

    horizons_sorted = np.sort(horizons.astype(int))
    h_to_row = {int(h): nH - 1 - ri for ri, h in enumerate(horizons_sorted)}

    last_im = None
    for wi, w in enumerate(windows.astype(int)):
        for h in horizons_sorted:
            ri = h_to_row[int(h)]
            ax = axes[ri, wi]
            cell = df_bin[(df_bin["window_index"] == w) & (df_bin["n_candles_forward"] == int(h))]
            med = _pivot_metric(cell, targets, stops, "median")
            mean = _pivot_metric(cell, targets, stops, "mean")
            std = _pivot_metric(cell, targets, stops, "std")
            cnt = _pivot_metric(cell, targets, stops, "count")

            bad = (~np.isfinite(cnt)) | (cnt < min_count)
            med_masked = med.copy()
            med_masked[bad] = np.nan

            im = ax.imshow(med_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            last_im = im

            lbl = str(int(min_count))
            for ti in range(targets.size):
                for sj in range(stops.size):
                    if bad[ti, sj]:
                        ax.text(
                            sj,
                            ti,
                            f"<{lbl}",
                            ha="center",
                            va="center",
                            color="#dddddd",
                            fontsize=inner_fs,
                            linespacing=1.05,
                        )
                    else:
                        txt = (
                            f"m={mean[ti, sj]:.3f}\n"
                            f"med={med[ti, sj]:.3f}\n"
                            f"sd={std[ti, sj]:.3f}\n"
                            f"n={int(cnt[ti, sj])}"
                        )
                        ax.text(
                            sj,
                            ti,
                            txt,
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=inner_fs,
                            linespacing=1.02,
                        )

            ax.set_xticks(np.arange(stops.size))
            ax.set_yticks(np.arange(targets.size))
            if ri == nH - 1:
                ax.set_xticklabels([f"{x:g}" for x in stops], fontsize=tick_fs)
                ax.set_xlabel("stop", fontsize=tick_fs + 1.0, labelpad=8)
            else:
                ax.set_xticklabels([])
            if wi == 0:
                ax.set_yticklabels([f"{x:g}" for x in targets], fontsize=tick_fs)
                ax.set_ylabel("target", fontsize=tick_fs + 1.0, labelpad=8)
            else:
                ax.set_yticklabels([])

            if wi == 0:
                ax.set_ylabel(f"H={int(h)}\n" + ax.get_ylabel(), fontsize=tick_fs + 1.0)
            if ri == nH - 1:
                ax.set_title(f"w={int(w)}", fontsize=coltitle_fs, pad=12)

    supt = (
        f"{title_prefix} | bin_id={bin_id} (spread={spread_i}, accel={accel_j}) | "
        f"color=median | symmetric vmin/vmax this figure | gray: n<{min_count}"
    )
    fig.suptitle(supt, fontsize=supt_fs, y=0.998)
    fig.subplots_adjust(
        left=0.09,
        right=0.84,
        top=0.91,
        bottom=0.06,
        wspace=0.42 + 0.015 * max(0, nW - 5),
        hspace=0.48 + 0.02 * max(0, nH - 8),
    )
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            fraction=0.045,
            pad=0.08,
            shrink=0.86,
        )
        cbar.set_label("median", fontsize=tick_fs + 0.5)
        cbar.ax.tick_params(labelsize=tick_fs)
    fig.savefig(
        out_path,
        dpi=200,
        facecolor=fig.get_facecolor(),
        bbox_inches=None,
        pad_inches=0.05,
    )
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per bin_id: window (x) × horizon (y) grid of target×stop heatmaps "
        "(heatmap2 drill-down). Optional spread/accel bin filters."
    )
    p.add_argument("run_root", help="Main backtest folder, e.g. output/EUR_USD_2023-01-01_2024-12-31")
    p.add_argument("--timeframe_minutes", type=int, required=True, help="e.g. 5 for folder 5m")
    p.add_argument("--lookback", type=int, required=True)
    p.add_argument("--atr", type=int, required=True)
    p.add_argument("--direction", default="long", choices=["long", "short"])
    p.add_argument("--min-count", type=int, default=30)
    p.add_argument(
        "--spread-bins",
        default=None,
        help="Comma-separated spread_bin indices to include (omit for all).",
    )
    p.add_argument(
        "--accel-bins",
        default=None,
        help="Comma-separated norm_accel_bin indices to include (omit for all).",
    )
    p.add_argument(
        "--out-dir",
        default="analysis_heatmap2",
        help="Subfolder under run_root (sibling to Sample CSV).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Skip sanity-check print of max count per window.",
    )
    p.add_argument(
        "--combine-long-short",
        action="store_true",
        help="Pool long and short when k,W match and paired edges pass atol/rtol.",
    )
    p.add_argument("--edge-atol", type=float, default=1e-5)
    p.add_argument(
        "--edge-rtol",
        type=float,
        default=0.02,
        help="tol = atol + rtol * max(|L|,|S|) per edge element.",
    )
    args = p.parse_args()

    plt.style.use("dark_background")

    run_root = os.path.abspath(args.run_root)
    tf_name = _tf_folder_name(args.timeframe_minutes)
    tf_path = os.path.join(run_root, tf_name)
    if not os.path.isdir(tf_path):
        raise SystemExit(f"No timeframe folder {tf_path!r}")

    atr_dir = os.path.join(tf_path, f"lookback_{args.lookback}", f"atr_{args.atr}")
    if not os.path.isdir(atr_dir):
        raise SystemExit(f"No atr folder {atr_dir!r}")

    spread_keep = _parse_int_list(args.spread_bins)
    accel_keep = _parse_int_list(args.accel_bins)

    out_base = os.path.join(run_root, args.out_dir, tf_name)
    os.makedirs(out_base, exist_ok=True)

    try:
        df = _load_parquet_in_atr_dir(atr_dir)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    combine = bool(args.combine_long_short)
    if combine:
        if not args.quiet:
            print("Note: --combine-long-short active; ignoring --direction for row filter.")
        long_d = load_edge_dict(atr_dir, "long")
        short_d = load_edge_dict(atr_dir, "short")
        ok, diag = evaluate_long_short_edges(long_d, short_d, float(args.edge_atol), float(args.edge_rtol))
        if not ok:
            raise SystemExit(
                f"--combine-long-short rejected: {diag.get('reason', 'edge mismatch')}. Diagnostics: {diag}"
            )
        merged = build_merged_edge_payload(long_d, short_d, diag)
        meta_path = os.path.join(
            out_base,
            f"edge_combine_meta_L-{args.lookback}_ATR-{args.atr}.json",
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f"wrote {meta_path}")
        k = int(long_d["k"])
        df = df[df["direction"].isin(["long", "short"])].copy()
        dir_slug = "combined"
    else:
        k = _load_k_from_edge(atr_dir, args.direction)
        df = df[df["direction"] == args.direction].copy()
        dir_slug = args.direction

    if df.empty:
        raise SystemExit("No rows for selected direction(s).")

    if not args.quiet:
        mx = df.groupby("window_index", sort=True)["count"].max()
        sm = df.groupby("window_index", sort=True)["count"].sum()
        print(
            "Window split sanity: max cell count per window_index "
            "(sparse regimes in later windows are common; not necessarily a plotting bug)."
        )
        for w in mx.index:
            print(f"  window {int(w):>3}: max_count={int(mx[w]):>6}  sum_counts={int(sm[w]):>9}")

    targets = np.sort(df["target_mult"].dropna().unique()).astype(np.float64)
    stops = np.sort(df["stop_mult"].dropna().unique()).astype(np.float64)
    windows = np.sort(df["window_index"].dropna().unique()).astype(int)
    horizons = np.sort(df["n_candles_forward"].dropna().unique()).astype(int)

    sidecar = df.copy()
    sidecar["pass_min_count"] = sidecar["count"] >= args.min_count
    sidecar["bin_id"] = sidecar["spread_bin"].astype(np.int64) * int(k) + sidecar["norm_accel_bin"].astype(
        np.int64
    )
    sidecar_path = os.path.join(
        out_base,
        f"stats_dir-{dir_slug}_L-{args.lookback}_ATR-{args.atr}.csv",
    )
    sidecar.to_csv(sidecar_path, index=False)
    print(f"wrote {sidecar_path}")

    title_prefix = f"{tf_name} L={args.lookback} ATR={args.atr} dir={dir_slug}"
    n_bins = k * k

    for bin_id in range(n_bins):
        spread_i = bin_id // k
        accel_j = bin_id % k
        if spread_keep is not None and spread_i not in spread_keep:
            continue
        if accel_keep is not None and accel_j not in accel_keep:
            continue

        df_bin = df[(df["spread_bin"] == spread_i) & (df["norm_accel_bin"] == accel_j)].copy()
        if df_bin.empty:
            continue

        ok = df_bin["count"] >= args.min_count
        med_vals = df_bin.loc[ok, "median"].to_numpy(dtype=np.float64)
        if med_vals.size == 0:
            vlim = 1.0
        else:
            vlim = float(np.nanmax(np.abs(med_vals)))
            if not np.isfinite(vlim) or vlim <= 0:
                vlim = 1.0
        vmin, vmax = -vlim, vlim

        out_png = os.path.join(
            out_base,
            f"bin_{bin_id:03d}_spread-{spread_i}_accel-{accel_j}_dir-{dir_slug}_L-{args.lookback}_ATR-{args.atr}.png",
        )
        _plot_bin_grid(
            df_bin,
            bin_id,
            spread_i,
            accel_j,
            windows,
            horizons,
            targets,
            stops,
            int(args.min_count),
            vmin,
            vmax,
            out_png,
            title_prefix,
        )
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
