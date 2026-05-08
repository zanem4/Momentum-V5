# Joint k×k grid: each cell = target×stop heatmap. Windows aggregated per horizon (count sum,
# count-weighted mean, pooled std, count-weighted avg of medians). One PNG per horizon; universal
# symmetric color scale per image.

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


def _pool_over_windows(sub: pd.DataFrame) -> pd.Series:
    """
    Aggregates rows that share the same (horizon, spread, accel, target, stop) across windows.
    count: sum; mean: count-weighted; std: pooled sample std from group means/stds/counts;
    median: count-weighted average of window medians (approximation without raw trades).
    """
    n = sub["count"].to_numpy(dtype=np.float64)
    mu = sub["mean"].to_numpy(dtype=np.float64)
    md = sub["median"].to_numpy(dtype=np.float64)
    sd = sub["std"].to_numpy(dtype=np.float64)
    N = float(np.nansum(n))
    if N < 1.0:
        return pd.Series({"count": 0, "mean": np.nan, "median": np.nan, "std": np.nan})

    ok_m = np.isfinite(mu) & (n > 0)
    if np.any(ok_m):
        gmean = float(np.sum(n[ok_m] * mu[ok_m]) / np.sum(n[ok_m]))
    else:
        gmean = float("nan")

    SS_w = 0.0
    for i in range(len(n)):
        ni = n[i]
        if ni >= 2.0 and np.isfinite(sd[i]):
            SS_w += (ni - 1.0) * (sd[i] ** 2)

    SS_b = 0.0
    if np.isfinite(gmean):
        for i in range(len(n)):
            ni = n[i]
            if ni > 0.0 and np.isfinite(mu[i]):
                SS_b += ni * (mu[i] - gmean) ** 2

    if N > 1.0 and np.isfinite(gmean):
        var = (SS_w + SS_b) / (N - 1.0)
        pst = float(np.sqrt(max(var, 0.0)))
    else:
        pst = float("nan")

    ok_md = np.isfinite(md) & (n > 0)
    if np.any(ok_md) and np.sum(n[ok_md]) > 0:
        wmed = float(np.average(md[ok_md], weights=n[ok_md]))
    else:
        wmed = float("nan")

    return pd.Series(
        {
            "count": int(N) if np.isfinite(N) else 0,
            "mean": round(gmean, 3) if np.isfinite(gmean) else np.nan,
            "median": round(wmed, 3) if np.isfinite(wmed) else np.nan,
            "std": round(pst, 3) if np.isfinite(pst) else np.nan,
        }
    )


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


def _plot_joint_grid(
    agg_h: pd.DataFrame,
    horizon: int,
    spread_bins: np.ndarray,
    accel_bins: np.ndarray,
    targets: np.ndarray,
    stops: np.ndarray,
    min_count: int,
    vmin: float,
    vmax: float,
    out_path: str,
    title_prefix: str,
) -> None:
    kx, ky = spread_bins.size, accel_bins.size
    nt, ns = int(targets.size), int(stops.size)
    # Outer grid: inches per (spread×accel) cell — primary knob for inner target×stop cell size.
    cell_w_in = max(5.4, min(8.8, 86.0 / max(kx, 1)))
    cell_h_in = max(5.4, min(8.8, 86.0 / max(ky, 1)))
    fig_w = max(14.0, cell_w_in * kx + 1.8)
    fig_h = max(13.0, cell_h_in * ky + 1.8)
    # Usable fraction of each outer subplot for the imshow region (rest = titles/ticks).
    frac = 0.62
    inner_w_in = (cell_w_in * frac) / max(ns, 1)
    inner_h_in = (cell_h_in * frac) / max(nt, 1)
    inner_sq_in = float(min(inner_w_in, inner_h_in))
    # Font: vertical budget (~4 lines) and horizontal budget (widest line ~12 chars) to reduce runover.
    fs_vert = (inner_sq_in * 72.0) / 4.5
    fs_horiz = (inner_w_in * 72.0) / 11.5
    inner_fs = max(6.0, min(11.5, min(fs_vert, fs_horiz)))
    tick_fs = max(inner_fs + 0.5, 8.0)
    supt_fs = min(12.5, max(9.0, tick_fs + 1.5))

    fig, axes = plt.subplots(ky, kx, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    fig.patch.set_facecolor("#111111")

    if kx == 1 and ky == 1:
        axes = np.array([[axes]])
    elif ky == 1:
        axes = np.array([axes])
    elif kx == 1:
        axes = np.array([[a] for a in axes])

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#555555")

    last_im = None
    for ai, a in enumerate(accel_bins):
        for si, s in enumerate(spread_bins):
            ax = axes[ai, si]
            cell = agg_h[(agg_h["spread_bin"] == s) & (agg_h["norm_accel_bin"] == a)]
            med = _pivot_metric(cell, targets, stops, "median")
            mean = _pivot_metric(cell, targets, stops, "mean")
            std = _pivot_metric(cell, targets, stops, "std")
            cnt = _pivot_metric(cell, targets, stops, "count")

            bad = (~np.isfinite(cnt)) | (cnt < min_count)
            med_masked = med.copy()
            med_masked[bad] = np.nan

            im = ax.imshow(med_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            ax.margins(x=0, y=0)
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
                            linespacing=0.98,
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
                            linespacing=0.96,
                        )

            ax.set_xticks(np.arange(stops.size))
            ax.set_yticks(np.arange(targets.size))
            if ai == ky - 1:
                ax.set_xticklabels([f"{x:g}" for x in stops], fontsize=tick_fs)
                ax.set_xlabel("stop", fontsize=tick_fs + 1.0, labelpad=5)
            else:
                ax.set_xticklabels([])
            if si == 0:
                ax.set_yticklabels([f"{x:g}" for x in targets], fontsize=tick_fs)
                ax.set_ylabel("target", fontsize=tick_fs + 1.0, labelpad=5)
            else:
                ax.set_yticklabels([])

            ax.set_title(f"s={int(s)} a={int(a)}", fontsize=tick_fs, pad=3)

    supt = (
        f"{title_prefix} | H={horizon} | windows=pooled | color=median | "
        f"symmetric vmin/vmax entire figure | gray: n<{min_count} | median≈count-wtd avg of window medians"
    )
    fig.suptitle(supt, fontsize=supt_fs, y=1.0)
    # Tight outer frame; small gaps between outer-grid panels; title and grid close together.
    fig.subplots_adjust(
        left=0.035,
        right=0.915,
        top=0.965,
        bottom=0.028,
        wspace=0.10 + 0.008 * max(0, kx - 8),
        hspace=0.12 + 0.008 * max(0, ky - 8),
    )
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            fraction=0.028,
            pad=0.012,
            shrink=0.98,
        )
        cbar.set_label("median (pooled)", fontsize=tick_fs + 0.5)
        cbar.ax.tick_params(labelsize=tick_fs)
    fig.savefig(
        out_path,
        dpi=160,
        facecolor=fig.get_facecolor(),
        bbox_inches=None,
        pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="k×k joint bins of target×stop heatmaps; aggregate all rolling windows per horizon."
    )
    p.add_argument("run_root", help="Main backtest folder, e.g. output/EUR_USD_2023-01-01_2024-12-31")
    p.add_argument("--timeframe_minutes", type=int, required=True, help="e.g. 5 for folder 5m")
    p.add_argument("--lookback", type=int, required=True)
    p.add_argument("--atr", type=int, required=True)
    p.add_argument("--direction", default="long", choices=["long", "short"])
    p.add_argument("--min-count", type=int, default=30)
    p.add_argument(
        "--out-dir",
        default="analysis_heatmap1",
        help="Subfolder under run_root (sibling to Sample CSV).",
    )
    p.add_argument(
        "--combine-long-short",
        action="store_true",
        help="Pool long and short Parquet rows when k,W match and edge arrays pass atol/rtol gate.",
    )
    p.add_argument(
        "--edge-atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance on paired edge differences (default 1e-5).",
    )
    p.add_argument(
        "--edge-rtol",
        type=float,
        default=0.02,
        help="Relative tolerance scale on paired edges: tol = atol + rtol*max(|L|,|S|).",
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

    out_base = os.path.join(run_root, args.out_dir, tf_name)
    os.makedirs(out_base, exist_ok=True)

    try:
        df = _load_parquet_in_atr_dir(atr_dir)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    combine = bool(args.combine_long_short)
    if combine:
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
        edge_note = (
            f"pooled long+short (edges gated atol={args.edge_atol:g} rtol={args.edge_rtol:g}); "
            "mean pairwise edges in edge_combine_meta JSON"
        )
    else:
        k = _load_k_from_edge(atr_dir, args.direction)
        df = df[df["direction"] == args.direction].copy()
        dir_slug = args.direction
        edge_note = "single direction"

    if df.empty:
        raise SystemExit("No rows for selected direction(s).")

    targets = np.sort(df["target_mult"].dropna().unique()).astype(np.float64)
    stops = np.sort(df["stop_mult"].dropna().unique()).astype(np.float64)
    horizons = np.sort(df["n_candles_forward"].dropna().unique()).astype(int)
    spread_bins = np.arange(k, dtype=int)
    accel_bins = np.arange(k, dtype=int)

    gcols = ["spread_bin", "norm_accel_bin", "target_mult", "stop_mult"]
    agg_parts: list[pd.DataFrame] = []
    for h in horizons:
        dh = df[df["n_candles_forward"] == int(h)]
        if dh.empty:
            continue
        try:
            pooled = dh.groupby(gcols, sort=False).apply(_pool_over_windows, include_groups=False)
        except TypeError:
            pooled = dh.groupby(gcols, sort=False).apply(_pool_over_windows)
        pooled = pooled.reset_index()
        pooled["n_candles_forward"] = int(h)
        agg_parts.append(pooled)
    agg_df = pd.concat(agg_parts, ignore_index=True) if agg_parts else pd.DataFrame()

    if not agg_df.empty:
        agg_df.insert(0, "direction", dir_slug)
        agg_df["pass_min_count"] = agg_df["count"] >= args.min_count
        agg_df["bin_id"] = agg_df["spread_bin"].astype(np.int64) * int(k) + agg_df["norm_accel_bin"].astype(
            np.int64
        )
        agg_df["note"] = (
            "mean=wtd; std=pooled; median≈wtd avg window medians; count=sum windows"
            + ("+dirs" if combine else "")
            + " | "
            + edge_note
        )

    sidecar_path = os.path.join(
        out_base,
        f"stats_joint_pooled_{dir_slug}_L-{args.lookback}_ATR-{args.atr}.csv",
    )
    agg_df.to_csv(sidecar_path, index=False)
    print(f"wrote {sidecar_path}")

    title_prefix = f"{tf_name} L={args.lookback} ATR={args.atr} dir={dir_slug} ({edge_note})"
    mc = int(args.min_count)

    for h in horizons:
        agg_h = agg_df[agg_df["n_candles_forward"] == int(h)] if not agg_df.empty else pd.DataFrame()
        if agg_h.empty:
            continue

        ok = agg_h["count"] >= mc
        med_vals = agg_h.loc[ok, "median"].to_numpy(dtype=np.float64)
        if med_vals.size == 0:
            vlim = 1.0
        else:
            vlim = float(np.nanmax(np.abs(med_vals)))
            if not np.isfinite(vlim) or vlim <= 0:
                vlim = 1.0
        vmin, vmax = -vlim, vlim

        out_png = os.path.join(
            out_base,
            f"joint_H-{int(h)}_dir-{dir_slug}_L-{args.lookback}_ATR-{args.atr}.png",
        )
        _plot_joint_grid(
            agg_h,
            int(h),
            spread_bins,
            accel_bins,
            targets,
            stops,
            mc,
            vmin,
            vmax,
            out_png,
            title_prefix,
        )
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()