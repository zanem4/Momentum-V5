# Generic heatmap: columns = window_index, rows = array_y slice; each cell = heatmap_y × heatmap_x.
# Aggregates all dimensions other than {window_index, array_y, heatmap_x, heatmap_y} within each cell
# (same pooling as heatmap1). Outputs under run_root/analysis/generic/; timeframe is in each filename (tf-5m).
#
# Discovery: exactly one pnl_summaries_forward_*.parquet per atr folder. Under each {tf}m, either
# exactly one lookback_* with one atr_* inside, or pass --lookback N --atr M to use lookback_N/atr_M.
# Column titles use window_bounds from long_edge_data.json (or direction JSON); spread_bin rows use
# marginal intervals from edges_spread_delta (same semantics as strategy.quantile.assign_marginal_bins).

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import os
import re
import sys

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


_TF_DIR_RE = re.compile(r"^(\d+)m$")


def _marginal_interval_str(edges: np.ndarray, k: int, bin_idx: int) -> str:
    """
    Human-readable bounds for marginal bin index, matching strategy.quantile.assign_marginal_bins:
    bin 0: (-inf, e0]; interior (e_{i-1}, e_i]; last (e_{k-2}, +inf).
    """
    k = int(k)
    bin_idx = int(bin_idx)
    if k <= 0 or bin_idx < 0 or bin_idx >= k:
        return "?"
    if k == 1:
        return "(-inf, +inf)"
    edges = np.asarray(edges, dtype=np.float64).ravel()
    if edges.size != k - 1:
        return "?"
    if bin_idx == 0:
        return f"(-inf, {edges[0]:.6g}]"
    if bin_idx == k - 1:
        return f"({edges[k - 2]:.6g}, +inf)"
    return f"({edges[bin_idx - 1]:.6g}, {edges[bin_idx]:.6g}]"


def _window_range_title_utc(bounds: list[dict], window_index: int) -> str:
    """Two-line UTC title from edge JSON window_bounds (tz-aware datetimes)."""
    for row in bounds:
        if int(row["window_index"]) != int(window_index):
            continue
        t0 = datetime.fromtimestamp(float(row["window_t0_utc"]), tz=timezone.utc)
        t1 = datetime.fromtimestamp(float(row["window_t1_utc"]), tz=timezone.utc)
        return f"{t0:%Y-%m-%d %H:%M} UTC →\n{t1:%Y-%m-%d %H:%M} UTC"
    raise KeyError(f"window_index {window_index} not in window_bounds")


def _array_y_row_caption(array_y: str, ay: float, k: int, edge_dict: dict) -> str:
    """spread_bin / norm_accel_bin show marginal interval; else ay=value."""
    bi = int(round(float(ay)))
    if array_y == "spread_bin":
        e = np.asarray(edge_dict["edges_spread_delta"], dtype=np.float64)
        interval = _marginal_interval_str(e, k, bi)
        return f"spread_bin={bi} ∈ {interval}"
    if array_y == "norm_accel_bin":
        e = np.asarray(edge_dict["edges_norm_accel"], dtype=np.float64)
        interval = _marginal_interval_str(e, k, bi)
        return f"norm_accel_bin={bi} ∈ {interval}"
    return f"{array_y}={ay:g}"


def _load_edge_metadata_for_plot(atr_dir: str, combine: bool, direction: str) -> dict:
    """Same JSON used by simulations: window_bounds + marginal edges + k."""
    if combine:
        path = os.path.join(atr_dir, "long_edge_data.json")
    else:
        path = os.path.join(
            atr_dir, "long_edge_data.json" if direction == "long" else "short_edge_data.json"
        )
    if not os.path.isfile(path):
        print(f"Missing edge JSON for metadata: {path!r}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    req = ("window_bounds", "k", "edges_spread_delta", "edges_norm_accel")
    for key in req:
        if key not in d:
            print(f"Edge JSON {path!r} missing {key!r}", file=sys.stderr)
            sys.exit(1)
    return d


def _parse_lookback_atr_from_path(atr_dir: str) -> tuple[int, int]:
    """atr_dir = .../lookback_L/atr_A"""
    base = os.path.basename(atr_dir.rstrip(os.sep))
    parent = os.path.basename(os.path.dirname(atr_dir))
    am = re.fullmatch(r"atr_(\d+)", base)
    lm = re.fullmatch(r"lookback_(\d+)", parent)
    if not am or not lm:
        raise ValueError(f"Cannot parse lookback/atr from path: {atr_dir!r}")
    return int(lm.group(1)), int(am.group(1))


def _discover_atr_dir(tf_path: str, lookback: int | None, atr: int | None) -> str:
    """
    Resolve .../lookback_L/atr_A. If lookback and atr are set, use that path only (must exist).
    Otherwise require exactly one lookback_* and one atr_* under it (strict).
    """
    if lookback is not None and atr is not None:
        atr_dir = os.path.join(tf_path, f"lookback_{int(lookback)}", f"atr_{int(atr)}")
        if not os.path.isdir(atr_dir):
            print(f"Missing {atr_dir!r} (use --lookback / --atr).", file=sys.stderr)
            sys.exit(1)
        return atr_dir

    lb_glob = sorted(glob.glob(os.path.join(tf_path, "lookback_*")))
    lb_dirs = [p for p in lb_glob if os.path.isdir(p)]
    if len(lb_dirs) != 1:
        msg = (
            f"Expected exactly one lookback_* directory under {tf_path!r}, "
            f"found {len(lb_dirs)}: {lb_dirs!r}. Pass --lookback and --atr to select one."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)
    atr_glob = sorted(glob.glob(os.path.join(lb_dirs[0], "atr_*")))
    atr_dirs = [p for p in atr_glob if os.path.isdir(p)]
    if len(atr_dirs) != 1:
        msg = (
            f"Expected exactly one atr_* directory under {lb_dirs[0]!r}, "
            f"found {len(atr_dirs)}: {atr_dirs!r}. Pass --lookback and --atr to select one."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)
    return atr_dirs[0]


def _load_parquet_strict(atr_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(atr_dir, "pnl_summaries_forward_*.parquet")))
    if len(paths) != 1:
        msg = (
            f"Expected exactly one pnl_summaries_forward_*.parquet in {atr_dir!r}, "
            f"found {len(paths)}: {paths!r}"
        )
        print(msg, file=sys.stderr)
        sys.exit(1)
    return pd.read_parquet(paths[0])


def _pool_over_extra(sub: pd.DataFrame) -> pd.Series:
    """Same aggregation as heatmap1._pool_over_windows (pool rows sharing one visual cell)."""
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


_COLOR_SCORE_EPS = 1e-8


def _heatmap_color_score(mean: np.ndarray, median: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Unitless colormap score:
        min(|mean|,|median|) / (stdev + ε)  −  |mean − median| / (min(|mean|,|median|) + ε)
    When mean > 0 and median > 0, min(|.|) equals min(mean, median), matching the intended
    gap penalty denominator; ε avoids division by zero. Cells with mean ≤ 0 or median ≤ 0 → NaN.
    """
    mean = np.asarray(mean, dtype=np.float64)
    median = np.asarray(median, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    eps = _COLOR_SCORE_EPS
    # with commission at $5/lot, need mean return of 0.5 pips or higher to survive final cost (@ $10 profit per lot)
    mean_minimum_penalty = 10 * (mean - 0.5)
    # positive median is necessary
    median_sign_penalty = (median - np.abs(median)) ** 2
    # mean - median approximates tail, so mean and median should be in as best agreement as possible
    tail_penalty = (mean - median) ** 2
    out = median + mean_minimum_penalty - median_sign_penalty - tail_penalty
    bad = ~(np.isfinite(mean) & np.isfinite(median) & np.isfinite(std))
    out = out.copy()
    out[bad] = np.nan
    return out


def _pivot_metric(
    cell_df: pd.DataFrame,
    y_vals: np.ndarray,
    x_vals: np.ndarray,
    heatmap_y: str,
    heatmap_x: str,
    col: str,
) -> np.ndarray:
    out = np.full((y_vals.size, x_vals.size), np.nan, dtype=np.float64)
    if cell_df.empty:
        return out
    tmp = cell_df.pivot_table(index=heatmap_y, columns=heatmap_x, values=col, aggfunc="first")
    for ti, t in enumerate(y_vals):
        if t not in tmp.index:
            continue
        for si, s in enumerate(x_vals):
            if s in tmp.columns:
                val = tmp.loc[t, s]
                out[ti, si] = float(val) if pd.notna(val) else np.nan
    return out


def _validate_columns(df: pd.DataFrame, names: list[str], ctx: str) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        msg = f"{ctx}: missing column(s) {missing!r}; available: {sorted(df.columns.tolist())!r}"
        print(msg, file=sys.stderr)
        sys.exit(1)


def _aggregate_for_heatmap(
    df: pd.DataFrame,
    array_x: str,
    array_y: str,
    heatmap_x: str,
    heatmap_y: str,
) -> pd.DataFrame:
    keys = [array_x, array_y, heatmap_y, heatmap_x]
    try:
        pooled = df.groupby(keys, sort=False).apply(_pool_over_extra, include_groups=False)
    except TypeError:
        pooled = df.groupby(keys, sort=False).apply(_pool_over_extra)
    return pooled.reset_index()


def _plot_generic_figure(
    agg: pd.DataFrame,
    array_x: str,
    array_y: str,
    heatmap_x: str,
    heatmap_y: str,
    windows: np.ndarray,
    ay_levels: np.ndarray,
    hx_vals: np.ndarray,
    hy_vals: np.ndarray,
    min_count: int,
    vmin: float,
    vmax: float,
    out_path: str,
    suptitle: str,
    window_title_for: dict[int, str],
    row_caption_for_ay: dict[float, str],
) -> None:
    n_w = int(windows.size)
    n_ay = int(ay_levels.size)
    nt = int(hy_vals.size)
    ns = int(hx_vals.size)
    # Outer panel inches per subplot — match heatmap1.py (large cells so inner target×stop stays readable).
    cell_w_in = max(5.4, min(8.8, 86.0 / max(n_w, 1)))
    cell_h_in = max(5.4, min(8.8, 86.0 / max(n_ay, 1)))
    fig_w = max(14.0, cell_w_in * n_w + 1.8)
    fig_h = max(13.0, cell_h_in * n_ay + 1.8)
    # Usable fraction of each outer subplot for imshow (same as heatmap1); scale inner text to inner grid.
    frac = 0.62
    inner_w_in = (cell_w_in * frac) / max(ns, 1)
    inner_h_in = (cell_h_in * frac) / max(nt, 1)
    inner_sq_in = float(min(inner_w_in, inner_h_in))
    fs_vert = (inner_sq_in * 72.0) / 4.5
    fs_horiz = (inner_w_in * 72.0) / 11.5
    inner_fs = max(6.0, min(11.5, min(fs_vert, fs_horiz)))
    tick_fs = max(inner_fs + 0.5, 8.0)
    coltitle_fs = inner_fs + 1.2
    supt_fs = min(12.5, max(9.0, tick_fs + 1.5))

    # sharex/sharey False avoids matplotlib stripping tick labels on shared axes; we still only
    # draw tick *labels* on bottom row / left column (facet style) to limit clutter.
    fig, axes = plt.subplots(n_ay, n_w, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    fig.patch.set_facecolor("#111111")

    if n_ay == 1 and n_w == 1:
        axes = np.array([[axes]])
    elif n_ay == 1:
        axes = np.array([axes])
    elif n_w == 1:
        axes = np.array([[a] for a in axes])

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#555555")

    ay_sorted = np.sort(ay_levels.astype(float))
    ay_to_row = {float(ay_sorted[i]): n_ay - 1 - i for i in range(n_ay)}
    tick_fs_cell = max(tick_fs - 0.5, 7.0)
    xtick_labels = [f"{heatmap_x}={v:g}" for v in hx_vals]
    ytick_labels = [f"{heatmap_y}={v:g}" for v in hy_vals]

    last_im = None
    for wi, w in enumerate(windows.astype(int)):
        for ay in ay_sorted:
            ri = ay_to_row[float(ay)]
            ax = axes[ri, wi]
            mask = (agg[array_x].astype(int) == int(w)) & np.isclose(
                agg[array_y].astype(float), float(ay), rtol=0.0, atol=0.0
            )
            cell = agg.loc[mask].copy()

            med = _pivot_metric(cell, hy_vals, hx_vals, heatmap_y, heatmap_x, "median")
            mean = _pivot_metric(cell, hy_vals, hx_vals, heatmap_y, heatmap_x, "mean")
            std = _pivot_metric(cell, hy_vals, hx_vals, heatmap_y, heatmap_x, "std")
            cnt = _pivot_metric(cell, hy_vals, hx_vals, heatmap_y, heatmap_x, "count")

            bad = (~np.isfinite(cnt)) | (cnt < min_count)
            clr = _heatmap_color_score(mean, med, std)
            clr_masked = clr.copy()
            clr_masked[bad] = np.nan

            im = ax.imshow(clr_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            ax.margins(x=0, y=0)
            last_im = im

            lbl = str(int(min_count))
            for ti in range(hy_vals.size):
                for sj in range(hx_vals.size):
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

            ax.set_xticks(np.arange(hx_vals.size))
            ax.set_yticks(np.arange(hy_vals.size))
            cap = row_caption_for_ay[float(ay)]
            if ri == n_ay - 1:
                ax.set_xticklabels(xtick_labels, fontsize=tick_fs_cell, rotation=45, ha="right")
                ax.set_xlabel(heatmap_x, fontsize=tick_fs + 1.0, labelpad=5)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            if wi == 0:
                ax.set_yticklabels(ytick_labels, fontsize=tick_fs_cell)
                ax.set_ylabel(f"{cap}\n{heatmap_y}", fontsize=tick_fs + 1.0, labelpad=5)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

            if ri == n_ay - 1:
                ax.set_title(window_title_for[int(w)], fontsize=min(coltitle_fs, 10.0), pad=10)

    # Subplot margins match heatmap1; slightly wider left for array_y + heatmap_y two-line y labels.
    fig.suptitle(suptitle, fontsize=supt_fs, y=1.0)
    fig.subplots_adjust(
        left=0.08,
        right=0.915,
        top=0.965,
        bottom=0.06 + 0.02 * min(ns, 12),
        wspace=0.10 + 0.008 * max(0, n_w - 8),
        hspace=0.14 + 0.008 * max(0, n_ay - 8),
    )
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            fraction=0.028,
            pad=0.012,
            shrink=0.98,
        )
        cbar.set_label("color score (unitless)", fontsize=tick_fs + 0.5)
        cbar.ax.tick_params(labelsize=tick_fs)
    fig.savefig(
        out_path,
        dpi=200,
        facecolor=fig.get_facecolor(),
        bbox_inches=None,
        pad_inches=0.05,
    )
    plt.close(fig)


def _run_one_timeframe(
    tf_name: str,
    tf_path: str,
    heatmap_x: str,
    heatmap_y: str,
    array_x: str,
    array_y: str,
    combine: bool,
    direction: str,
    edge_max_pct_diff: float,
    min_count: int,
    run_root: str,
    lookback: int | None,
    atr: int | None,
) -> None:
    atr_dir = _discover_atr_dir(tf_path, lookback, atr)
    lookback_n, atr_n = _parse_lookback_atr_from_path(atr_dir)

    try:
        df = _load_parquet_strict(atr_dir)
    except Exception as e:
        print(f"Failed to load parquet from {atr_dir!r}: {e}", file=sys.stderr)
        sys.exit(1)

    req = ["count", "mean", "std", "median", "direction", heatmap_x, heatmap_y, array_x, array_y]
    _validate_columns(df, req, f"{tf_name} parquet")

    dup_axes = {heatmap_x, heatmap_y, array_x, array_y}
    if len(dup_axes) != 4:
        msg = f"heatmap_x, heatmap_y, array_x, array_y must be four distinct columns; got {dup_axes!r}"
        print(msg, file=sys.stderr)
        sys.exit(1)

    out_base = os.path.join(run_root, "analysis", "generic")
    os.makedirs(out_base, exist_ok=True)

    dir_slug: str
    if combine:
        long_d = load_edge_dict(atr_dir, "long")
        short_d = load_edge_dict(atr_dir, "short")
        ok, diag = evaluate_long_short_edges(long_d, short_d, float(edge_max_pct_diff))
        if not ok:
            print(
                f"{tf_name}: --combine-long-short rejected: {diag.get('reason', 'edge mismatch')}. {diag}",
                file=sys.stderr,
            )
            sys.exit(1)
        merged = build_merged_edge_payload(long_d, short_d, diag)
        meta_path = os.path.join(
            out_base,
            f"edge_combine_meta_tf-{tf_name}_L-{lookback_n}_ATR-{atr_n}.json",
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f"wrote {meta_path}")
        df = df[df["direction"].isin(["long", "short"])].copy()
        dir_slug = "combined"
    else:
        df = df[df["direction"] == direction].copy()
        dir_slug = direction

    if df.empty:
        print(f"{tf_name}: no rows after direction filter.", file=sys.stderr)
        sys.exit(1)

    agg = _aggregate_for_heatmap(df, array_x, array_y, heatmap_x, heatmap_y)
    if agg.empty:
        print(f"{tf_name}: aggregated frame is empty.", file=sys.stderr)
        sys.exit(1)

    sidecar = os.path.join(
        out_base,
        f"stats_generic_tf-{tf_name}_hx-{heatmap_x}_hy-{heatmap_y}_ay-{array_y}_dir-{dir_slug}_L-{lookback_n}_ATR-{atr_n}.csv",
    )
    agg.to_csv(sidecar, index=False)

    windows = np.sort(agg[array_x].dropna().unique().astype(int))
    ay_levels = np.sort(agg[array_y].dropna().unique().astype(float))
    hx_vals = np.sort(agg[heatmap_x].dropna().unique().astype(float))
    hy_vals = np.sort(agg[heatmap_y].dropna().unique().astype(float))

    edge_meta = _load_edge_metadata_for_plot(atr_dir, combine, direction)
    k_meta = int(edge_meta["k"])

    window_title_for: dict[int, str] = {}
    for w in windows.astype(int):
        try:
            window_title_for[int(w)] = _window_range_title_utc(edge_meta["window_bounds"], int(w))
        except KeyError:
            print(
                f"{tf_name}: window_index {int(w)} missing from edge JSON window_bounds.",
                file=sys.stderr,
            )
            sys.exit(1)

    row_caption_for_ay: dict[float, str] = {}
    for ay in ay_levels.astype(float):
        row_caption_for_ay[float(ay)] = _array_y_row_caption(array_y, float(ay), k_meta, edge_meta)

    ok_score = agg["count"] >= min_count
    sub = agg.loc[ok_score, ["mean", "median", "std"]]
    if sub.empty:
        vlim = 1.0
    else:
        sc = _heatmap_color_score(
            sub["mean"].to_numpy(dtype=np.float64),
            sub["median"].to_numpy(dtype=np.float64),
            sub["std"].to_numpy(dtype=np.float64),
        )
        vlim = float(np.nanmax(np.abs(sc)))
        if not np.isfinite(vlim) or vlim <= 0:
            vlim = 1.0
    vmin, vmax = -vlim, vlim

    edge_note = (
        f"pooled L+S edges gated max_pct_diff={edge_max_pct_diff:g}"
        if combine
        else f"single direction={direction}"
    )
    title_prefix = (
        f"{tf_name} L={lookback_n} ATR={atr_n} dir={dir_slug} ({edge_note}) | "
        f"rows={array_y} cols={array_x} cell_y={heatmap_y} cell_x={heatmap_x}"
    )
    supt = (
        f"{title_prefix} | color=min(|m|,|med|)/(σ+ε)-|m-med|/(min(|m|,|med|)+ε) unitless; m>0&med>0 | symmetric vmin/vmax | "
        f"gray: n<{min_count} | pool within cell over other summary dimensions | cells: m, med, sd, n"
    )

    slug = (
        f"generic_tf-{tf_name}_hx-{heatmap_x}_hy-{heatmap_y}_ay-{array_y}_ax-{array_x}_dir-{dir_slug}_L-{lookback_n}_ATR-{atr_n}"
    )
    out_png = os.path.join(out_base, f"{slug}.png")

    _plot_generic_figure(
        agg,
        array_x,
        array_y,
        heatmap_x,
        heatmap_y,
        windows,
        ay_levels,
        hx_vals,
        hy_vals,
        min_count,
        vmin,
        vmax,
        out_png,
        supt,
        window_title_for,
        row_caption_for_ay,
    )
    print(f"wrote {out_png}")
    print(f"wrote {sidecar}")


def main(
    run_root: str,
    heatmap_x: str,
    heatmap_y: str,
    array_x: str,
    array_y: str,
    *,
    combine_long_short: bool = True,
    direction: str = "long",
    edge_max_pct_diff: float = 5.0,
    min_count: int = 30,
    lookback: int | None = None,
    atr: int | None = None,
) -> None:
    run_root = os.path.abspath(run_root)
    if not os.path.isdir(run_root):
        print(f"Not a directory: {run_root!r}", file=sys.stderr)
        sys.exit(1)

    children = sorted(os.listdir(run_root))
    tf_dirs: list[tuple[str, str]] = []
    for name in children:
        m = _TF_DIR_RE.fullmatch(name)
        if not m:
            continue
        path = os.path.join(run_root, name)
        if os.path.isdir(path):
            tf_dirs.append((name, path))

    if not tf_dirs:
        print(
            f"No timeframe folders matching NAME like '5m' under {run_root!r}.",
            file=sys.stderr,
        )
        sys.exit(1)

    plt.style.use("dark_background")
    for tf_name, tf_path in tf_dirs:
        _run_one_timeframe(
            tf_name,
            tf_path,
            heatmap_x,
            heatmap_y,
            array_x,
            array_y,
            combine_long_short,
            direction,
            edge_max_pct_diff,
            min_count,
            run_root,
            lookback,
            atr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic heatmap: rows=array_y, cols=window_index; edit dimensions in this file."
    )
    parser.add_argument(
        "run_root",
        help="Backtest output root (contains 3m, 5m, …).",
    )
    parser.add_argument(
        "--no-combine-long-short",
        action="store_true",
        help="Disable pooling long+short (default: combine on).",
    )
    parser.add_argument(
        "--direction",
        default="long",
        choices=["long", "short"],
        help="Direction when not combining (default long).",
    )
    parser.add_argument(
        "--edge-max-pct-diff",
        type=float,
        default=5.0,
        dest="edge_max_pct_diff",
        help="Gate for combine mode (default 5.0).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=30,
        dest="min_count",
        help="Suppress cells with count below this (default 30).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Select lookback_N under each timeframe (required if multiple lookback_* exist).",
    )
    parser.add_argument(
        "--atr",
        type=int,
        default=None,
        help="Select atr_N under the chosen lookback (required if multiple atr_* exist).",
    )
    cli = parser.parse_args()
    if (cli.lookback is None) ^ (cli.atr is None):
        print("Provide both --lookback and --atr, or neither.", file=sys.stderr)
        sys.exit(1)

    heatmap_x = "stop_mult"
    heatmap_y = "target_mult"
    array_x = "window_index"
    array_y = "norm_accel_bin"

    main(
        cli.run_root,
        heatmap_x,
        heatmap_y,
        array_x,
        array_y,
        combine_long_short=not cli.no_combine_long_short,
        direction=cli.direction,
        edge_max_pct_diff=float(cli.edge_max_pct_diff),
        min_count=int(cli.min_count),
        lookback=cli.lookback,
        atr=cli.atr,
    )
