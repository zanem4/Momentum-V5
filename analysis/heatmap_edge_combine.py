# Shared helpers: compare long vs short quantile edges; optional merge metadata for combined heatmaps.

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def load_edge_dict(atr_dir: str, which: str) -> dict[str, Any]:
    """Load ``long_edge_data.json`` or ``short_edge_data.json`` (slim JSON from save_data)."""
    if which not in ("long", "short"):
        raise ValueError("which must be 'long' or 'short'")
    path = os.path.join(atr_dir, "long_edge_data.json" if which == "long" else "short_edge_data.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _paired_ttest_rel(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    try:
        from scipy import stats  # type: ignore[import-untyped]
    except ImportError:
        return None, None
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return None, None
    t_stat, p_two = stats.ttest_rel(a, b)
    return float(t_stat), float(p_two)


def evaluate_long_short_edges(
    long_d: dict[str, Any],
    short_d: dict[str, Any],
    atol: float,
    rtol: float,
) -> tuple[bool, dict[str, Any]]:
    """
    Gate combining long+short on ordinal bin alignment:

    1. Require ``k`` and ``W`` equal.
    2. Element-wise tolerance: ``|L-S| <= atol + rtol * max(|L|,|S|,eps)`` on both edge arrays.
    3. Paired ``ttest_rel`` on spread and accel edge vectors when SciPy is installed (report-only).

    Returns (passed_gate, diagnostics).
    """
    k_lo = int(long_d["k"])
    k_sh = int(short_d["k"])
    W_lo = int(long_d["W"])
    W_sh = int(short_d["W"])
    eps = 1e-12
    diag: dict[str, Any] = {
        "k_long": k_lo,
        "k_short": k_sh,
        "W_long": W_lo,
        "W_short": W_sh,
        "atol": float(atol),
        "rtol": float(rtol),
    }

    if k_lo != k_sh:
        diag["reason"] = "k_long != k_short"
        return False, diag
    if W_lo != W_sh:
        diag["reason"] = "W_long != W_short"
        return False, diag

    es_lo = np.asarray(long_d["edges_spread_delta"], dtype=np.float64).ravel()
    es_sh = np.asarray(short_d["edges_spread_delta"], dtype=np.float64).ravel()
    ea_lo = np.asarray(long_d["edges_norm_accel"], dtype=np.float64).ravel()
    ea_sh = np.asarray(short_d["edges_norm_accel"], dtype=np.float64).ravel()

    if es_lo.shape != es_sh.shape:
        diag["reason"] = "edges_spread_delta length mismatch"
        return False, diag
    if ea_lo.shape != ea_sh.shape:
        diag["reason"] = "edges_norm_accel length mismatch"
        return False, diag

    d_sp = np.abs(es_lo - es_sh)
    d_ac = np.abs(ea_lo - ea_sh)
    s_sp = np.maximum(np.maximum(np.abs(es_lo), np.abs(es_sh)), eps)
    s_ac = np.maximum(np.maximum(np.abs(ea_lo), np.abs(ea_sh)), eps)
    tol_sp = float(atol) + float(rtol) * s_sp
    tol_ac = float(atol) + float(rtol) * s_ac

    diag["spread_max_abs_diff"] = float(np.max(d_sp)) if d_sp.size else 0.0
    diag["accel_max_abs_diff"] = float(np.max(d_ac)) if d_ac.size else 0.0
    diag["spread_max_abs_diff_over_tol"] = float(np.max(d_sp / tol_sp)) if d_sp.size else 0.0
    diag["accel_max_abs_diff_over_tol"] = float(np.max(d_ac / tol_ac)) if d_ac.size else 0.0

    ok_sp = bool(np.all(d_sp <= tol_sp)) if d_sp.size else True
    ok_ac = bool(np.all(d_ac <= tol_ac)) if d_ac.size else True
    passed = ok_sp and ok_ac
    diag["spread_within_tol"] = ok_sp
    diag["accel_within_tol"] = ok_ac
    if not passed:
        diag["reason"] = "edge element-wise tolerance failed (spread and/or accel)"
        if not ok_sp:
            diag["failed_axis"] = "edges_spread_delta"
        elif not ok_ac:
            diag["failed_axis"] = "edges_norm_accel"

    # Supplementary paired t-tests (do not override the tolerance gate)
    t_sp, p_sp = _paired_ttest_rel(es_lo, es_sh)
    t_ac, p_ac = _paired_ttest_rel(ea_lo, ea_sh)
    diag["paired_ttest"] = {
        "edges_spread_delta": {"t_statistic": t_sp, "pvalue_two_sided": p_sp},
        "edges_norm_accel": {"t_statistic": t_ac, "pvalue_two_sided": p_ac},
        "note": "Informational; install scipy for values. Does not replace tolerance gate.",
    }

    return passed, diag


def build_merged_edge_payload(
    long_d: dict[str, Any],
    short_d: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Arithmetic mean of pairwise edges + shared metadata (for JSON sidecar)."""
    es_lo = np.asarray(long_d["edges_spread_delta"], dtype=np.float64).ravel()
    es_sh = np.asarray(short_d["edges_spread_delta"], dtype=np.float64).ravel()
    ea_lo = np.asarray(long_d["edges_norm_accel"], dtype=np.float64).ravel()
    ea_sh = np.asarray(short_d["edges_norm_accel"], dtype=np.float64).ravel()
    merged_sp = (es_lo + es_sh) / 2.0
    merged_ac = (ea_lo + ea_sh) / 2.0
    out: dict[str, Any] = {
        "combine_long_short": True,
        "k": int(long_d["k"]),
        "W": int(long_d["W"]),
        "edges_spread_delta_mean_pairwise": [float(x) for x in merged_sp],
        "edges_norm_accel_mean_pairwise": [float(x) for x in merged_ac],
        "window_bounds": long_d.get("window_bounds"),
        "window_length_seconds": long_d.get("window_length_seconds"),
        "window_step_seconds": long_d.get("window_step_seconds"),
        "window_bounds_note": long_d.get("window_bounds_note"),
        "diagnostics": diagnostics,
        "interpretation_note": (
            "Pooling Parquet rows assumes spread_bin and norm_accel_bin ordinals are comparable "
            "when long/short edges are close under atol/rtol. Not a formal proof of identical "
            "joint distributions."
        ),
    }
    return out
