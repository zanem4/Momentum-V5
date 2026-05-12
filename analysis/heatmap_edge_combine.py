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


def _pairwise_percent_diff(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Per-index percent difference on magnitudes (stable when L and S straddle zero):

    ``||a| - |b|| / max(mean(|a|, |b|), eps)`` with ``mean(|a|,|b|) = (|a|+|b|)/2`` element-wise.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    aa = np.abs(a)
    bb = np.abs(b)
    m = (aa + bb) / 2.0
    d = np.abs(aa - bb)
    den = np.maximum(m, eps)
    return d / den


def evaluate_long_short_edges(
    long_d: dict[str, Any],
    short_d: dict[str, Any],
    max_pct_diff: float,
) -> tuple[bool, dict[str, Any]]:
    """
    Gate combining long+short when ``k`` and ``W`` match and, for every edge index ``i``,

    ``||L_i| - |S_i|| / max((|L_i|+|S_i|)/2, eps) <= max_pct_diff``

    on both ``edges_spread_delta`` and ``edges_norm_accel`` (long vs short for each).
    ``max_pct_diff`` = 0.10 is 10% (fraction, not 0.1%).
    """
    k_lo = int(long_d["k"])
    k_sh = int(short_d["k"])
    W_lo = int(long_d["W"])
    W_sh = int(short_d["W"])
    thr = float(max_pct_diff)
    diag: dict[str, Any] = {
        "k_long": k_lo,
        "k_short": k_sh,
        "W_long": W_lo,
        "W_short": W_sh,
        "max_pct_diff": thr,
        "max_pct_diff_percent": round(100.0 * thr, 6),
    }

    if k_lo != k_sh:
        diag["reason"] = "k_long != k_short"
        return False, diag
    if W_lo != W_sh:
        diag["reason"] = "W_long != W_short"
        return False, diag
    if not np.isfinite(thr) or thr < 0:
        diag["reason"] = "max_pct_diff must be a non-negative finite number"
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

    pct_sp = _pairwise_percent_diff(es_lo, es_sh)
    pct_ac = _pairwise_percent_diff(ea_lo, ea_sh)

    diag["spread_max_pct_diff"] = float(np.max(pct_sp)) if pct_sp.size else 0.0
    diag["accel_max_pct_diff"] = float(np.max(pct_ac)) if pct_ac.size else 0.0

    ok_sp = bool(np.all(pct_sp <= thr)) if pct_sp.size else True
    ok_ac = bool(np.all(pct_ac <= thr)) if pct_ac.size else True
    passed = ok_sp and ok_ac
    diag["spread_within_tol"] = ok_sp
    diag["accel_within_tol"] = ok_ac
    if not passed:
        diag["reason"] = (
            f"edge percent diff ||L|-|S||/mean(|L|,|S|) exceeds {thr:.4g} "
            f"({100.0 * thr:.4g}%); see spread_max_pct_diff / accel_max_pct_diff"
        )
        if not ok_sp:
            diag["failed_axis"] = "edges_spread_delta"
        elif not ok_ac:
            diag["failed_axis"] = "edges_norm_accel"

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
            "when long/short marginal edges agree per index: "
            "||L|-|S||/mean(|L|,|S|) <= max_pct_diff. Not a formal proof of identical joint distributions."
        ),
    }
    return out
