# Marginal quantile grids & joint bin ids — see utils/METHODOLOGY.md §2–4.

from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_E_MIN = 20
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 25
# Upper bound exclusive: np.arange(3, 8) → W ∈ {3,4,5,6,7}
DEFAULT_W_LOW = 3
DEFAULT_W_HIGH = 8


def choose_windows_and_bins(
    n: int,
    e_min: float = DEFAULT_E_MIN,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    w_low: int = DEFAULT_W_LOW,
    w_high: int = DEFAULT_W_HIGH,
) -> Optional[tuple[int, int]]:
    """
    Pick rolling-window count W and marginal bin count k from setup cardinality.

    Target (uniform fantasy): n / (W * k**2) ≈ e_min  ⇒  k ≈ √(n / (W * e_min)).

    Prefer larger W lexicographically, then larger k, among feasible pairs.

    Returns ``None`` if no pair achieves ``k >= k_min`` (_INSUFFICIENT SAMPLE for intended grid).
    """
    if n <= 0:
        return None

    candidates: list[tuple[int, int]] = []
    for W in range(w_low, w_high):
        k = int(np.sqrt(n / (W * e_min)))
        if k >= k_min:
            candidates.append((W, min(k, k_max)))

    if not candidates:
        return None

    return max(candidates, key=lambda x: (x[0], x[1]))


def choose_windows_and_bins_or_coarse(
    n: int,
    e_min: float = DEFAULT_E_MIN,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    w_low: int = DEFAULT_W_LOW,
    w_high: int = DEFAULT_W_HIGH,
) -> tuple[int, int]:
    """
    Same as ``choose_windows_and_bins``, but if nothing qualifies use coarsest
    ``W=w_low``, ``k=k_min`` so callers always get a usable grid (sparse regimes).
    """
    chosen = choose_windows_and_bins(n, e_min, k_min, k_max, w_low, w_high)
    if chosen is not None:
        return chosen
    return (w_low, k_min)


def fit_marginal_edges(values: np.ndarray, k: int) -> np.ndarray:
    """
    Internal quantile cut points between ``k`` equal-population marginal bins.

    Parameters
    ----------
    values :
        1D finite samples used **only** for calibration (IS).
    k :
        Number of marginal bins (same count used for spread and accel sides).

    Returns
    -------
    edges :
        Shape ``(k - 1,)``, strictly sorted ascending where unique.
        Bin ``j`` spans ``(-inf, edges[0]]``, ..., ``(edges[j-1], edges[j]]``, ...
        with tie-breaking handled in ``assign_marginal_bins``.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    k = int(k)
    if k <= 1:
        return np.array([], dtype=np.float64)
    if values.size == 0:
        return np.full(k - 1, np.nan, dtype=np.float64)

    qs = np.linspace(0.0, 1.0, k + 1)[1:-1]
    edges = np.quantile(values, qs)
    edges.sort()
    return edges


def assign_marginal_bins(values: np.ndarray, edges: np.ndarray, k: int) -> np.ndarray:
    """
    Map scalar values to marginal bin indices ``0 .. k-1`` using frozen ``edges``.
    Non-finite inputs → ``-1``.
    """
    values = np.asarray(values, dtype=np.float64)
    k = int(k)
    out = np.full(values.shape, -1, dtype=np.intp)

    if k <= 1:
        m = np.isfinite(values)
        out[m] = 0
        return out

    edges = np.asarray(edges, dtype=np.float64).ravel()
    if edges.size != k - 1:
        raise ValueError(f"edges must have length k-1 ({k - 1}), got {edges.size}")

    m = np.isfinite(values)
    if not np.any(m):
        return out

    vv = values[m]
    idx = np.searchsorted(edges, vv, side="right")
    idx = np.clip(idx, 0, k - 1)
    out[m] = idx
    return out


def joint_bin_ids(spread_bin: np.ndarray, accel_bin: np.ndarray, k: int) -> np.ndarray:
    """``bin_id = spread_bin * k + accel_bin`` (invalid rows keep ``-1`` if inputs ``-1``)."""
    spread_bin = np.asarray(spread_bin, dtype=np.intp)
    accel_bin = np.asarray(accel_bin, dtype=np.intp)
    out = np.full(spread_bin.shape, -1, dtype=np.intp)
    ok = (spread_bin >= 0) & (accel_bin >= 0)
    out[ok] = spread_bin[ok] * int(k) + accel_bin[ok]
    return out


def calculate_quantiles(
    metrics: dict,
    setup_indices: np.ndarray,
    e_min: float = DEFAULT_E_MIN,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    w_low: int = DEFAULT_W_LOW,
    w_high: int = DEFAULT_W_HIGH,
    *,
    fallback_coarse: bool = True,
) -> dict:
    """
    Fit marginal spread/accel edges on finite setups and assign joint ``bin_id``.

    Parameters
    ----------
    metrics :
        Dict with numpy arrays ``spread_delta`` and ``norm_accel``, same length as setups.
    setup_indices :
        Row indices into composites (validated against metric lengths).

    Returns
    -------
    dict with keys:
        ``W``, ``k``, ``edges_spread_delta``, ``edges_norm_accel`` (rounded to 5 decimals
        before bin assignment), ``spread_bin``, ``norm_accel_bin``, ``bin_id``.
    """
    setup_indices = np.asarray(setup_indices).ravel()

    spread = np.asarray(metrics["spread_delta"], dtype=np.float64).ravel()
    accel = np.asarray(metrics["norm_accel"], dtype=np.float64).ravel()

    if spread.shape != accel.shape:
        raise ValueError("spread_delta and norm_accel must have the same shape")

    if setup_indices.size != spread.size:
        raise ValueError(
            f"setup_indices length {setup_indices.size} != metrics length {spread.size}"
        )

    finite = np.isfinite(spread) & np.isfinite(accel)
    n_fit = int(np.sum(finite))
    n_total = int(spread.size)

    if fallback_coarse:
        W, k = choose_windows_and_bins_or_coarse(
            n_fit, e_min, k_min, k_max, w_low, w_high
        )
        fallback_used = (
            choose_windows_and_bins(n_fit, e_min, k_min, k_max, w_low, w_high)
            is None
        )
    else:
        pair = choose_windows_and_bins(n_fit, e_min, k_min, k_max, w_low, w_high)
        if pair is None:
            raise ValueError(
                f"Insufficient finite setups ({n_fit}) for grid with "
                f"e_min={e_min}, k_min={k_min}, W∈[{w_low},{w_high})"
            )
        W, k = pair
        fallback_used = False

    spread_fit = spread[finite]
    accel_fit = accel[finite]

    edges_spread = fit_marginal_edges(spread_fit, k)
    edges_accel = fit_marginal_edges(accel_fit, k)
    edges_spread = np.round(np.asarray(edges_spread, dtype=np.float64), 5)
    edges_accel = np.round(np.asarray(edges_accel, dtype=np.float64), 5)

    spread_bin = assign_marginal_bins(spread, edges_spread, k)
    accel_bin = assign_marginal_bins(accel, edges_accel, k)
    bin_id = joint_bin_ids(spread_bin, accel_bin, k)

    return setup_indices, {
        "W": W,
        "k": k,
        "edges_spread_delta": edges_spread,
        "edges_norm_accel": edges_accel,
        "spread_bin": spread_bin,
        "norm_accel_bin": accel_bin,
        "bin_id": bin_id,
    }