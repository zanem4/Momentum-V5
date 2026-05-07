# sort trades by calendar windows × quantile bins × target/stop grid

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from numba import njit


@njit(cache=True)
def _medians_from_sorted_bins(bid_sorted: np.ndarray, vals_sorted: np.ndarray, nbins: int) -> np.ndarray:
    """One median per bin id ``0 .. nbins-1``; empty bins stay NaN. ``bid_sorted`` non-decreasing."""
    out = np.empty(nbins, dtype=np.float64)
    for b in range(nbins):
        out[b] = np.nan
    n = bid_sorted.shape[0]
    i = 0
    while i < n:
        b = bid_sorted[i]
        j = i + 1
        while j < n and bid_sorted[j] == b:
            j += 1
        mlen = j - i
        if mlen == 1:
            out[b] = vals_sorted[i]
        else:
            tmp = vals_sorted[i:j].copy()
            tmp.sort()
            half = mlen // 2
            if mlen % 2 == 1:
                out[b] = tmp[half]
            else:
                out[b] = 0.5 * (tmp[half - 1] + tmp[half])
        i = j
    return out


def _parse_utc_epoch_seconds(date_str: str) -> float:
    dt = datetime.fromisoformat(date_str.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _aggregate_window_mt_ms(
    v: np.ndarray,
    joint_bin: np.ndarray,
    in_win: np.ndarray,
    valid_bin: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (count, mean, median, std) each shape (k, k) for one window and one (mt, ms) column.
    """
    m = in_win & valid_bin & np.isfinite(v)
    if not np.any(m):
        kk = int(k)
        z = np.zeros((kk, kk), dtype=np.intp)
        nan2 = np.full((kk, kk), np.nan, dtype=np.float64)
        return z, nan2.copy(), nan2.copy(), nan2.copy()

    bid = (joint_bin[m]).astype(np.int64, copy=False)
    vals = v[m]
    nbins = int(k) * int(k)

    cnt = np.bincount(bid, minlength=nbins)
    sm = np.bincount(bid, weights=vals, minlength=nbins)
    sm2 = np.bincount(bid, weights=vals * vals, minlength=nbins)

    mean_flat = np.full(nbins, np.nan, dtype=np.float64)
    std_flat = np.full(nbins, np.nan, dtype=np.float64)
    nz = cnt > 0
    mean_flat[nz] = sm[nz] / cnt[nz].astype(np.float64)
    ge2 = cnt >= 2
    # sample variance (ddof=1): (sumsq - sum^2/n) / (n-1)
    var_num = sm2[ge2] - (sm[ge2] * sm[ge2]) / cnt[ge2].astype(np.float64)
    std_flat[ge2] = np.sqrt(np.maximum(var_num / (cnt[ge2].astype(np.float64) - 1.0), 0.0))

    order = np.argsort(bid, kind="mergesort")
    med_flat = _medians_from_sorted_bins(bid[order], vals[order], nbins)

    return (
        cnt.reshape(k, k).astype(np.intp, copy=False),
        mean_flat.reshape(k, k),
        med_flat.reshape(k, k),
        std_flat.reshape(k, k),
    )


def sort_trades(pnl, indices, edge_data, parameters, composite_candles):
    """
    Aggregate PnL into (window, spread_bin, norm_accel_bin, target_mult_idx, stop_mult_idx).

    Windows: ``edge_data["W"]`` equal-length intervals over
    [parameters start_date, parameters end_date), overlap fraction 0.5 between consecutive starts.
    Setup time = composite_candles[indices, 0] (UNIX UTC).

    Requires rows aligned: ``pnl.shape[0] == len(indices) == len(edge_data["spread_bin"])``.
    """
    pnl = np.asarray(pnl, dtype=np.float64)
    indices = np.asarray(indices).ravel()
    if pnl.ndim == 1:
        pnl = pnl[:, np.newaxis, np.newaxis]

    Wn = int(edge_data["W"])
    k = int(edge_data["k"])
    spread_bin = np.asarray(edge_data["spread_bin"], dtype=np.intp).ravel()
    accel_bin = np.asarray(edge_data["norm_accel_bin"], dtype=np.intp).ravel()

    if not (pnl.shape[0] == indices.size == spread_bin.size == accel_bin.size):
        raise ValueError(
            "pnl, indices, and edge_data bins must have the same length along the setup axis"
        )

    Mt, Ms = pnl.shape[1], pnl.shape[2]
    start_ts = _parse_utc_epoch_seconds(parameters["start_date"])
    end_ts = _parse_utc_epoch_seconds(parameters["end_date"])
    T = end_ts - start_ts
    if T <= 0:
        raise ValueError("end_date must be after start_date")

    if Wn < 1:
        raise ValueError("edge_data['W'] must be >= 1")
    L = 2.0 * T / float(Wn + 1)
    step = L / 2.0

    window_t0_utc = start_ts + np.arange(Wn, dtype=np.float64) * step
    window_t1_utc = window_t0_utc + L

    setup_times = np.asarray(composite_candles[indices, 0], dtype=np.float64).ravel()
    valid_bin = (spread_bin >= 0) & (accel_bin >= 0)
    joint_bin = np.full(spread_bin.shape[0], -1, dtype=np.int64)
    joint_bin[valid_bin] = spread_bin[valid_bin].astype(np.int64) * int(k) + accel_bin[valid_bin].astype(
        np.int64
    )

    mean = np.full((Wn, k, k, Mt, Ms), np.nan, dtype=np.float64)
    median = np.full((Wn, k, k, Mt, Ms), np.nan, dtype=np.float64)
    std = np.full((Wn, k, k, Mt, Ms), np.nan, dtype=np.float64)
    count = np.zeros((Wn, k, k, Mt, Ms), dtype=np.intp)

    for w in range(Wn):
        t0 = window_t0_utc[w]
        t1 = window_t1_utc[w]
        in_win = (setup_times >= t0) & (setup_times < t1)
        for mt in range(Mt):
            for ms in range(Ms):
                c, mn, md, st = _aggregate_window_mt_ms(pnl[:, mt, ms], joint_bin, in_win, valid_bin, k)
                count[w, :, :, mt, ms] = c
                mean[w, :, :, mt, ms] = np.round(mn, 3)
                median[w, :, :, mt, ms] = np.round(md, 3)
                std[w, :, :, mt, ms] = np.round(st, 3)

    return {
        "W": Wn,
        "k": k,
        "window_length_seconds": L,
        "window_step_seconds": step,
        "window_t0_utc": window_t0_utc,
        "window_t1_utc": window_t1_utc,
        "mean": mean,
        "median": median,
        "std": std,
        "count": count,
    }
