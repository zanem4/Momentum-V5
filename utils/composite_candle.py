# composite candle by timeframe — NumPy, UTC bucket alignment
import numpy as np

# Input / output column layout (TOHLCV + five spread percentiles)
# 0: Time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume
# 6: p10_spread, 7: p30_spread, 8: p50_spread, 9: p70_spread, 10: p90_spread
_N_COL = 11

_MIN_COVERAGE_NUM = 4
_MIN_COVERAGE_DEN = 5
_MAX_CONSECUTIVE_MISSING = 3
# Reject if any window of this many expected minutes is all missing
_GAP_REJECT_WINDOW = _MAX_CONSECUTIVE_MISSING + 1


def composite_candle(data, timeframe_minutes):
    """
    Build composite (HTF) bars from m1 data. Buckets are UTC-aligned:
    period_id = floor(time / (timeframe_minutes * 60)).

    A bucket is kept if expected minute stamps bucket_start + i*60 for
    i in 0..timeframe_minutes-1 satisfy: at least 80% have a row at that exact
    Unix time, and no more than 3 consecutive expected minutes are missing.

    Rows are ordered by (period_id, time); aggregation uses reduceat over runs.
    Column 0 is set to bucket_start.

    Input ``data`` must have at least 11 columns (TOHLCV + p10/p30/p50/p70/p90 spread).
    For each spread column, the composite value is the **mean** of m1 values in the bucket.

    Returns composites with shape (n, 11) and the m1 row indices used (for alignment with m1).
    """
    if data.size == 0:
        return np.empty((0, _N_COL), dtype=data.dtype), np.array([], dtype=np.intp)

    if data.shape[1] < _N_COL:
        raise ValueError(
            f"expected at least {_N_COL} columns (TOHLCV + 5 spread percentiles), got {data.shape[1]}"
        )

    period_seconds = timeframe_minutes * 60
    m1_times = data[:, 0]
    period_id = (m1_times // period_seconds).astype(np.int64)

    # Primary: period_id, secondary: time — contiguous runs are chronological
    order = np.lexsort((m1_times, period_id))
    ds = data[order]
    ps = period_id[order]
    times = ds[:, 0].astype(np.int64, copy=False)

    boundaries = np.concatenate([[0], np.where(np.diff(ps))[0] + 1, [len(ps)]])
    run_starts = boundaries[:-1]
    run_ends = boundaries[1:]
    n_runs = len(run_starts)
    if n_runs == 0:
        return np.empty((0, _N_COL), dtype=data.dtype), np.array([], dtype=np.intp)

    run_lengths = (run_ends - run_starts).astype(np.float64)
    run_id = np.repeat(np.arange(n_runs, dtype=np.intp), (run_ends - run_starts).astype(np.intp))

    ps_i64 = ps.astype(np.int64, copy=False)
    bucket_row = ps_i64 * np.int64(period_seconds)
    slot = (times - bucket_row) // 60
    on_grid = (times == bucket_row + slot * 60) & (slot >= 0) & (slot < timeframe_minutes)

    nb = n_runs * timeframe_minutes
    flat = run_id.astype(np.int64) * timeframe_minutes + slot.astype(np.int64)
    counts = np.bincount(flat[on_grid], minlength=nb)
    present = (counts.reshape(n_runs, timeframe_minutes) > 0)

    present_count = present.sum(axis=1)
    cov_ok = present_count * _MIN_COVERAGE_DEN >= _MIN_COVERAGE_NUM * timeframe_minutes

    missing = ~present
    if timeframe_minutes >= _GAP_REJECT_WINDOW:
        z = np.zeros((n_runs, 1), dtype=np.int32)
        cs = np.cumsum(np.hstack([z, missing.astype(np.int32)]), axis=1)
        w = _GAP_REJECT_WINDOW
        roll = cs[:, w:] - cs[:, :-w]
        gap_ok = (roll < w).all(axis=1)
    else:
        gap_ok = np.ones(n_runs, dtype=bool)

    eligible = cov_ok & gap_ok
    if not np.any(eligible):
        return np.empty((0, _N_COL), dtype=data.dtype), np.array([], dtype=np.intp)

    e = eligible
    buck = ps_i64[run_starts] * np.int64(period_seconds)

    vol = np.add.reduceat(ds[:, 5], run_starts)
    hi = np.maximum.reduceat(ds[:, 2], run_starts)
    lo = np.minimum.reduceat(ds[:, 3], run_starts)

    open_ = ds[run_starts, 1]
    re_1 = run_ends - 1
    close_ = ds[re_1, 4]

    sp_means: list[np.ndarray] = []
    for j in range(6, _N_COL):
        sm = np.add.reduceat(ds[:, j].astype(np.float64, copy=False), run_starts)
        sp_means.append(sm / np.maximum(run_lengths, 1.0))

    out_dt = np.result_type(data.dtype, np.float64)
    out = np.column_stack(
        [
            buck[e].astype(out_dt, copy=False),
            open_[e].astype(out_dt, copy=False),
            hi[e].astype(out_dt, copy=False),
            lo[e].astype(out_dt, copy=False),
            close_[e].astype(out_dt, copy=False),
            vol[e].astype(out_dt, copy=False),
            *[m[e].astype(out_dt, copy=False) for m in sp_means],
        ]
    )

    keep_row = e[run_id]
    m1_indices = order[keep_row]

    return out, m1_indices
