# composite candle by timeframe — NumPy, UTC bucket alignment
import numpy as np

# Column indices (same as run.py comment)
# 0: Time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume
# 6: Bid Open, 7: Bid High, 8: Bid Low, 9: Bid Close
# 10: Ask Open, 11: Ask High, 12: Ask Low, 13: Ask Close, 14: Spread Close

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

    Returns composites and the m1 row indices used (for alignment with m1).
    """
    if data.size == 0:
        return np.empty((0, data.shape[1]), dtype=data.dtype), np.array([], dtype=np.intp)

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
        return np.empty((0, data.shape[1]), dtype=data.dtype), np.array([], dtype=np.intp)

    run_lengths = run_ends - run_starts
    run_id = np.repeat(np.arange(n_runs, dtype=np.intp), run_lengths)

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
        return np.empty((0, data.shape[1]), dtype=data.dtype), np.array([], dtype=np.intp)

    e = eligible
    buck = ps_i64[run_starts] * np.int64(period_seconds)

    vol = np.add.reduceat(ds[:, 5], run_starts)
    hi = np.maximum.reduceat(ds[:, 2], run_starts)
    lo = np.minimum.reduceat(ds[:, 3], run_starts)
    bid_hi = np.maximum.reduceat(ds[:, 7], run_starts)
    bid_lo = np.minimum.reduceat(ds[:, 8], run_starts)
    ask_hi = np.maximum.reduceat(ds[:, 11], run_starts)
    ask_lo = np.minimum.reduceat(ds[:, 12], run_starts)

    open_ = ds[run_starts, 1]
    bid_open = ds[run_starts, 6]
    ask_open = ds[run_starts, 10]
    re_1 = run_ends - 1
    close_ = ds[re_1, 4]
    bid_close = ds[re_1, 9]
    ask_close = ds[re_1, 13]
    spr_close = ds[re_1, 14]

    out = np.column_stack([
        buck[e].astype(data.dtype, copy=False),
        open_[e], hi[e], lo[e], close_[e], vol[e],
        bid_open[e], bid_hi[e], bid_lo[e], bid_close[e],
        ask_open[e], ask_hi[e], ask_lo[e], ask_close[e], spr_close[e],
    ]).astype(data.dtype, copy=False)

    keep_row = e[run_id]
    m1_indices = order[keep_row]

    return out, m1_indices
