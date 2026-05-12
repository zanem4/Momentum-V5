# simulate trades

import json
import os
from typing import Any, Optional

import numpy as np


def _write_trade_plot_sample(
    path: str,
    extra: dict[str, Any],
    composite_candles: np.ndarray,
    indices: np.ndarray,
    l_value: int,
    direction: str,
    horizon: int,
    mt_i: int,
    ms_i: int,
    trading_indices: np.ndarray,
    trade_outcome: np.ndarray,
    stop_exit_bar: np.ndarray,
    target_exit_bar: np.ndarray,
    timeout_exit_bar: np.ndarray,
    close_stop: np.ndarray,
    close_target: np.ndarray,
    close_timeout: np.ndarray,
    entry_prices: np.ndarray,
    raw_delta: np.ndarray,
    target_mults: np.ndarray,
    stop_mults: np.ndarray,
) -> None:
    """First setup only; tiny JSON for candlestick plotter. O(1) work."""
    if indices.size == 0:
        return
    s0, H = 0, int(horizon)
    setup_end = int(indices[s0])
    setup_first = setup_end - (int(l_value) - 1)
    entry_bar = int(trading_indices[s0])
    ep = float(entry_prices[s0])
    d0 = float(raw_delta[s0])
    tm = float(np.asarray(target_mults, dtype=np.float64).ravel()[mt_i])
    sm = float(np.asarray(stop_mults, dtype=np.float64).ravel()[ms_i])
    if direction == "long":
        tgt_px = ep + d0 * tm
        stp_px = ep - d0 * sm
    else:
        tgt_px = ep - d0 * tm
        stp_px = ep + d0 * sm
    out = int(trade_outcome[s0, mt_i, ms_i])
    if out == -1:
        exit_bar = int(stop_exit_bar[s0, mt_i, ms_i])
        mark_price = float(close_stop[s0, mt_i, ms_i])
        mark_kind = "stop"
    elif out == 1:
        exit_bar = int(target_exit_bar[s0, mt_i, ms_i])
        mark_price = float(close_target[s0, mt_i, ms_i])
        mark_kind = "target"
    else:
        exit_bar = int(timeout_exit_bar[s0])
        mark_price = float(close_timeout[s0, 0, 0])
        mark_kind = "timeout"

    n = composite_candles.shape[0]
    i_lo = max(0, setup_first - 3)
    i_hi = min(n - 1, entry_bar + H - 1 + 3)
    sl = slice(i_lo, i_hi + 1)
    chunk = composite_candles[sl]
    payload = {
        "meta": {
            **extra,
            "direction": direction,
            "horizon": H,
            "lookback": int(l_value),
            "mt_idx": int(mt_i),
            "ms_idx": int(ms_i),
            "target_mult": float(extra.get("target_mult", 0.0)),
            "stop_mult": float(extra.get("stop_mult", 0.0)),
        },
        "unix_time": chunk[:, 0].astype(np.float64).tolist(),
        "open": chunk[:, 1].astype(np.float64).tolist(),
        "high": chunk[:, 2].astype(np.float64).tolist(),
        "low": chunk[:, 3].astype(np.float64).tolist(),
        "close": chunk[:, 4].astype(np.float64).tolist(),
        "setup_first_bar": setup_first,
        "setup_last_bar": setup_end,
        "entry_bar": entry_bar,
        "exit_bar": exit_bar,
        "entry_price": ep,
        "target_price": tgt_px,
        "stop_price": stp_px,
        "outcome": out,
        "marker": {"bar": exit_bar, "price": mark_price, "kind": mark_kind},
        "slice_start_bar": int(i_lo),
        "delta_left_price": float(
            composite_candles[setup_first, 3] if direction == "long" else composite_candles[setup_first, 2]
        ),
        "delta_right_price": float(
            composite_candles[setup_end, 2] if direction == "long" else composite_candles[setup_end, 3]
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def simulate_trades(
    composite_candles,
    indices,
    pip_value,
    direction,
    horizon,
    target_multipliers,
    stop_multipliers,
    metrics,
    *,
    plot_sample_path: Optional[str] = None,
    plot_sample_l_value: Optional[int] = None,
    plot_sample_extra: Optional[dict[str, Any]] = None,
    plot_sample_mt: int = 0,
    plot_sample_ms: int = 0,
):

    # TOHLCV + p10..p90 spreads at cols 6–10; simulation uses random percentile per fill
    pct = composite_candles[:, 6:11]
    opens = composite_candles[:, 1]
    closes = composite_candles[:, 4]
    rng = np.random.default_rng()

    # direction-aware callables
    target_check = np.greater_equal if direction == "long" else np.less_equal
    stop_check = np.less_equal if direction == "long" else np.greater_equal
    target_dir = np.add if direction == "long" else np.subtract
    stop_dir = np.subtract if direction == "long" else np.add
    target_idx = 2 if direction == "long" else 3
    stop_idx = 3 if direction == "long" else 2
    side = 1.0 if direction == "long" else -1.0

    H = int(horizon)

    # entry bar: half-spread from open using one random percentile row per setup
    trading_indices = indices + 1  # trade begins on bar after setup
    S = int(trading_indices.shape[0])
    pick_e = rng.integers(0, 5, size=S, endpoint=False)
    spreads_entry = pct[trading_indices, pick_e]
    entry_prices = opens[trading_indices] + (spreads_entry / 2)
    price_delta = metrics["raw_delta"]

    # prep array of prices to check for targets and stops per horizon array
    horizon_array = trading_indices[:, np.newaxis] + np.arange(H)

    # full grid (M_t × M_s): target on axis 2, stop on axis 3 — broadcast to (S, H, M_t, M_s)
    entry_s = entry_prices[:, np.newaxis, np.newaxis, np.newaxis]
    d_s = price_delta[:, np.newaxis, np.newaxis, np.newaxis]
    mt = np.asarray(target_multipliers, dtype=np.float64).reshape(1, 1, -1, 1)
    ms = np.asarray(stop_multipliers, dtype=np.float64).reshape(1, 1, 1, -1)

    target_price_array = target_dir(entry_s, d_s * mt)
    stop_price_array = stop_dir(entry_s, d_s * ms)

    # check for stop and target hits
    path_target = composite_candles[horizon_array, target_idx][:, :, np.newaxis, np.newaxis]
    path_stop = composite_candles[horizon_array, stop_idx][:, :, np.newaxis, np.newaxis]
    target_hit = target_check(path_target, target_price_array)
    stop_hit = stop_check(path_stop, stop_price_array)

    # convert hit booleans to time indices per (target mult, stop mult) along horizon axis
    h = np.arange(H, dtype=np.intp).reshape(1, H, 1, 1)
    target_hit_times = np.min(
        np.where(
            target_hit,
            h,
            H,
        ),
        axis=1,
    )
    stop_hit_times = np.min(
        np.where(
            stop_hit,
            h,
            H,
        ),
        axis=1,
    )

    # -1 = stop first (tie → stop), 0 = timeout, 1 = target
    trade_outcome = np.where(
        (target_hit_times == H) & (stop_hit_times == H),
        0,
        np.where(stop_hit_times <= target_hit_times, -1, 1),
    )

    # exit bar index in composite rows: horizon_array[s, t] = entry_bar_s + t
    s = np.arange(horizon_array.shape[0], dtype=np.intp)[:, np.newaxis, np.newaxis]
    t_stop = np.clip(stop_hit_times, 0, H - 1)
    t_tgt = np.clip(target_hit_times, 0, H - 1)
    stop_exit_bar = horizon_array[s, t_stop]
    target_exit_bar = horizon_array[s, t_tgt]
    timeout_exit_bar = horizon_array[:, -1]

    # get price value at corresponding close
    close_stop = closes[stop_exit_bar]
    close_target = closes[target_exit_bar]
    close_timeout = closes[timeout_exit_bar][:, np.newaxis, np.newaxis]

    # exit: full spread cost — independent random percentile per (setup, Mt, Ms) at exit bar
    r_stop = rng.integers(0, 5, size=stop_exit_bar.shape, endpoint=False)
    r_tgt = rng.integers(0, 5, size=target_exit_bar.shape, endpoint=False)
    r_to = rng.integers(0, 5, size=timeout_exit_bar.shape, endpoint=False)
    sp_stop = pct[stop_exit_bar, r_stop]
    sp_tgt = pct[target_exit_bar, r_tgt]
    sp_to = pct[timeout_exit_bar, r_to][:, np.newaxis, np.newaxis]
    spread_end = np.where(
        trade_outcome == -1,
        sp_stop,
        np.where(trade_outcome == 1, sp_tgt, sp_to),
    )

    # concatenate into a single array
    close_prices = np.where(
        trade_outcome == -1,
        close_stop,
        np.where(trade_outcome == 1, close_target, close_timeout),
    )

    # calculate PnL based on close price and outcome label. pay full spread at close (market order)
    entry_grid = entry_prices[:, np.newaxis, np.newaxis]
    pnl = np.where(
        trade_outcome == -1,
        (-1 * np.abs(entry_grid - close_stop) - spread_end) * pip_value,
        np.where(
            trade_outcome == 1,
            (np.abs(close_target - entry_grid) - spread_end) * pip_value,
            (side * (close_timeout - entry_grid) - spread_end) * pip_value,
        ),
    )

    if plot_sample_path and plot_sample_l_value is not None and plot_sample_extra is not None:
        tgt_arr = np.asarray(target_multipliers, dtype=np.float64).ravel()
        stp_arr = np.asarray(stop_multipliers, dtype=np.float64).ravel()
        mt_i = int(np.clip(plot_sample_mt, 0, max(0, tgt_arr.size - 1)))
        ms_i = int(np.clip(plot_sample_ms, 0, max(0, stp_arr.size - 1)))
        ex = dict(plot_sample_extra)
        ex["target_mult"] = float(tgt_arr[mt_i])
        ex["stop_mult"] = float(stp_arr[ms_i])
        _write_trade_plot_sample(
            plot_sample_path,
            ex,
            composite_candles,
            indices,
            int(plot_sample_l_value),
            direction,
            H,
            mt_i,
            ms_i,
            trading_indices,
            trade_outcome,
            stop_exit_bar,
            target_exit_bar,
            timeout_exit_bar,
            close_stop,
            close_target,
            close_timeout,
            entry_prices,
            metrics["raw_delta"],
            target_multipliers,
            stop_multipliers,
        )

    return pnl
