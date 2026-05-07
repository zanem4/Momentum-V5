# simulate trades

# imports
import numpy as np


def simulate_trades(composite_candles, indices, pip_value, direction, horizon, target_multipliers, stop_multipliers, metrics):

    # get needed ohlc
    spreads = composite_candles[:, 14]
    opens = composite_candles[:, 1]
    closes = composite_candles[:, 4]

    # direction-aware callables
    target_check = np.greater_equal if direction == "long" else np.less_equal
    stop_check = np.less_equal if direction == "long" else np.greater_equal
    target_dir = np.add if direction == "long" else np.subtract
    stop_dir = np.subtract if direction == "long" else np.add
    target_idx = 2 if direction == "long" else 3
    stop_idx = 3 if direction == "long" else 2
    side = 1.0 if direction == "long" else -1.0

    H = int(horizon)

    # get trading indices and calculate entry price (pay half of spread as move from mid to bid or ask)
    trading_indices = indices + 1  # trade begins on bar after setup
    entry_prices = opens[trading_indices] + (spreads[trading_indices] / 2)
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

    # get spread at corresponding trade end
    spread_end = np.where(
        trade_outcome == -1,
        spreads[stop_exit_bar],
        np.where(
            trade_outcome == 1,
            spreads[target_exit_bar],
            spreads[timeout_exit_bar][:, np.newaxis, np.newaxis],
        ),
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
    return pnl