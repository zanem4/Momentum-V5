# calculate joint metrics for each setup

# imports
import numpy as np

def calculate_metrics(composite_candles, setup_indices, atr, direction, l_value, pip_value):

    # direction-aware callable
    starting_ohlc = 3 if direction == "long" else 2
    ending_ohlc = 2 if direction == "long" else 3

    # find price delta of setup. longs: highs[i] - lows[i-N]. shorts: lows[i] - highs[i-N]
    # absolute does both since magnitude is the only concern
    price_delta = np.abs(
        composite_candles[setup_indices, ending_ohlc] - composite_candles[setup_indices - l_value + 1, starting_ohlc]
    )

    # find acceleration (mid close based)
    accel_bars = setup_indices[:, np.newaxis] + np.arange(-l_value + 1, 1)
    close_values = composite_candles[accel_bars, 4]
    velo_values = np.diff(close_values, axis=1)
    accel_values = np.diff(velo_values, axis=1)
    acceleration_avg = np.mean(accel_values, axis=1)

    # get spread at entry
    spread_at_entry = composite_candles[setup_indices, 14]

    # calculate and return metrics
    metrics = {
        "norm_accel": np.round(acceleration_avg / (atr / pip_value), 5),
        "spread_delta": np.round(spread_at_entry / price_delta, 5),
        "raw_delta": price_delta
    }
    return metrics