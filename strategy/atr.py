# calculate ATR

# imports
import numpy as np


def calculate_atr(htf_data, setup_indices, atr_length, pip_value):

    # get HLC data
    highs = htf_data[:, 2]
    lows = htf_data[:, 3]
    closes = htf_data[:, 4]

    # validate indices (need enough bars before setup for enough data)
    valid_indices = setup_indices[setup_indices >= atr_length]
    bars = valid_indices[:, np.newaxis] + np.arange(-(atr_length - 1), 1)

    # calculate true range TR
    high_low = highs[bars] - lows[bars]
    prev_close = closes[bars - 1]
    high_prev = np.abs(highs[bars] - prev_close)
    low_prev = np.abs(lows[bars] - prev_close)
    tr = np.maximum(high_low, np.maximum(high_prev, low_prev))

    # SMA of TR values
    atr = np.round(np.mean(tr, axis=1) * pip_value, 2)

    return valid_indices, atr