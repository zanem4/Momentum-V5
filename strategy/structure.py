# structure

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def structure(composite_candles, l_value, direction):
    """
    Strict ordered-chain mask on mid OHLC over the last ``l_value`` bars.

    For each completion bar index ``i`` (``i >= l_value - 1``), requires
    pairwise ``order(O[k], O[k+1])`` for all consecutive pairs within the
    window ending at ``i``, and likewise for H, L, C.

    ``direction == "long"`` uses ``np.greater``; ``"short"`` uses ``np.less``.

    Returns a length-``len(composite_candles)`` bool vector aligned to composite
    rows (False for indices ``< l_value - 1``).
    """
    order = np.greater if direction == "long" else np.less

    opens = composite_candles[:, 1]
    highs = composite_candles[:, 2]
    lows = composite_candles[:, 3]
    closes = composite_candles[:, 4]

    n = opens.shape[0]
    if n == 0:
        return np.array([], dtype=bool)

    if l_value < 1:
        raise ValueError("l_value must be >= 1")

    wo = sliding_window_view(opens, l_value)
    wh = sliding_window_view(highs, l_value)
    wl = sliding_window_view(lows, l_value)
    wc = sliding_window_view(closes, l_value)

    chain_ok = (
        order(wo[:, 1:], wo[:, :-1]).all(axis=1) # ordered opens
        & order(wh[:, 1:], wh[:, :-1]).all(axis=1) # ordered highs
        & order(wl[:, 1:], wl[:, :-1]).all(axis=1) # ordered lows
        & order(wc[:, 1:], wc[:, :-1]).all(axis=1) # ordered closes
        & order(wc, wo).all(axis=1) # ordered closes vs opens (forces green candles for longs, red for shorts)
    )

    bool_indices = np.zeros(n, dtype=bool)
    bool_indices[l_value - 1 :] = chain_ok

    # convert bool array to numeric indices of composite_candles
    numeric_indices = np.flatnonzero(bool_indices)
    return numeric_indices