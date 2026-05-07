# persist edge calibration (JSON) and sorted PnL summaries (Parquet)

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32, np.intp)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (bool, str)) or obj is None:
        return obj
    return obj


def _parse_utc_epoch_seconds(date_str: str) -> float:
    dt = datetime.fromisoformat(str(date_str).strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _slim_edge_for_json(edge_data: dict, parameters: dict) -> dict[str, Any]:
    """Calibration + window bounds only (no per-setup bin arrays). Matches ``sort_trades`` geometry."""
    Wn = int(edge_data["W"])
    start_ts = _parse_utc_epoch_seconds(parameters["start_date"])
    end_ts = _parse_utc_epoch_seconds(parameters["end_date"])
    T = end_ts - start_ts
    if T <= 0:
        raise ValueError("end_date must be after start_date")
    L = 2.0 * T / float(Wn + 1)
    step = L / 2.0
    bounds: list[dict[str, Any]] = []
    for w in range(Wn):
        t0 = start_ts + w * step
        t1 = t0 + L
        bounds.append(
            {
                "window_index": w,
                "window_t0_utc": float(t0),
                "window_t1_utc": float(t1),
            }
        )
    return {
        "W": Wn,
        "k": int(edge_data["k"]),
        "edges_spread_delta": _json_sanitize(edge_data["edges_spread_delta"]),
        "edges_norm_accel": _json_sanitize(edge_data["edges_norm_accel"]),
        "window_bounds": bounds,
        "window_length_seconds": float(L),
        "window_step_seconds": float(step),
        "window_bounds_note": "window_t0_utc / window_t1_utc are UNIX epoch seconds (UTC). Join Parquet on window_index.",
    }


def save_edge_data_json(
    atr_dir: str,
    long_edge_data: dict,
    short_edge_data: dict,
    parameters: dict,
) -> None:
    """Write slim edge JSON (edges, W, k, window_bounds). Per-setup bin arrays omitted."""
    os.makedirs(atr_dir, exist_ok=True)
    path_long = os.path.join(atr_dir, "long_edge_data.json")
    path_short = os.path.join(atr_dir, "short_edge_data.json")
    with open(path_long, "w", encoding="utf-8") as f:
        json.dump(_slim_edge_for_json(long_edge_data, parameters), f, indent=2)
    with open(path_short, "w", encoding="utf-8") as f:
        json.dump(_slim_edge_for_json(short_edge_data, parameters), f, indent=2)


def _sorted_to_long_frame(
    sorted_dict: dict,
    direction: str,
    target_multipliers: np.ndarray,
    stop_multipliers: np.ndarray,
    n_candles_forward: int,
) -> pd.DataFrame:
    mean = np.asarray(sorted_dict["mean"], dtype=np.float64)
    median = np.asarray(sorted_dict["median"], dtype=np.float64)
    std = np.asarray(sorted_dict["std"], dtype=np.float64)
    count = np.asarray(sorted_dict["count"], dtype=np.intp)

    Wn, k, k2, Mt, Ms = mean.shape
    if k != k2:
        raise ValueError("mean must be square along spread/accel bin axes")

    tgt = np.asarray(target_multipliers, dtype=np.float64).ravel()
    stp = np.asarray(stop_multipliers, dtype=np.float64).ravel()
    if tgt.size != Mt or stp.size != Ms:
        raise ValueError("target/stop multiplier lists must match sorted tensor shape")

    g = np.indices((Wn, k, k, Mt, Ms))
    w, i, j, mt, ms = (g[t].ravel() for t in range(5))

    return pd.DataFrame(
        {
            "direction": direction,
            "n_candles_forward": np.full(w.size, int(n_candles_forward), dtype=np.int32),
            "window_index": w.astype(np.int32),
            "spread_bin": i.astype(np.int32),
            "norm_accel_bin": j.astype(np.int32),
            "target_mult": tgt[mt],
            "stop_mult": stp[ms],
            "mean": mean.ravel(),
            "median": median.ravel(),
            "std": std.ravel(),
            "count": count.ravel().astype(np.int64),
        }
    )


def save_data(
    atr_dir: str,
    per_horizon: list[tuple[int, dict, dict]],
    parameters: dict,
) -> None:
    """
    Concatenate long-format summaries for every forward horizon and write one Parquet
    under ``atr_dir`` (no per-horizon subfolder). Each row includes ``n_candles_forward``.

    Filename encodes the configured forward range from parameters:
    ``pnl_summaries_forward_{lo}_to_{hi}.parquet``, or ``pnl_summaries_forward_{h}.parquet``
    when min and max horizon are the same.
    """
    if not per_horizon:
        return

    os.makedirs(atr_dir, exist_ok=True)

    tgt = np.array(parameters["target_multiplier"], dtype=np.float64)
    stp = np.array(parameters["stop_multiplier"], dtype=np.float64)

    parts: list[pd.DataFrame] = []
    for horizon, long_sorted, short_sorted in per_horizon:
        df_long = _sorted_to_long_frame(long_sorted, "long", tgt, stp, int(horizon))
        df_short = _sorted_to_long_frame(short_sorted, "short", tgt, stp, int(horizon))
        parts.append(pd.concat([df_long, df_short], ignore_index=True))

    df = pd.concat(parts, ignore_index=True)

    h_fwd = parameters["n_candles_forward"]
    lo, hi = int(min(h_fwd)), int(max(h_fwd))
    if lo == hi:
        fname = f"pnl_summaries_forward_{lo}.parquet"
    else:
        fname = f"pnl_summaries_forward_{lo}_to_{hi}.parquet"
    path = os.path.join(atr_dir, fname)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")
