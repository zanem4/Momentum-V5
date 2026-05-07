# backtest runner

# module imports
import os
import sys
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import time
import random

# add directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# file imports
from utils.composite_candle import composite_candle
from strategy.structure import structure
from strategy.atr import calculate_atr
from strategy.calculate_metrics import calculate_metrics
from strategy.quantile import calculate_quantiles
from strategy.sim_trades import simulate_trades
from strategy.sort_trades import sort_trades
from strategy.save_data import save_data, save_edge_data_json


def _count_simulations(n_long: int, n_short: int, n_target: int, n_stop: int) -> int:
    """One simulation = one setup × one target mult × one stop mult (per direction)."""
    return n_long * n_target * n_stop + n_short * n_target * n_stop


def _write_throughput_report(
    main_dir: str,
    total_sims: int,
    total_elapsed_s: float,
    by_timeframe: dict,
) -> None:
    path = os.path.join(main_dir, "simulation_throughput.txt")
    lines = [
        "Simulation throughput",
        "=====================",
        "",
        "A simulation is one (setup, target_multiplier, stop_multiplier) evaluation",
        "per direction (long and short simulate_trades each horizon).",
        "",
        "Per timeframe: one wall clock from start to end of that timeframe iteration",
        "(composite_candle, all lookbacks/ATRs/metrics/quantiles/saves, and all sim work).",
        "",
        "Per timeframe (minutes)",
        "------------------------",
    ]

    tfs = sorted(by_timeframe.keys())
    sum_seconds = 0.0
    sum_sims_tf = 0
    sims_per_sec_list: list[float] = []
    sec_per_sim_list: list[float] = []

    for tf in tfs:
        b = by_timeframe[tf]
        sims = int(b["simulations"])
        sec = float(b["seconds"])
        sps = float(b["sims_per_sec"])
        spsim = float(b["sec_per_sim"])
        sum_seconds += sec
        sum_sims_tf += sims
        sims_per_sec_list.append(sps)
        sec_per_sim_list.append(spsim)
        spsim_str = f"{spsim:.6e}" if np.isfinite(spsim) else "n/a"
        lines.append(
            f"  {tf}m:  {sims} sims,  {sec:.3f} s,  {sps:,.2f} sim/s,  {spsim_str} s/sim"
        )

    mean_sps = float(np.nanmean(np.asarray(sims_per_sec_list, dtype=np.float64)))
    mean_spsim = float(np.nanmean(np.asarray(sec_per_sim_list, dtype=np.float64)))

    lines.extend(
        [
            "",
            "Summary (across timeframes)",
            "---------------------------",
            f"  Total simulations (sum of per-timeframe): {sum_sims_tf}",
            f"  Sum of timeframe seconds: {sum_seconds:.3f}",
            f"  Mean sims/s (simple mean over timeframes): {mean_sps:,.2f}",
            f"  Mean s/sim (simple mean over timeframes): {mean_spsim:.6e}",
            "",
            f"Whole-run wall clock (s): {total_elapsed_s:.3f}  |  total simulations (run counter): {total_sims}",
        ]
    )

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():

    # read parameters file
    with open("utils/parameters.json", "r") as f:
        parameters = json.load(f)
    
    # determine symbol and data path
    data_path = parameters["data_path"]
    if parameters["use_random_asset"]:
        symbol = random.choice(list(parameters["pip_value_by_symbol"].keys()))
        data_path = os.path.join(data_path, f"{symbol}.parquet")
    else:
        symbol = f"{parameters["base"]}_{parameters["quote"]}"
        data_path = os.path.join(data_path, f"{symbol}.parquet")
    
    # read data
    data = pq.read_table(data_path).to_pandas().to_numpy()

    """
    since hypothesis considers momentum (acceleration) and liquidity state as crucial,
    those metrics need first iteration for global analysis.

    ATR is the metric that determines the normalization of acceleration, so it has to be iterated.
    for stability, quantile edges must be calculated globally before rolling window split.
    """

    # make main directory
    main_dir = os.path.join("output", f"{symbol}_{parameters["start_date"]}_{parameters["end_date"]}")
    os.makedirs(main_dir, exist_ok=True)

    n_t = len(parameters["target_multiplier"])
    n_s = len(parameters["stop_multiplier"])
    total_sims = 0
    by_timeframe: dict = {}
    start_perf = time.perf_counter()

    # iterate through timeframes first, as timeframe affects ATR
    for timeframe in parameters["timeframe_minutes"]:

        t_tf0 = time.perf_counter()
        sims_tf = 0

        # make timeframe directory for each timeframe
        timeframe_dir = os.path.join(main_dir, f"{timeframe}m")
        os.makedirs(timeframe_dir, exist_ok=True)

        # create composite candles
        composite_candles, m1_indices = composite_candle(data, timeframe)

        # iterate through all possible L-values
        for l_value in parameters["n_candles_back"]:

            # make L-value directory for each L-value
            l_value_dir = os.path.join(timeframe_dir, f"lookback_{l_value}")
            os.makedirs(l_value_dir, exist_ok=True)

            # detect setups
            long_indices = structure(composite_candles, l_value, "long")
            short_indices = structure(composite_candles, l_value, "short")

            # drop setups that cannot complete max forward horizon (keep arrays aligned through quantiles → sim → sort)
            H_max = max(parameters["n_candles_forward"])
            n_comp = len(composite_candles)
            long_indices = long_indices[long_indices < n_comp - H_max]
            short_indices = short_indices[short_indices < n_comp - H_max]

            print(f"Number of setups for L={l_value} and timeframe={timeframe}: {len(long_indices) + len(short_indices)}")

            # iterate through ATR values before metrics since ATR is crucial in norm accel calculation
            for atr_length in parameters["atr_length"]:

                # make ATR directory for each ATR value
                atr_dir = os.path.join(l_value_dir, f"atr_{atr_length}")
                os.makedirs(atr_dir, exist_ok=True)

                # calculate ATR for each setup
                pip_value = parameters["pip_value_by_symbol"][symbol]
                long_indices, long_atr = calculate_atr(composite_candles, long_indices, atr_length, pip_value)
                short_indices, short_atr = calculate_atr(composite_candles, short_indices, atr_length, pip_value)

                # calculate joint metrics for each setup
                long_metrics = calculate_metrics(composite_candles, long_indices, long_atr, "long", l_value, pip_value)
                short_metrics = calculate_metrics(composite_candles, short_indices, short_atr, "short", l_value, pip_value)

                # globally calculate quantiles before further testing
                long_indices, long_edge_data = calculate_quantiles(long_metrics, long_indices)
                short_indices, short_edge_data = calculate_quantiles(short_metrics, short_indices)

                # save direction-specific edge data (JSON; numpy → lists)
                save_edge_data_json(atr_dir, long_edge_data, short_edge_data, parameters)

                # simulate all forward time horizons, target and stop level combinations irrespective of time horizon
                t_atr0 = time.perf_counter()
                sims_atr = 0
                per_horizon: list[tuple[int, dict, dict]] = []

                target_multipliers = np.array(parameters["target_multiplier"])
                stop_multipliers = np.array(parameters["stop_multiplier"])

                plot_extra = {
                    "symbol": symbol,
                    "timeframe_minutes": int(timeframe),
                    "atr_length": int(atr_length),
                    "pip_value": float(pip_value),
                }

                horizons = list(parameters["n_candles_forward"])
                sample_horizon = horizons[len(horizons) // 2]
                for horizon in horizons:
                    do_sample = int(horizon) == int(sample_horizon)
                    long_sample_path = os.path.join(atr_dir, f"trade_sample_long_H{horizon}.json") if do_sample else None
                    short_sample_path = os.path.join(atr_dir, f"trade_sample_short_H{horizon}.json") if do_sample else None

                    long_pnl = simulate_trades(
                        composite_candles,
                        long_indices,
                        pip_value,
                        "long",
                        horizon,
                        target_multipliers,
                        stop_multipliers,
                        long_metrics,
                        plot_sample_path=long_sample_path,
                        plot_sample_l_value=l_value,
                        plot_sample_extra=dict(plot_extra),
                        plot_sample_mt=0,
                        plot_sample_ms=0,
                    )
                    short_pnl = simulate_trades(
                        composite_candles,
                        short_indices,
                        pip_value,
                        "short",
                        horizon,
                        target_multipliers,
                        stop_multipliers,
                        short_metrics,
                        plot_sample_path=short_sample_path,
                        plot_sample_l_value=l_value,
                        plot_sample_extra=dict(plot_extra),
                        plot_sample_mt=0,
                        plot_sample_ms=0,
                    )

                    long_sorted = sort_trades(long_pnl, long_indices, long_edge_data, parameters, composite_candles)
                    short_sorted = sort_trades(short_pnl, short_indices, short_edge_data, parameters, composite_candles)

                    sims = _count_simulations(len(long_indices), len(short_indices), n_t, n_s)
                    total_sims += sims
                    sims_tf += sims
                    sims_atr += sims

                    per_horizon.append((int(horizon), long_sorted, short_sorted))

                save_data(atr_dir, per_horizon, parameters)

                elapsed_atr = time.perf_counter() - t_atr0
                rate_atr = sims_atr / elapsed_atr if elapsed_atr > 0 else 0.0
                print(
                    f"[throughput] {timeframe}m  lookback={l_value}  atr={atr_length}  |  "
                    f"{sims_atr} sims in {elapsed_atr:.2f}s  =>  {rate_atr:,.0f} sim/s"
                )

        elapsed_tf_wall = time.perf_counter() - t_tf0
        rate_tf = sims_tf / elapsed_tf_wall if elapsed_tf_wall > 0 else float("nan")
        sec_per_sim_tf = elapsed_tf_wall / sims_tf if sims_tf > 0 else float("nan")
        by_timeframe[timeframe] = {
            "simulations": sims_tf,
            "seconds": elapsed_tf_wall,
            "sims_per_sec": rate_tf,
            "sec_per_sim": sec_per_sim_tf,
        }
        spsim_s = f"{sec_per_sim_tf:.4e}" if np.isfinite(sec_per_sim_tf) else "n/a"
        print(
            f"[throughput] timeframe {timeframe}m: {sims_tf} sims, {elapsed_tf_wall:.2f}s, "
            f"{rate_tf:,.0f} sim/s, {spsim_s} s/sim"
        )

    total_elapsed = time.perf_counter() - start_perf
    _write_throughput_report(main_dir, total_sims, total_elapsed, by_timeframe)

    tfs = sorted(by_timeframe.keys())
    sum_sec = sum(float(by_timeframe[tf]["seconds"]) for tf in tfs)
    mean_sps = float(np.nanmean(np.array([by_timeframe[tf]["sims_per_sec"] for tf in tfs], dtype=np.float64))) if tfs else float("nan")
    mean_spsim = float(np.nanmean(np.array([by_timeframe[tf]["sec_per_sim"] for tf in tfs], dtype=np.float64))) if tfs else float("nan")
    print(
        f"[throughput] run complete: sum_tf_s={sum_sec:.2f}s, mean_sims/s={mean_sps:,.0f}, mean_s/sim={mean_spsim:.4e}  "
        f"| wall {total_elapsed:.2f}s, {total_sims} sims  |  {os.path.join(main_dir, 'simulation_throughput.txt')}"
    )


if __name__ == "__main__":
    main()