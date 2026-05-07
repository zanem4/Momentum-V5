# individual parquet to CSV conversions
import sys
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


def main(path: str) -> None:
    df = pq.read_table(path).to_pandas()
    p = Path(path).resolve()
    parts = p.parts
    out_i = next((i for i, seg in enumerate(parts) if seg.lower() == "output"), None)
    if out_i is None or out_i + 1 >= len(parts):
        backtest_main_path = p.parent
    else:
        backtest_main_path = Path(*parts[: out_i + 2])
    save_path = backtest_main_path / "Sample CSV"
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path / f"{p.stem}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
