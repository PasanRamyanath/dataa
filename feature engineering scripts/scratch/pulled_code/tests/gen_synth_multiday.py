import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

schema_path = sys.argv[1]
out_dir = sys.argv[2]
n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 20
start_date = sys.argv[4] if len(sys.argv) > 4 else "2026-01-01"

with open(schema_path, "r", encoding="utf-8") as f:
    dtypes = json.load(f)

rng = np.random.default_rng(0)
n = 8640  # BARS_PER_DAY @ 10s
start_day = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

import os
os.makedirs(out_dir, exist_ok=True)

walk_global = 50000.0
for d in range(n_days):
    day = start_day + timedelta(days=d)
    start = day
    cols = {}
    for name, dt in dtypes.items():
        if name == "ts":
            cols[name] = [start + timedelta(seconds=10 * i) for i in range(n)]
        elif name == "label_triple_barrier":
            cols[name] = rng.integers(-1, 2, size=n).astype(np.int8)
        elif name == "label_barrier_type":
            cols[name] = rng.choice(["upper", "lower", "time"], size=n)
        elif name == "label_exit_bar_offset":
            cols[name] = rng.integers(1, 100, size=n).astype(np.int32)
        elif name == "label_is_valid":
            valid = np.ones(n, dtype=bool)
            valid[-50:] = False
            cols[name] = valid
        elif name == "label_meta_act":
            cols[name] = rng.integers(0, 2, size=n).astype(np.int8)
        elif name.startswith("label_return"):
            cols[name] = (rng.standard_normal(n) * 0.001).astype(np.float32)
        elif dt == "Float32":
            cols[name] = (rng.standard_normal(n) * 0.5 + 1.0).astype(np.float32)
        elif dt == "Int8":
            cols[name] = rng.integers(0, 3, size=n).astype(np.int8)
        elif dt == "Int32":
            cols[name] = rng.integers(0, 100, size=n).astype(np.int32)
        elif dt == "UInt32":
            cols[name] = rng.integers(0, 100, size=n).astype(np.uint32)
        elif dt == "Boolean":
            cols[name] = rng.integers(0, 2, size=n).astype(bool)
        elif dt == "String":
            cols[name] = rng.choice(["a", "b"], size=n)
        else:
            cols[name] = (rng.standard_normal(n)).astype(np.float32)

    walk_global = walk_global + np.cumsum(rng.standard_normal(n) * 5)
    if "close" in cols:
        cols["close"] = walk_global.astype(np.float32)
        cols["open"] = walk_global.astype(np.float32)
        cols["high"] = (walk_global + 1).astype(np.float32)
        cols["low"] = (walk_global - 1).astype(np.float32)
    if "volume" in cols:
        cols["volume"] = np.abs(rng.standard_normal(n) * 10 + 20).astype(np.float32)
    if "notional_volume" in cols:
        cols["notional_volume"] = np.abs(rng.standard_normal(n) * 500000 + 1000000).astype(np.float32)

    df = pl.DataFrame(cols)
    out_path = f"{out_dir}/features_BTCUSDT_{day.strftime('%Y-%m-%d')}.parquet"
    df.write_parquet(out_path, compression="snappy")
    print("wrote", out_path, df.shape)
