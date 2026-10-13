import sys

import numpy as np
import polars as pl

synth_dir = sys.argv[1]
oof_dir = sys.argv[2]

import glob
import os

files = sorted(glob.glob(f"{synth_dir}/features_BTCUSDT_*.parquet"))
frames = [pl.read_parquet(f).select(["ts", "label_triple_barrier", "label_is_valid"]) for f in files]
df = pl.concat(frames).sort("ts").filter(pl.col("label_is_valid"))

ts = df["ts"].to_numpy()
y = df["label_triple_barrier"].to_numpy().astype(np.int64) + 1  # {0,1,2}

rng = np.random.default_rng(1)
os.makedirs(oof_dir, exist_ok=True)

# "good" member: mostly correct (70% match true label, rest random)
n = len(y)
good_proba = np.full((n, 3), 0.05, dtype=np.float32)
correct_mask = rng.random(n) < 0.7
for i in range(n):
    if correct_mask[i]:
        good_proba[i, y[i]] = 0.9
    else:
        good_proba[i, rng.integers(0, 3)] = 0.9
good_proba = good_proba / good_proba.sum(axis=1, keepdims=True)
pl.DataFrame({
    "ts": ts, "prob_down": good_proba[:, 0], "prob_flat": good_proba[:, 1], "prob_up": good_proba[:, 2],
}).write_parquet(f"{oof_dir}/test-member-good.parquet", compression="snappy")

# "random" member: pure noise, should fail kill criteria
rand_proba = rng.dirichlet([1, 1, 1], size=n).astype(np.float32)
pl.DataFrame({
    "ts": ts, "prob_down": rand_proba[:, 0], "prob_flat": rand_proba[:, 1], "prob_up": rand_proba[:, 2],
}).write_parquet(f"{oof_dir}/test-member-random.parquet", compression="snappy")

print("wrote synthetic OOF for", n, "rows, ts range", ts.min(), ts.max())
