"""Purged within-day walk-forward sanity check: leaky (old) vs fixed features."""
import sys
import warnings

import numpy as np
import polars as pl

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from common.feature_contract import FEATURE_COLUMNS


def run(path, label):
    df = pl.read_parquet(path).filter(pl.col("label_is_valid"))
    feats = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    n = len(X)
    split, purge = int(n * 0.70), 360  # purge = max label horizon used below
    tr, te = slice(0, split - purge), slice(split, n)
    print(f"\n== {label}: {len(feats)} features, train {split - purge} rows, test {n - split} rows")
    for tgt, h in (("label_return_h1", 12), ("label_return_h2", 60)):
        y = df[tgt].to_numpy().astype(np.float64)
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=0)
        ok_tr = np.isfinite(y[tr])
        m.fit(X[tr][ok_tr], y[tr][ok_tr])
        p, yt = m.predict(X[te]), y[te]
        ok = np.isfinite(yt)
        p, yt = p[ok], yt[ok]
        ic = np.corrcoef(p, yt)[0, 1]
        hit = np.mean(np.sign(p) == np.sign(yt))
        base_hit = max(np.mean(yt > 0), np.mean(yt < 0))
        # non-overlapping subsample (every h-th row) for an honest hit-rate
        sub = slice(0, None, h)
        hit_no = np.mean(np.sign(p[sub]) == np.sign(yt[sub]))
        print(f"  {tgt}: IC={ic:+.3f}  sign-hit={hit:.3f} (non-overlap {hit_no:.3f}, n={len(yt[sub])})  "
              f"always-majority={base_hit:.3f}")
    y = df["label_triple_barrier"].to_numpy().astype(int) + 1
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=0)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    maj = np.bincount(y[tr]).argmax()
    print(f"  triple_barrier: acc={accuracy_score(y[te], p):.3f} bal_acc={balanced_accuracy_score(y[te], p):.3f}  "
          f"majority acc={np.mean(y[te] == maj):.3f}")


run("output/features_BTCUSDT_2026-09-06.parquet", "OLD (leaky, stale)")
run(sys.argv[1], "FIXED (rebuilt)")
