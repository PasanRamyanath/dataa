import sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
import config
from common.feature_contract import FEATURE_COLUMNS, LABEL_COLUMNS, VALIDITY_COLUMN

df = pl.read_parquet(config.OUTPUT_ROOT / "features_BTCUSDT_2026-09-06.parquet")
print("shape", df.shape, "ts", df["ts"].min(), df["ts"].max(), df["ts"].dtype)
print("label cols:", LABEL_COLUMNS, "validity:", VALIDITY_COLUMN)
missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
extra = [c for c in df.columns if c not in FEATURE_COLUMNS and c not in LABEL_COLUMNS and c not in (VALIDITY_COLUMN, "ts")]
print("contract features:", len(FEATURE_COLUMNS), "missing:", len(missing), missing[:40])
print("extra non-contract cols:", extra)
valid = df.filter(pl.col(VALIDITY_COLUMN)) if VALIDITY_COLUMN in df.columns else df
print("valid rows", valid.height, "/", df.height)
for c in LABEL_COLUMNS:
    if c in df.columns:
        s = valid[c]
        if s.dtype in (pl.Float32, pl.Float64):
            print(f"  {c}: nulls={s.null_count()} mean={s.mean():.3g} std={s.std():.3g}")
        else:
            print(f"  {c}: nulls={s.null_count()}", s.value_counts().sort(c).to_dicts())

present = [c for c in FEATURE_COLUMNS if c in df.columns]
X = valid.select([pl.col(c).cast(pl.Float64) for c in present])
nulls = {c: X[c].null_count() / X.height for c in present}
print("features >50% null:", sorted([(c, round(v, 2)) for c, v in nulls.items() if v > 0.5], key=lambda t: -t[1])[:30])
nan_inf = [c for c in present if X[c].is_nan().sum() or X[c].is_infinite().sum()]
print("features with NaN/inf:", nan_inf[:30])
const = [c for c in present if X[c].drop_nulls().n_unique() <= 1]
print("constant features:", len(const), const[:40])

# leakage screen: |corr| of each feature with forward labels on valid rows
tgt_cols = [c for c in LABEL_COLUMNS if c in valid.columns and valid[c].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)]
M = X.fill_null(0).fill_nan(0).to_numpy()
M[~np.isfinite(M)] = 0
for t in tgt_cols:
    y = valid[t].cast(pl.Float64).fill_null(0).to_numpy()
    if y.std() == 0:
        continue
    sd = M.std(0)
    ok = sd > 0
    corr = np.zeros(M.shape[1])
    corr[ok] = ((M[:, ok] - M[:, ok].mean(0)) * (y - y.mean())[:, None]).mean(0) / (sd[ok] * y.std())
    top = np.argsort(-np.abs(corr))[:6]
    print(f"top |corr| with {t}:", [(present[i], round(float(corr[i]), 3)) for i in top])
