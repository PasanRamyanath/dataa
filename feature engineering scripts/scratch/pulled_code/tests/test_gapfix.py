import sys
import tempfile
import types
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import config  # noqa: E402

TMP = Path(tempfile.mkdtemp())
import common.oof as oof  # noqa: E402

oof.OOF_DIR = TMP / "oof"

# ---- synthetic multi-day data: 6 real-schema days, day index 2 missing ----
pfiles = sorted((REPO / "output").glob("features_*.parquet")) or sorted(Path(r"d:\backup\Binance data\feature engineering scripts\output").glob("features_*.parquet"))
day = pl.read_parquet(pfiles[0])
frames = []
for k in [0, 1, 3, 4, 5, 6]:
    d = day.with_columns((pl.col("ts") + timedelta(days=k)).alias("ts"),
                         (pl.col("close") * (1 + 0.001 * k)).alias("close"))
    frames.append(d)
full = pl.concat(frames).sort("ts")
print("rows", full.height, "invalid", full.filter(~pl.col("label_is_valid")).height)

from common.windowing import run_ids, window_end_indices  # noqa: E402

rid = run_ids(full, 10)
assert rid.max() == 1, rid.max()  # exactly one real gap (missing day), no day-boundary gaps
ends = window_end_indices(full, lookback=720, interval_seconds=10, valid_col="label_is_valid",
                          nonnull_cols=("label_triple_barrier",), stride=1)
assert full["label_is_valid"].to_numpy()[ends].all()
assert (rid[ends - 719] == rid[ends]).all()
gap_row = int(np.argmax(rid == 1))
assert not ((ends >= gap_row) & (ends < gap_row + 719)).any()
print("window_end_indices ok:", len(ends))

import common.data as cdata  # noqa: E402

cdata.load_days = lambda **kw: full

# ---- SSM: lazy datasets + a real (tiny) CPU training run ----
import models.ssm_challenger.train as ssm  # noqa: E402

ssm.load_days = lambda **kw: full
ssm.CHECKPOINT_DIR = TMP / "ssm"
m = ssm.run(val_days=1, max_days=None, lookback=60, stride=500, epochs=1, batch_size=32, lr=1e-3,
            upload=False, gpus=0, resume=False)
print("ssm metrics", {k: m[k] for k in ("n_train_windows", "n_val_windows", "accuracy")})
from common.cv import purged_walk_forward_splits  # noqa: E402

fold = purged_walk_forward_splits(full, n_folds=1, val_size=config.BARS_PER_DAY)[0].fold
tr, es, va, tsv, sc = ssm.build_datasets(full, fold.train_end, fold.val_start, fold.val_end, 720, 10)
x0 = tr[0]
assert x0[0].shape == (720, tr.features.shape[1]) and len(tsv) == len(va)
assert tsv.min() >= full["ts"].to_numpy()[fold.val_start]
o = pl.read_parquet(oof.OOF_DIR / "ssm-challenger.parquet")
assert len(o) == m["n_val_windows"]
print("ssm ok; lazy val windows at stride 10:", len(va), "train:", len(tr))

# ---- neuralforecast harness with a fake library faithful to cross_validation semantics ----
calls = {}


class FakeNF:
    def __init__(self, models, freq):
        self.models, self.freq = models, freq

    def cross_validation(self, df, n_windows, test_size, step_size, val_size, refit):
        h = self.models[0].h
        assert n_windows is None and refit is False and step_size == config.SEQ_OOF_STEP_BARS and (test_size - h) % step_size == 0
        assert pd.Series(df["ds"]).diff().dropna().eq(pd.Timedelta(seconds=10)).all(), "series not contiguous"
        exp_cols = set(self.models[0].kw.get("hist_exog_list", [])) | set(self.models[0].kw.get("futr_exog_list", []))
        assert exp_cols <= set(df.columns)
        calls["test_size"], calls["n"] = test_size, len(df)
        rows = []
        y = df["y"].to_numpy()
        ds = df["ds"].to_numpy()
        first = len(df) - test_size
        for c in range(first - 1, len(df) - h, step_size):
            for j in range(1, h + 1):
                rows.append((config.SYMBOL_UM, ds[c + j], ds[c], y[c + j], y[c] * 1.0001))
        out = pd.DataFrame(rows, columns=["unique_id", "ds", "cutoff", "y", type(self.models[0]).__name__])
        return out.set_index("unique_id")  # some versions return unique_id as the index

    def save(self, path, model_index, overwrite, save_dataset):
        Path(path, "fake.ckpt").write_text("x")


class _Base:
    def __init__(self, **kw):
        self.kw, self.h = kw, kw["h"]


class TFT(_Base):
    pass


class PatchTST(_Base):
    EXOGENOUS_HIST = False
    EXOGENOUS_FUTR = False


class NHITS(_Base):
    pass


class iTransformer(PatchTST):
    MULTIVARIATE = True


nf_mod = types.ModuleType("neuralforecast")
nf_mod.NeuralForecast = FakeNF
nf_models = types.ModuleType("neuralforecast.models")
for c in (TFT, PatchTST, NHITS, iTransformer):
    setattr(nf_models, c.__name__, c)
sys.modules["neuralforecast"] = nf_mod
sys.modules["neuralforecast.models"] = nf_models

from common.neuralforecast_harness import build_model  # noqa: E402

assert "hist_exog_list" not in build_model(PatchTST, 90, 720).kw
assert build_model(TFT, 90, 720).kw["futr_exog_list"]

import importlib  # noqa: E402

for name in ("tft", "patchtst", "itransformer", "nhits"):
    mod = importlib.import_module(f"models.{name}.train")
    mod.load_days = lambda **kw: full
    mod.CHECKPOINT_DIR = TMP / name
    met = mod.run(val_days=1, max_days=None, max_steps=1, upload=False, gpus=0)
    # series = last contiguous run (after the missing day) up to val_end
    assert calls["test_size"] == 8640 and calls["n"] == 4 * 8640, calls
    assert met["n_windows"] == 856, met
    o = pl.read_parquet(oof.OOF_DIR / f"{name}.parquet")
    assert len(o) == met["n_windows"]
    print(name, "ok", {k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()})

# ---- Kronos: val windows skip gaps ----
import models.kronos_finetune.train as kr  # noqa: E402


class FakePred:
    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
        return pd.DataFrame({"close": np.full(pred_len, df["close"].iloc[-1])})


gappy_val = full.slice(8640 * 2 - 3000, 6000)  # spans the missing day
mk, ts_, pr = kr.evaluate_on_val(FakePred(), gappy_val, 512, 90)
assert mk["n_windows"] > 0, mk
print("kronos ok", mk)
print("ALL OK")
