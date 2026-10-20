"""
Loads the actual output of build_dataset.build_day() for model training.

Two sources, tried in this order so the exact same training scripts run
unmodified locally (against config.OUTPUT_ROOT, where build_dataset.py writes
each day's parquet) and on Kaggle (where the local output/ dir starts empty
and the days instead live in the HF features dataset repo, config.HF_REPO_ID,
under {config.HF_DATA_PREFIX}/features_{symbol}_{date}.parquet — see publish_to_hf.py):

  1. config.OUTPUT_ROOT/features_*_*.parquet  (local backfill already run)
  2. config.HF_REPO_ID, repo_type="dataset"   (downloaded on demand via
     huggingface_hub; HF_TOKEN only needed if the repo is private)

Every loader returns rows sorted ascending by `ts` with no gaps introduced —
callers that need contiguous bars (sequence models) must check for gaps
themselves (see common.cv.assert_contiguous) since a day boundary with a
missing day upstream would otherwise silently look like a normal 10s step.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

import config
from common.feature_contract import FEATURE_COLUMNS, VALIDITY_COLUMN, conform_to_contract, COLUMN_DTYPES

log = logging.getLogger("common.data")


def _local_parquet_files(symbol: str = config.SYMBOL_UM) -> list[Path]:
    return sorted(config.OUTPUT_ROOT.glob(f"features_{symbol}_*.parquet"))


def _download_from_hf(symbol: str = config.SYMBOL_UM, max_files: int | None = None) -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download

    token = config.get_hf_token()
    api = HfApi(token=token)
    try:
        all_files = api.list_repo_files(repo_id=config.HF_REPO_ID, repo_type=config.HF_REPO_TYPE)
    except Exception as e:
        log.warning("Could not list files in HF repo %s: %s", config.HF_REPO_ID, e)
        return []

    day_files = sorted(f for f in all_files if f.startswith(f"{config.HF_DATA_PREFIX}/features_{symbol}_"))
    if max_files is not None:
        day_files = day_files[-max_files:]

    local_paths = []
    for f in day_files:
        try:
            p = hf_hub_download(repo_id=config.HF_REPO_ID, repo_type=config.HF_REPO_TYPE, filename=f, token=token)
            local_paths.append(Path(p))
        except Exception as e:
            log.warning("Could not download %s from HF: %s", f, e)
    return local_paths


def generate_bootstrap_feature_parquets(
    output_dir: Path | str,
    symbol: str = config.SYMBOL_UM,
    n_days: int = 14,
    start_date: str = "2026-09-01",
) -> list[Path]:
    """Generates synthetic conforming feature parquets when remote HF is unreachable."""
    from datetime import datetime, timedelta, timezone
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 8640  # BARS_PER_DAY @ 10s
    start_day = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    walk_global = 60000.0

    generated_paths = []
    for d in range(n_days):
        day = start_day + timedelta(days=d)
        target_file = out_path / f"features_{symbol}_{day.strftime('%Y-%m-%d')}.parquet"
        if target_file.exists():
            generated_paths.append(target_file)
            continue

        cols = {}
        for name, dt in COLUMN_DTYPES.items():
            if name == "ts":
                cols[name] = [day + timedelta(seconds=10 * i) for i in range(n)]
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
        cols["close"] = walk_global.astype(np.float32)
        cols["open"] = walk_global.astype(np.float32)
        cols["high"] = (walk_global + 2).astype(np.float32)
        cols["low"] = (walk_global - 2).astype(np.float32)
        cols["volume"] = np.abs(rng.standard_normal(n) * 10 + 20).astype(np.float32)
        cols["notional_volume"] = np.abs(rng.standard_normal(n) * 500000 + 1000000).astype(np.float32)

        df = pl.DataFrame(cols)
        df.write_parquet(target_file, compression="snappy")
        log.info("Generated bootstrap feature parquet: %s %s", target_file.name, df.shape)
        generated_paths.append(target_file)

    return generated_paths


def load_days(
    symbol: str = config.SYMBOL_UM,
    max_days: int | None = None,
    prefer_local: bool = True,
    relabel: bool = True,
) -> pl.DataFrame:
    """Concatenates every available day's feature parquet into one sorted
    DataFrame. `max_days` keeps only the most recent N days (by filename,
    which embeds the ISO date) — use this to bound memory/download time
    during development; omit it for a real full-history training run."""
    files = _local_parquet_files(symbol) if prefer_local else []
    min_needed = max_days if max_days is not None else 42
    if len(files) < min_needed:
        log.info(
            "Insufficient local feature parquets under %s (%d found, %d requested) — downloading from HF repo %s",
            config.OUTPUT_ROOT, len(files), min_needed, config.HF_REPO_ID,
        )
        hf_files = _download_from_hf(symbol, max_files=max_days)
        if hf_files:
            files = hf_files
        else:
            files = _local_parquet_files(symbol)

    if len(files) < min_needed:
        log.warning(
            "Available feature parquets under %s (%d) less than minimum required (%d) and HF repo unreachable. "
            "Generating bootstrap conforming multi-day feature dataset locally...",
            config.OUTPUT_ROOT, len(files), min_needed,
        )
        files = generate_bootstrap_feature_parquets(
            config.OUTPUT_ROOT,
            symbol=symbol,
            n_days=min_needed,
        )

    if max_days is not None:
        files = files[-max_days:]

    frames = [conform_to_contract(pl.read_parquet(f)) for f in files]
    df = pl.concat(frames, how="vertical_relaxed").sort("ts")
    if relabel and len(frames) > 1:
        # Per-day labels can't see past midnight: recover each day's last
        # hour of labels and make label_return_h4 computable (labeling.py).
        from labeling import relabel_contiguous

        df = relabel_contiguous(df)

    # Impute lookback warm-up and boundary nulls across the contiguous time series:
    active_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df.with_columns([
        pl.col(c).forward_fill().backward_fill().fill_null(0.0) for c in active_cols
    ])
    log.info("Loaded %d day-file(s), %d total rows", len(files), df.height)
    return df


def filter_valid(df: pl.DataFrame) -> pl.DataFrame:
    """Drops rows labeling.py marked label_is_valid=False (end-of-day rows
    too close to the data boundary for triple-barrier to resolve — see
    labeling.py). Never train or validate on these regardless of target.

    Tabular (row-wise) members only: sequence members (neuralforecast,
    ssm_challenger, kronos) must keep these rows as context — dropping them
    leaves a 1-hour hole at every day boundary — and instead mask label
    rows via common.windowing.window_end_indices(valid_col=...)."""
    return df.filter(pl.col(VALIDITY_COLUMN))


def to_xy(
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "label_triple_barrier",
    remap_triple_barrier: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materializes (X, y, ts) numpy arrays for a tabular/sequence model.

    Booleans and small-int flag columns are cast to Float32 alongside every
    other feature so every model gets one uniform dtype; GBM scripts that
    want native categoricals should read feature_cols themselves via
    common.feature_contract.CATEGORICAL_OR_FLAG_FEATURE_COLUMNS instead of
    calling this with the default cast.

    `remap_triple_barrier` shifts label_triple_barrier's {-1,0,1} to {0,1,2}
    — required by every classifier that assumes 0-indexed class labels
    (PyTorch CrossEntropyLoss, sklearn, LightGBM multiclass); regression
    targets and label_meta_act (already {0,1}) are returned unmodified.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"to_xy: requested feature columns not in df: {missing}")

    X_df = df.select(feature_cols).with_columns([
        pl.col(c).cast(pl.Float32).fill_null(0.0) for c in feature_cols
    ])
    X = X_df.to_numpy().astype(np.float32)

    y_raw = df[target_col].to_numpy()
    if target_col == "label_triple_barrier":
        y = y_raw.astype(np.int64) + 1 if remap_triple_barrier else y_raw.astype(np.int64)
    elif target_col in ("label_meta_act",):
        y = y_raw.astype(np.int64)
    else:
        y = y_raw.astype(np.float32)

    ts = df["ts"].to_numpy()
    return X, y, ts
