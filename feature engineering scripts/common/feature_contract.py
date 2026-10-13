"""
The single source of truth for "what does build_dataset.build_day() actually
output, and which of those columns are features vs. labels vs. identifiers".

Every model-training script (models/*/train.py), the validation harness, and
the fusion layer import from HERE rather than each re-deriving its own column
list — that's what keeps them mutually compatible: change the pipeline's
output schema once (feature_engine/*.py), refresh COLUMN_DTYPES below by
re-running scripts/dump_schema.py, and every downstream script picks up the
new contract automatically.

COLUMN_DTYPES was extracted empirically (2026-09-12) by calling
build_dataset.build_day() on a real day and dumping polars dtypes for all 226
output columns — NOT hand-copied from feature_engine source, so it reflects
what the pipeline actually emits, including any incidental column-order or
naming quirks. See col_schema.json in the project root for the raw dump.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "col_schema.json"

with open(_SCHEMA_PATH, "r", encoding="utf-8") as _f:
    COLUMN_DTYPES: dict[str, str] = json.load(_f)

# Row identifier / time index — never a feature, never dropped from output.
ID_COLUMNS = ["ts"]

# add_all_labels() (labeling.py) output — forward-looking by construction.
# MUST NEVER appear in any model's input feature vector (anti-leakage).
LABEL_COLUMNS = [
    "label_triple_barrier",     # Int8 {-1, 0, 1}: primary directional target
    "label_barrier_type",       # String: which barrier was touched (diagnostic, not a training target)
    "label_exit_bar_offset",    # Int32: bars until barrier touch (diagnostic / sample-weighting input)
    "label_is_valid",           # Boolean: False near the end of a day's data — those rows must be filtered, not trained on
    "label_return_h1",          # Float32: forward log-return, 12 bars (~2min)
    "label_return_h2",          # Float32: forward log-return, 60 bars (~10min)
    "label_return_h3",          # Float32: forward log-return, 360 bars (~1hr)
    "label_return_h4",          # Float32: forward log-return, 8640 bars (~24hr)
    "label_meta_act",           # Int8 {0, 1}: meta-labeling target (was the primary bet decisive?)
]

# Emitted by the pipeline (and kept in the parquet for diagnostics, labels,
# Kronos's OHLCV input and neuralforecast's `y`) but NOT model inputs:
#  - absolute price levels / dollar-scaled indicators: a split like
#    "close > 61k" learned in one regime is meaningless at 110k, so they make
#    walk-forward accuracy collapse. Every one has a scale-free sibling kept
#    below (log_return_*, *_pct, *_deviation, bollinger_pct_b, atr_pct, rsi).
#  - cvd / cvd_ma_*: a running sum reset at every day file's midnight, so its
#    level mostly encodes time-since-midnight x trend; volume_delta,
#    taker_buy_sell_imbalance and the OFI columns carry the flow signal.
#  - month / quarter / week-of-year cycles: a ~2-year history holds only ~2
#    cycles of each, so they act as date identifiers the trees memorize.
NON_STATIONARY_COLUMNS = [
    "open", "high", "low", "close", "vwap", "intra_bar_price_std",
    "index_price", "mark_price", "spot_close", "bybit_close",
    "bollinger_mid", "bollinger_upper", "bollinger_lower",
    "atr", "macd", "macd_signal", "macd_histogram",
    "htf_1m_close", "htf_1m_high", "htf_1m_low", "htf_5m_close", "htf_5m_high", "htf_5m_low",
    "cvd", "cvd_ma_30", "cvd_ma_180",
    "month_sin", "month_cos", "quarter_sin", "quarter_cos", "week_of_year_sin", "week_of_year_cos",
]

# Discontinued upstream by exchanges (Binance Vision liquidationSnapshot post-2024-10-14,
# Bybit premium_index/spot_index post-2020-03-10). The pipeline emits typed nulls in the
# parquet for schema consistency, but all models exclude them from input vectors:
DEAD_FEATURE_COLUMNS = [
    "bybit_premium_index", "bybit_basis_pct",
    "cross_exch_basis_divergence", "cross_exch_basis_divergence_ma_180", "cross_exch_basis_divergence_ma_2880",
    "liquidation_notional", "liquidation_notional_1h", "liquidation_side_imbalance_1h",
    "liquidation_notional_8h", "liquidation_side_imbalance_8h",
]

FEATURE_COLUMNS = [c for c in COLUMN_DTYPES
                   if c not in ID_COLUMNS and c not in LABEL_COLUMNS and c not in NON_STATIONARY_COLUMNS
                   and c not in DEAD_FEATURE_COLUMNS]

# Columns whose dtype is a Polars String/Boolean/Int category rather than a
# plain numeric feature — every numeric model (GBM excepted, which can take
# raw ints) must either encode or drop these before use.
CATEGORICAL_OR_FLAG_FEATURE_COLUMNS = [
    c for c in FEATURE_COLUMNS
    if COLUMN_DTYPES[c] in ("Int8", "Int32", "Boolean") and (
        c.startswith(("breakout_", "is_", "session_", "pre_funding", "post_funding", "friday_close_window"))
        or c in ("vol_regime_bucket", "day_of_week")
    )
]

# Split of FEATURE_COLUMNS by whether a value at bar t+k (k>0) is knowable at
# bar t — used by the neuralforecast-based sequence members (TFT/PatchTST/
# iTransformer/N-HiTS: models/{tft,patchtst,itransformer,nhits}/train.py via
# common/neuralforecast_harness.py) to populate `futr_exog_list` (fed as
# known future covariates over the forecast horizon) vs. `hist_exog_list`
# (only ever fed as past context). Only pure calendar/cyclical-time features
# are genuinely known ahead of time; everything derived from price, volume,
# order flow, funding, OI, or any other market observation is not.
FUTURE_KNOWN_FEATURE_COLUMNS = [c for c in [
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "day_of_week_sin", "day_of_week_cos", "day_of_month_sin", "day_of_month_cos",
    "month_sin", "month_cos", "quarter_sin", "quarter_cos",
    "week_of_year_sin", "week_of_year_cos",
    "is_month_start", "is_month_end_window", "is_quarter_end_window",
    "session_asian", "session_europe", "session_us", "is_weekend", "day_of_week",
    "friday_close_window", "minutes_to_funding", "minutes_from_funding",
    "pre_funding_1h", "post_funding_1h", "funding_cycle_phase",
] if c in FEATURE_COLUMNS]
HISTORICAL_ONLY_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in FUTURE_KNOWN_FEATURE_COLUMNS]

CLASSIFICATION_TARGET = "label_triple_barrier"     # {-1, 0, 1} -> remapped to {0, 1, 2} by common.data
META_LABEL_TARGET = "label_meta_act"               # {0, 1}
REGRESSION_TARGETS = ["label_return_h1", "label_return_h2", "label_return_h3", "label_return_h4"]
VALIDITY_COLUMN = "label_is_valid"

_DTYPE_BY_NAME = {
    "Float32": pl.Float32, "Float64": pl.Float64, "Int8": pl.Int8, "Int32": pl.Int32, "Int64": pl.Int64,
    "UInt32": pl.UInt32, "String": pl.String, "Boolean": pl.Boolean,
    "Datetime(time_unit='ns', time_zone=None)": pl.Datetime("ns"),
}


def conform_to_contract(df: pl.DataFrame) -> pl.DataFrame:
    """Every day file gets exactly COLUMN_DTYPES' columns in its order. The
    feature modules emit fewer columns on a source-unavailable day (no
    metrics -> no OI ratios, no spot -> no spot_close, ...) and extra ones
    when a source the contract doesn't cover shows up (bookTicker on older
    dates), and a single such day would otherwise break the vertical concat
    in common.data.load_days. Missing columns become typed nulls — exactly
    what an unavailable source already produces for the columns it does
    emit — and extras are dropped."""
    exprs = []
    for col, dtype_name in COLUMN_DTYPES.items():
        dtype = _DTYPE_BY_NAME[dtype_name]
        if col not in df.columns:
            exprs.append(pl.lit(None, dtype=dtype).alias(col))
        elif df.schema[col] != dtype:
            exprs.append(pl.col(col).cast(dtype, strict=False))
        else:
            exprs.append(pl.col(col))
    return df.select(exprs)

assert len(COLUMN_DTYPES) == 226, f"expected 226 total columns in col_schema.json, got {len(COLUMN_DTYPES)} — pipeline schema has changed, re-run scripts/dump_schema.py and review FEATURE_COLUMNS/LABEL_COLUMNS above"
assert set(LABEL_COLUMNS) <= set(COLUMN_DTYPES), "LABEL_COLUMNS references a column not present in col_schema.json"
assert set(NON_STATIONARY_COLUMNS) <= set(COLUMN_DTYPES), "NON_STATIONARY_COLUMNS references a column not present in col_schema.json"
assert len(FEATURE_COLUMNS) == len(COLUMN_DTYPES) - len(ID_COLUMNS) - len(LABEL_COLUMNS) - len(NON_STATIONARY_COLUMNS) - len(DEAD_FEATURE_COLUMNS)
