"""
feature engineering scripts/parsers/multi_tf_parser.py
Loads and normalizes all 13 native kline resolutions (1s, 1m, 3m, 5m, 15m, 30m,
1h, 2h, 4h, 6h, 8h, 12h, 1d) plus 10s aggTrade bars into causally aligned
Polars DataFrames where every bar carries both `open_ts` and `close_ts` (nanoseconds),
and cross-timeframe joins are strictly keyed on `decision_ts >= close_ts`.
"""
from __future__ import annotations

from pathlib import Path
import zipfile
import numpy as np
import polars as pl

from schemas import KLINE_COLUMNS, KLINE_DTYPES
from utils import normalize_epoch_column_to_ns, safe_div

INTERVAL_SECONDS_MAP: dict[str, int] = {
    "1s": 1,
    "10s": 10,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def read_kline_archive_series(
    zip_paths: list[Path],
    market: str,
    interval: str,
) -> pl.DataFrame:
    """
    Reads a chronological list of monthly/daily kline .zip archives for a given
    market ('spot', 'um', 'cm') and interval ('1s' .. '1d'), normalizes timestamps
    to UTC nanoseconds, computes exact causal `close_ts = open_ts + interval_ns`,
    and derives core microstructure columns (vwap, taker_sell_volume, volume_delta,
    taker_buy_ratio, avg_trade_notional).
    """
    frames: list[pl.DataFrame] = []
    has_header = market in ("um", "cm")
    interval_ns = INTERVAL_SECONDS_MAP[interval] * 1_000_000_000

    for zp in sorted(zip_paths):
        if not zp.exists() or zp.stat().st_size == 0:
            continue
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    continue
                with zf.open(csv_names[0]) as raw_csv:
                    first_byte = raw_csv.peek(1)[:1]
                    file_has_header = has_header or (first_byte and not first_byte.isdigit())
                    df = pl.read_csv(
                        raw_csv,
                        has_header=bool(file_has_header),
                        new_columns=None if file_has_header else KLINE_COLUMNS,
                        schema_overrides=KLINE_DTYPES,
                        ignore_errors=True,
                    )
                    if not file_has_header and df.columns != KLINE_COLUMNS:
                        df = df.rename(dict(zip(df.columns, KLINE_COLUMNS)))
                    frames.append(df)
        except Exception:
            continue

    if not frames:
        return pl.DataFrame()

    combined = pl.concat(frames, how="vertical_relaxed")
    combined = normalize_epoch_column_to_ns(combined, "open_time", out_col="open_ts")
    combined = combined.with_columns([
        pl.from_epoch(pl.col("open_ts"), time_unit="ns").alias("open_dt"),
        pl.from_epoch(pl.col("open_ts") + interval_ns, time_unit="ns").alias("ts"),  # Canonical CLOSE availability ts!
    ]).sort("ts").unique(subset=["ts"], keep="last")

    # Derive intrinsic kline order-flow columns available across all 13 resolutions
    taker_sell_vol = (pl.col("volume") - pl.col("taker_buy_volume")).clip(lower_bound=0.0)
    combined = combined.with_columns([
        taker_sell_vol.alias("taker_sell_volume"),
        (pl.col("taker_buy_volume") - taker_sell_vol).alias("volume_delta"),
        safe_div(pl.col("taker_buy_volume") - taker_sell_vol, pl.col("volume")).alias("taker_buy_sell_imbalance"),
        safe_div(pl.col("quote_volume"), pl.col("volume")).alias("vwap"),
        safe_div(pl.col("quote_volume"), pl.col("count").cast(pl.Float64)).alias("avg_trade_quote_size"),
        ((pl.col("high") - pl.col("low")) / pl.col("close").clip(lower_bound=1e-8)).alias("bar_range_pct"),
        ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-8)).alias("bar_body_efficiency"),
    ])
    return combined


def causal_multi_timeframe_asof_join(
    base_df: pl.DataFrame,
    htf_df: pl.DataFrame,
    htf_prefix: str,
    feature_cols: list[str],
    base_ts_col: str = "ts",
    htf_close_ts_col: str = "ts",
) -> pl.DataFrame:
    """
    Strictly causal join: both `base_df[base_ts_col]` and `htf_df[htf_close_ts_col]`
    represent the bar's COMPLETION timestamp (`open_ts + interval`).
    Thus `strategy="backward"` guarantees that `base_df` at decision time `t`
    only ever sees an `htf_df` row whose `close_ts <= t` (100% closed prior to `t`).
    """
    if htf_df.is_empty():
        return base_df.with_columns([
            pl.lit(0.0, dtype=pl.Float32).alias(f"{htf_prefix}_{c}") for c in feature_cols
        ])

    renamed_map = {c: f"{htf_prefix}_{c}" for c in feature_cols if c in htf_df.columns}
    htf_sub = (
        htf_df
        .select([htf_close_ts_col, *renamed_map.keys()])
        .rename(renamed_map)
        .sort(htf_close_ts_col)
    )
    joined = base_df.sort(base_ts_col).join_asof(
        htf_sub,
        left_on=base_ts_col,
        right_on=htf_close_ts_col,
        strategy="backward",
    )
    return joined
