"""
Category 10: known-future-at-inference-time calendar/session features.

These are the only features in the whole pipeline that are legitimately
safe to compute for timestamps beyond the last observed bar (a TFT-style
model's "known future inputs"), because they are pure functions of the
calendar and Binance's fixed 8-hour UTC funding schedule (00:00/08:00/16:00),
not derived from observed market data.
"""
from __future__ import annotations

import math

import polars as pl

FUNDING_HOURS_UTC = [0, 8, 16]


def add_temporal_features(df: pl.DataFrame, ts_col: str = "ts") -> pl.DataFrame:
    dt = pl.col(ts_col)
    if df.schema[ts_col] != pl.Datetime:
        dt = pl.from_epoch(pl.col(ts_col), time_unit="ns")

    hour = dt.dt.hour()
    minute = dt.dt.minute()
    dow = dt.dt.weekday()          # 1=Mon .. 7=Sun
    dom = dt.dt.day()
    month = dt.dt.month()
    quarter = dt.dt.quarter()
    week = dt.dt.week()
    # dt.dt.hour()/minute() are Int8: hour*60 alone (up to 1380) overflows
    # Int8's 127 max and silently wraps to negative values, so cast to a
    # wide integer type before combining them.
    minute_of_day = hour.cast(pl.Int32) * 60 + minute.cast(pl.Int32)

    df = df.with_columns(
        (2 * math.pi * hour / 24).sin().alias("hour_sin"), (2 * math.pi * hour / 24).cos().alias("hour_cos"),
        (2 * math.pi * minute / 60).sin().alias("minute_sin"), (2 * math.pi * minute / 60).cos().alias("minute_cos"),
        (2 * math.pi * dow / 7).sin().alias("day_of_week_sin"), (2 * math.pi * dow / 7).cos().alias("day_of_week_cos"),
        (2 * math.pi * dom / 31).sin().alias("day_of_month_sin"), (2 * math.pi * dom / 31).cos().alias("day_of_month_cos"),
        (2 * math.pi * month / 12).sin().alias("month_sin"), (2 * math.pi * month / 12).cos().alias("month_cos"),
        (2 * math.pi * quarter / 4).sin().alias("quarter_sin"), (2 * math.pi * quarter / 4).cos().alias("quarter_cos"),
        (2 * math.pi * week / 52).sin().alias("week_of_year_sin"), (2 * math.pi * week / 52).cos().alias("week_of_year_cos"),
    )

    df = df.with_columns(
        (dom == 1).cast(pl.Int8).alias("is_month_start"),
        ((hour >= 0) & (hour < 8)).cast(pl.Int8).alias("session_asian"),
        ((hour >= 8) & (hour < 16)).cast(pl.Int8).alias("session_europe"),
        ((hour >= 16) & (hour < 24)).cast(pl.Int8).alias("session_us"),
        (dow >= 6).cast(pl.Int8).alias("is_weekend"),
        dow.alias("day_of_week"),
        ((dow == 5) & (hour >= 20)).cast(pl.Int8).alias("friday_close_window"),
    )

    # Minutes to/from the next/previous fixed 8h UTC funding boundary.
    minute_of_day_funding = [h * 60 for h in FUNDING_HOURS_UTC] + [24 * 60]
    next_boundary = pl.lit(minute_of_day_funding[-1])
    for b in reversed(minute_of_day_funding[:-1]):
        next_boundary = pl.when(minute_of_day < b).then(pl.lit(b)).otherwise(next_boundary)
    prev_boundary = pl.lit(minute_of_day_funding[0])
    for b in minute_of_day_funding[:-1]:
        prev_boundary = pl.when(minute_of_day >= b).then(pl.lit(b)).otherwise(prev_boundary)

    df = df.with_columns(
        ((next_boundary - minute_of_day) % (24 * 60)).alias("minutes_to_funding"),
        (minute_of_day - prev_boundary).alias("minutes_from_funding"),
    )
    df = df.with_columns(
        (pl.col("minutes_to_funding") <= 60).cast(pl.Int8).alias("pre_funding_1h"),
        (pl.col("minutes_from_funding") <= 60).cast(pl.Int8).alias("post_funding_1h"),
        (pl.col("minutes_from_funding") / (8 * 60)).alias("funding_cycle_phase"),
    )

    df = df.with_columns(
        (dom >= 28).cast(pl.Int8).alias("is_month_end_window"),
        (month.is_in([3, 6, 9, 12]) & (dom >= 28)).cast(pl.Int8).alias("is_quarter_end_window"),
    )
    return df
