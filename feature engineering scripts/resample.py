"""
Resample raw tick/trade streams into fixed-width OHLCV bars at multiple
resolutions (1s / 10s / 1m / 5m), using left-closed, left-labeled windows so
that a bar labeled `ts` only ever contains trades with `ts <= trade_ts <
ts + interval` — i.e. strictly backward-looking, no centered windows.

All bar builders expect a `ts` column already normalized to int64 nanoseconds
(see utils.normalize_epoch_column_to_ns) and a `Datetime` cast is applied
internally since `group_by_dynamic` requires a temporal dtype.
"""
from __future__ import annotations

import polars as pl

from utils import safe_div

INTERVAL_TO_POLARS = {"1s": "1s", "10s": "10s", "1m": "1m", "5m": "5m"}


def _as_datetime(df: pl.DataFrame, ts_col: str = "ts") -> pl.DataFrame:
    if df.schema[ts_col] == pl.Datetime:
        return df
    return df.with_columns(pl.from_epoch(pl.col(ts_col), time_unit="ns").alias(ts_col))


def build_trade_bars(trades: pl.DataFrame, interval: str, ts_col: str = "ts",
                      price_col: str = "price", qty_col: str = "quantity",
                      is_buyer_maker_col: str = "is_buyer_maker") -> pl.DataFrame:
    """OHLCV + trade-count + buy/sell volume split + VWAP from a raw trade
    stream (aggTrades or trades schema)."""
    df = _as_datetime(trades, ts_col).sort(ts_col)
    pl_interval = INTERVAL_TO_POLARS[interval]

    buy_qty = pl.when(~pl.col(is_buyer_maker_col)).then(pl.col(qty_col)).otherwise(0.0)
    sell_qty = pl.when(pl.col(is_buyer_maker_col)).then(pl.col(qty_col)).otherwise(0.0)
    notional = pl.col(price_col) * pl.col(qty_col)

    bars = df.group_by_dynamic(ts_col, every=pl_interval, closed="left", label="left").agg([
        pl.col(price_col).first().alias("open"),
        pl.col(price_col).max().alias("high"),
        pl.col(price_col).min().alias("low"),
        pl.col(price_col).last().alias("close"),
        pl.col(qty_col).sum().alias("volume"),
        pl.len().alias("trade_count"),
        buy_qty.sum().alias("buy_volume"),
        sell_qty.sum().alias("sell_volume"),
        notional.sum().alias("notional_volume"),
        pl.col(price_col).std().alias("intra_bar_price_std"),
        notional.max().alias("max_trade_notional"),
        pl.col(qty_col).std().alias("trade_size_std"),
    ])
    bars = bars.with_columns([
        safe_div(pl.col("notional_volume"), pl.col("volume")).alias("vwap"),
        (pl.col("buy_volume") - pl.col("sell_volume")).alias("volume_delta"),
    ])
    return bars


def forward_fill_to_grid(df: pl.DataFrame, start_ts, end_ts, interval: str,
                          ts_col: str = "ts", ffill_cols: list[str] | None = None) -> pl.DataFrame:
    """Reindex a lower-frequency source (e.g. funding rate, metrics) onto a
    uniform grid at `interval` and forward-fill — the ONLY direction allowed,
    since backward-fill would leak future values into earlier bars."""
    pl_interval = INTERVAL_TO_POLARS[interval]
    grid = pl.DataFrame({
        ts_col: pl.datetime_range(start=start_ts, end=end_ts, interval=pl_interval, eager=True, time_unit="ns")
    })
    df = _as_datetime(df, ts_col).sort(ts_col)
    out = grid.join_asof(df, on=ts_col, strategy="backward")
    if ffill_cols:
        out = out.with_columns([pl.col(c).forward_fill() for c in ffill_cols])
    return out


def asof_join_backward(left: pl.DataFrame, right: pl.DataFrame, ts_col: str = "ts",
                        by: list[str] | None = None) -> pl.DataFrame:
    """Join a lower-frequency right-hand source onto `left`'s bar grid using
    strategy='backward' (the only leakage-safe direction: each left row only
    ever sees the most recent right-hand value at or before its own ts)."""
    left = _as_datetime(left, ts_col).sort(ts_col)
    right = _as_datetime(right, ts_col).sort(ts_col)
    return left.join_asof(right, on=ts_col, by=by, strategy="backward")
