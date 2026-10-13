"""
Parsers for Bybit public data (public.bybit.com): raw tick trades under
`trading/`, and 1-minute premium/spot index bars under `premium_index/` and
`spot_index/`.

Key fact (confirmed by direct sample audit): `trading/` files branch schema
by CONTRACT TYPE, not by date. Perpetual symbols (e.g. BTCUSDT) carry a
trailing `rpi` (retail price improvement) flag column; dated/quarterly
futures (e.g. BTCUSDH24) do not. Branch on column count after reading the
header, never on the file's date.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

import schemas as sch
from utils import normalize_epoch_column_to_ns


def _read_gzip_csv(path: Path) -> pl.DataFrame:
    with gzip.open(path, "rb") as f:
        return pl.read_csv(f, infer_schema_length=10_000)


def read_bybit_trading(path: Path) -> pl.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = _read_gzip_csv(path)
    n_cols = df.width
    if n_cols == len(sch.BYBIT_TRADING_COLUMNS_PERP):
        columns, dtypes = sch.BYBIT_TRADING_COLUMNS_PERP, sch.BYBIT_TRADING_DTYPES_PERP
    elif n_cols == len(sch.BYBIT_TRADING_COLUMNS_DATED):
        columns, dtypes = sch.BYBIT_TRADING_COLUMNS_DATED, sch.BYBIT_TRADING_DTYPES_DATED
    else:
        raise ValueError(
            f"Unexpected Bybit trading column count ({n_cols}) in {path}; "
            f"expected {len(sch.BYBIT_TRADING_COLUMNS_DATED)} (dated) or "
            f"{len(sch.BYBIT_TRADING_COLUMNS_PERP)} (perpetual)."
        )
    df.columns = columns
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in dtypes.items()])
    # `timestamp` is unix seconds as a float with fractional microseconds.
    df = df.with_columns((pl.col("timestamp") * 1_000_000_000).cast(pl.Int64).alias("ts"))
    # Bybit's `side` names the TAKER's side directly (side="Sell" means an
    # aggressive sell hit the bid). To match Binance's `is_buyer_maker`
    # convention (True == the buyer was passive/maker, i.e. taker sold) so
    # this feeds resample.build_trade_bars() unmodified, is_buyer_maker must
    # be True exactly when side=="Sell" — NOT when side=="Buy".
    df = df.with_columns((pl.col("side").str.to_lowercase() == "sell").alias("is_buyer_maker"))
    return df.sort("ts")


def read_bybit_index(path: Path) -> pl.DataFrame | None:
    """Shared reader for premium_index/ and spot_index/ 1-minute bars."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = _read_gzip_csv(path)
    df.columns = sch.BYBIT_INDEX_COLUMNS
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in sch.BYBIT_INDEX_DTYPES.items()])
    df = normalize_epoch_column_to_ns(df, "start_at", "ts")
    df = df.with_columns(pl.col("ts") + pl.col("period").cast(pl.Int64) * 60 * 1_000_000_000)
    return df.sort("ts")
