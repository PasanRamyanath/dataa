"""
Parsers for Binance Vision zip exports (futures um/cm + spot).

Every reader:
  * detects and skips zero-byte files (known-listing-day / not-yet-published
    gaps — never treated as corruption, but never silently fed downstream
    either),
  * verifies the sibling .CHECKSUM file when present,
  * branches header-presence and timestamp-unit by MARKET (futures has a
    header + ms epochs; spot has no header + us epochs) rather than by
    datatype name,
  * special-cases the one known non-standard Spark export
    (futures/um/monthly/aggTrades/BTCUSDT/part-00000-*.zip),
  * normalizes every timestamp column to int64 nanoseconds in a `ts` column
    so every downstream feature module works in one unit regardless of source.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl

import paths as _paths
import schemas as sch
from utils import normalize_epoch_column_to_ns, verify_checksum, is_zero_byte

MARKET_UM = "um"
MARKET_CM = "cm"
MARKET_SPOT = "spot"


def _read_single_csv_from_zip(zip_path: Path, has_header: bool) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV member found inside {zip_path}")
        return zf.read(names[0])


def _load_zip_checked(zip_path: Path) -> bytes | None:
    if not zip_path.exists():
        return None
    if is_zero_byte(zip_path):
        return None
    ok = verify_checksum(zip_path)
    if ok is False:
        raise ValueError(f"Checksum mismatch for {zip_path}")
    return _read_single_csv_from_zip(zip_path, has_header=True)


def _is_spark_artifact(zip_path: Path) -> bool:
    return zip_path.name.startswith("part-")


def _has_header(raw: bytes) -> bool:
    """Sniffs the first line rather than trusting the market: spot never has
    a header, and older futures dumps (roughly pre-2022) were published
    without one too. Only valid for files whose first column is a numeric
    epoch/id — a numeric first field means the first line is already data."""
    first_field = raw.split(b"\n", 1)[0].split(b",", 1)[0].strip().strip(b'"')
    try:
        float(first_field)
    except ValueError:
        return True
    return False


def _read_csv(raw: bytes) -> pl.DataFrame:
    return pl.read_csv(io.BytesIO(raw), has_header=_has_header(raw), infer_schema_length=10_000)


def read_agg_trades(zip_path: Path, market: str) -> pl.DataFrame | None:
    raw = _load_zip_checked(zip_path)
    if raw is None:
        return None

    if market == MARKET_SPOT:
        columns, dtypes = sch.AGGTRADES_COLUMNS_SPOT, sch.AGGTRADES_DTYPES_SPOT
    else:
        columns, dtypes = sch.AGGTRADES_COLUMNS_FUTURES, sch.AGGTRADES_DTYPES_FUTURES

    if market != MARKET_SPOT and _is_spark_artifact(zip_path):
        # Non-standard export: 8 cols (7 standard + trailing `formatted_dt`
        # string like "2023-11"), still with a header row. Drop the extra col.
        df = pl.read_csv(io.BytesIO(raw), infer_schema_length=10_000)
        if sch.AGGTRADES_SPARK_EXTRA_COLUMN in df.columns:
            df = df.drop(sch.AGGTRADES_SPARK_EXTRA_COLUMN)
        df.columns = columns
        df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in dtypes.items()])
    else:
        df = _read_csv(raw)
        df.columns = columns
        df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in dtypes.items()])

    df = normalize_epoch_column_to_ns(df, "transact_time", "ts")
    return df


def read_trades(zip_path: Path, market: str) -> pl.DataFrame | None:
    raw = _load_zip_checked(zip_path)
    if raw is None:
        return None

    if market == MARKET_SPOT:
        columns, dtypes = sch.TRADES_COLUMNS_SPOT, sch.TRADES_DTYPES_SPOT
    elif market == MARKET_CM:
        columns, dtypes = sch.TRADES_COLUMNS_CM, sch.TRADES_DTYPES_CM
    else:
        columns, dtypes = sch.TRADES_COLUMNS_UM, sch.TRADES_DTYPES_UM

    df = _read_csv(raw)
    df.columns = columns
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in dtypes.items()])
    df = normalize_epoch_column_to_ns(df, "time", "ts")
    return df


def read_klines(zip_path: Path, market: str) -> pl.DataFrame | None:
    """Shared reader for klines / indexPriceKlines / markPriceKlines /
    premiumIndexKlines — all four share the identical 12-column layout."""
    raw = _load_zip_checked(zip_path)
    if raw is None:
        return None
    df = _read_csv(raw)
    df.columns = sch.KLINE_COLUMNS
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in sch.KLINE_DTYPES.items()])
    # Point-in-time key: a kline's OHLC is only known once the bar CLOSES.
    # Keying on open_time let every asof-backward join see the minute's close
    # up to 60s early (a look-ahead leak into basis/mark/premium/term-structure
    # features). ts = close_time rounded up to the next ms (close_time is
    # open+interval-1ms; ms or us epochs both land on open+interval).
    df = normalize_epoch_column_to_ns(df, "open_time", "open_ts")
    df = normalize_epoch_column_to_ns(df, "close_time", "close_ts")
    df = df.with_columns(((pl.col("close_ts") // 1_000_000 + 1) * 1_000_000).alias("ts"))
    return df.sort("ts")


def read_book_ticker(zip_path: Path) -> pl.DataFrame | None:
    """KNOWN STALE for both um (2024-03-30) and cm (2024-10-14) samples —
    only one file exists per directory. Callers should treat bookTicker as
    unavailable in production and prefer bookDepth for spread/microprice."""
    raw = _load_zip_checked(zip_path)
    if raw is None:
        return None
    df = _read_csv(raw)
    df.columns = sch.BOOKTICKER_COLUMNS
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in sch.BOOKTICKER_DTYPES.items()])
    df = normalize_epoch_column_to_ns(df, "transaction_time", "ts")
    return df


def read_book_depth(zip_path: Path) -> pl.DataFrame | None:
    raw = _load_zip_checked(zip_path)
    if raw is None:
        return None
    df = pl.read_csv(io.BytesIO(raw), has_header=True, infer_schema_length=10_000)
    df.columns = sch.BOOKDEPTH_COLUMNS
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in sch.BOOKDEPTH_DTYPES.items()])
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime(time_unit="ns"), "%Y-%m-%d %H:%M:%S", strict=False).alias("ts")
    )
    return df


METRICS_PERIOD_MINUTES = 5  # Binance Vision futures metrics granularity


def read_metrics(zip_path: Path) -> pl.DataFrame | None:
    raw = _load_zip_checked(zip_path)
    if raw is None:
        return None
    df = pl.read_csv(io.BytesIO(raw), has_header=True, infer_schema_length=10_000)
    df.columns = sch.METRICS_COLUMNS
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in sch.METRICS_DTYPES.items() if c != "create_time"])
    # create_time is the START of the 5-minute stats window: the row stamped
    # T carries the taker buy/sell ratio of [T, T+5m) (matches trade-derived
    # flow at r=0.999, vs 0.2 for [T-5m, T)) and OI valued at the T+5m price.
    # Key it at window end so asof-backward joins can't see it 5 minutes early.
    df = df.with_columns(
        (pl.col("create_time").str.strptime(pl.Datetime(time_unit="ns"), "%Y-%m-%d %H:%M:%S", strict=False)
         + pl.duration(minutes=METRICS_PERIOD_MINUTES)).cast(pl.Datetime(time_unit="ns")).alias("ts")
    )
    return df.sort("ts")


def read_funding_rate(zip_path: Path) -> pl.DataFrame | None:
    raw = _load_zip_checked(zip_path)
    if raw is None:
        return None
    df = _read_csv(raw)
    df.columns = sch.FUNDINGRATE_COLUMNS
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in sch.FUNDINGRATE_DTYPES.items()])
    df = normalize_epoch_column_to_ns(df, "calc_time", "ts")
    return df


def read_funding_rate_for_day(market: str, symbol: str, day, max_lookback_months: int = 6) -> pl.DataFrame | None:
    """Binance Vision publishes monthly fundingRate dumps with a ~1-month
    publication lag, so the *current* month's file is routinely absent for
    any recent/live day (confirmed in samples: BTCUSDT's latest fundingRate
    file is 2026-08 while sample market data runs into 2026-09). Walk
    backward month-by-month until a published file is found, so `day` still
    gets the last known funding rate to forward-fill instead of going
    entirely null just because this month hasn't been published yet."""
    year, month = day.year, day.month
    for _ in range(max_lookback_months + 1):
        df = read_funding_rate(_paths.funding_rate_path(market, symbol, year, month))
        if df is not None:
            return df
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return None


def read_liquidation_snapshot(zip_path: Path) -> pl.DataFrame | None:
    """cm only. Vendor exports are known to contain exact duplicate rows
    (confirmed 50% duplication in the 2024-10-14 sample) — always dedup."""
    raw = _load_zip_checked(zip_path)
    if raw is None:
        return None
    df = _read_csv(raw)
    df.columns = sch.LIQUIDATIONSNAPSHOT_COLUMNS
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in sch.LIQUIDATIONSNAPSHOT_DTYPES.items()])
    df = df.unique(keep="first")
    df = normalize_epoch_column_to_ns(df, "time", "ts")
    return df.sort("ts")


def read_bvol_index(zip_path: Path) -> pl.DataFrame | None:
    """Binance's BTC implied-volatility index (option/daily/BVOLIndex/) —
    sub-second-cadence header CSV, `index_value` is an annualized IV
    percentage."""
    raw = _load_zip_checked(zip_path)
    if raw is None:
        return None
    df = _read_csv(raw)
    df.columns = sch.BVOL_INDEX_COLUMNS
    df = df.with_columns([pl.col(c).cast(t, strict=False) for c, t in sch.BVOL_INDEX_DTYPES.items()])
    df = normalize_epoch_column_to_ns(df, "calc_time", "ts")
    return df.sort("ts")
