"""
Verified raw-column schemas for every data source in this pipeline.

Every schema here was confirmed by actually decompressing and inspecting
sample files (see project research notes), not guessed from vendor docs.
Do NOT "helpfully" reorder or rename these — column order and exact names
are load-bearing for the CSV readers in parsers/.

Known cross-cutting facts (see parsers/binance_parser.py and
parsers/bybit_parser.py for how these are handled):
  * Binance futures (um & cm) CSVs have a header row and millisecond epochs.
  * Binance spot CSVs have NO header row and MICROSECOND epochs (16-digit).
  * cm `trades` differs from um `trades` only in column 4: base_qty (BTC)
    instead of quote_qty (USDT) — same position, different unit.
  * cm `bookDepth`'s `depth`/`notional` semantics are reversed vs um's
    (coin-margined convention): treat as opaque numeric columns and let the
    feature layer decide what each side means per market.
  * cm `indexPriceKlines` is keyed by the underlying pair (BTCUSD), not the
    contract symbol (BTCUSD_PERP).
  * Bybit `trading/` has two distinct schemas coexisting by CONTRACT TYPE,
    not by date: perpetual symbols carry a trailing `rpi` flag (11 cols),
    dated/quarterly futures do not (10 cols).
"""
from __future__ import annotations

import polars as pl

# --------------------------------------------------------------------------
# Binance Vision — trades / aggTrades
# --------------------------------------------------------------------------
AGGTRADES_COLUMNS_FUTURES = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]
AGGTRADES_COLUMNS_SPOT = AGGTRADES_COLUMNS_FUTURES + ["is_best_match"]

AGGTRADES_DTYPES_FUTURES = {
    "agg_trade_id": pl.Int64, "price": pl.Float64, "quantity": pl.Float64,
    "first_trade_id": pl.Int64, "last_trade_id": pl.Int64,
    "transact_time": pl.Int64, "is_buyer_maker": pl.Boolean,
}
AGGTRADES_DTYPES_SPOT = {**AGGTRADES_DTYPES_FUTURES, "is_best_match": pl.Boolean}

# Non-standard Spark-job export found under
# futures/um/monthly/aggTrades/BTCUSDT/part-00000-*.zip — 8 cols, extra
# trailing `formatted_dt` (e.g. "2023-11"). Detected by filename prefix
# "part-" and/or column-count mismatch; the extra column is dropped.
AGGTRADES_SPARK_EXTRA_COLUMN = "formatted_dt"

TRADES_COLUMNS_UM = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]
TRADES_COLUMNS_CM = ["id", "price", "qty", "base_qty", "time", "is_buyer_maker"]
TRADES_COLUMNS_SPOT = TRADES_COLUMNS_UM + ["is_best_match"]

TRADES_DTYPES_UM = {
    "id": pl.Int64, "price": pl.Float64, "qty": pl.Float64, "quote_qty": pl.Float64,
    "time": pl.Int64, "is_buyer_maker": pl.Boolean,
}
TRADES_DTYPES_CM = {
    "id": pl.Int64, "price": pl.Float64, "qty": pl.Float64, "base_qty": pl.Float64,
    "time": pl.Int64, "is_buyer_maker": pl.Boolean,
}
TRADES_DTYPES_SPOT = {**TRADES_DTYPES_UM, "is_best_match": pl.Boolean}

# --------------------------------------------------------------------------
# Binance Vision — kline family (klines / indexPriceKlines / markPriceKlines /
# premiumIndexKlines all share this exact 12-column layout on um & cm; spot
# klines share it too but with microsecond timestamps and no header).
# --------------------------------------------------------------------------
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
KLINE_DTYPES = {
    "open_time": pl.Int64, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
    "close": pl.Float64, "volume": pl.Float64, "close_time": pl.Int64,
    "quote_volume": pl.Float64, "count": pl.Int64, "taker_buy_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64, "ignore": pl.Int64,
}

# --------------------------------------------------------------------------
# Binance Vision — bookTicker (KNOWN STALE: um sample dated 2024-03-30,
# cm sample dated 2024-10-14; only one file exists in each directory —
# treat as unavailable in production and fall back to bookDepth).
# --------------------------------------------------------------------------
BOOKTICKER_COLUMNS = [
    "update_id", "best_bid_price", "best_bid_qty", "best_ask_price",
    "best_ask_qty", "transaction_time", "event_time",
]
BOOKTICKER_DTYPES = {
    "update_id": pl.Int64, "best_bid_price": pl.Float64, "best_bid_qty": pl.Float64,
    "best_ask_price": pl.Float64, "best_ask_qty": pl.Float64,
    "transaction_time": pl.Int64, "event_time": pl.Int64,
}

# --------------------------------------------------------------------------
# Binance Vision — bookDepth (um/cm only; timestamp is a human-readable
# string, not epoch)
# --------------------------------------------------------------------------
BOOKDEPTH_COLUMNS = ["timestamp", "percentage", "depth", "notional"]
BOOKDEPTH_DTYPES = {
    "timestamp": pl.Utf8, "percentage": pl.Float64, "depth": pl.Float64, "notional": pl.Float64,
}

# --------------------------------------------------------------------------
# Binance Vision — open-interest / long-short-ratio metrics (um/cm)
# --------------------------------------------------------------------------
METRICS_COLUMNS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]
METRICS_DTYPES = {
    "create_time": pl.Utf8, "symbol": pl.Utf8,
    "sum_open_interest": pl.Float64, "sum_open_interest_value": pl.Float64,
    "count_toptrader_long_short_ratio": pl.Float64, "sum_toptrader_long_short_ratio": pl.Float64,
    "count_long_short_ratio": pl.Float64, "sum_taker_long_short_vol_ratio": pl.Float64,
}

# --------------------------------------------------------------------------
# Binance Vision — fundingRate (monthly only, um & cm)
# --------------------------------------------------------------------------
FUNDINGRATE_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
FUNDINGRATE_DTYPES = {
    "calc_time": pl.Int64, "funding_interval_hours": pl.Int64, "last_funding_rate": pl.Float64,
}

# --------------------------------------------------------------------------
# Binance Vision — liquidationSnapshot (cm only; KNOWN to contain exact
# duplicate rows in real vendor exports — always dedup after load)
# --------------------------------------------------------------------------
LIQUIDATIONSNAPSHOT_COLUMNS = [
    "time", "side", "order_type", "time_in_force", "original_quantity",
    "price", "average_price", "order_status", "last_fill_quantity",
    "accumulated_fill_quantity",
]
LIQUIDATIONSNAPSHOT_DTYPES = {
    "time": pl.Int64, "side": pl.Utf8, "order_type": pl.Utf8, "time_in_force": pl.Utf8,
    "original_quantity": pl.Float64, "price": pl.Float64, "average_price": pl.Float64,
    "order_status": pl.Utf8, "last_fill_quantity": pl.Float64,
    "accumulated_fill_quantity": pl.Float64,
}

# --------------------------------------------------------------------------
# Bybit — trading/ raw tick trades. Two schemas by CONTRACT TYPE.
# --------------------------------------------------------------------------
BYBIT_TRADING_COLUMNS_PERP = [
    "timestamp", "symbol", "side", "size", "price", "tickDirection",
    "trdMatchID", "grossValue", "homeNotional", "foreignNotional", "rpi",
]
BYBIT_TRADING_COLUMNS_DATED = BYBIT_TRADING_COLUMNS_PERP[:-1]  # no trailing rpi

BYBIT_TRADING_DTYPES_COMMON = {
    "timestamp": pl.Float64,  # unix seconds with fractional microseconds, e.g. 1712345678.123456
    "symbol": pl.Utf8, "side": pl.Utf8, "size": pl.Float64, "price": pl.Float64,
    "tickDirection": pl.Utf8, "trdMatchID": pl.Utf8, "grossValue": pl.Float64,
    "homeNotional": pl.Float64, "foreignNotional": pl.Float64,
}
BYBIT_TRADING_DTYPES_PERP = {**BYBIT_TRADING_DTYPES_COMMON, "rpi": pl.Utf8}
BYBIT_TRADING_DTYPES_DATED = dict(BYBIT_TRADING_DTYPES_COMMON)

# --------------------------------------------------------------------------
# Bybit — premium_index / spot_index (1-minute bars, header present, Unix
# seconds int).
# --------------------------------------------------------------------------
BYBIT_INDEX_COLUMNS = ["start_at", "symbol", "period", "open", "high", "low", "close"]
BYBIT_INDEX_DTYPES = {
    "start_at": pl.Int64, "symbol": pl.Utf8, "period": pl.Utf8,
    "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
}

# --------------------------------------------------------------------------
# Binance Vision — option/daily/BVOLIndex (BTC implied-volatility index,
# confirmed via direct sample inspection: header present, `calc_time` is a
# 13-digit millisecond epoch, sub-second sampling cadence, `index_value` is
# an annualized IV percentage e.g. 40.7851 == 40.79% IV).
# --------------------------------------------------------------------------
BVOL_INDEX_COLUMNS = ["calc_time", "symbol", "base_asset", "quote_asset", "index_value"]
BVOL_INDEX_DTYPES = {
    "calc_time": pl.Int64, "symbol": pl.Utf8, "base_asset": pl.Utf8,
    "quote_asset": pl.Utf8, "index_value": pl.Float64,
}


def detect_timestamp_unit(sample_value: int) -> str:
    """Classify an integer epoch timestamp by its digit count.

    10 digits -> seconds, 13 -> milliseconds, 16 -> microseconds,
    19 -> nanoseconds. Never hardcode a unit per-datatype: Binance spot
    switched from ms to us epochs mid-history, so the same datatype can
    carry either unit depending on the file's date.
    """
    digits = len(str(int(abs(sample_value))))
    if digits <= 10:
        return "s"
    if digits <= 13:
        return "ms"
    if digits <= 16:
        return "us"
    return "ns"


_UNIT_TO_NS_MULTIPLIER = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


def timestamp_unit_multiplier_to_ns(unit: str) -> int:
    return _UNIT_TO_NS_MULTIPLIER[unit]
