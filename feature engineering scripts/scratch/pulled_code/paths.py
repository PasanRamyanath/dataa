"""
Resolves on-disk paths for Binance Vision and Bybit files, matching the
exact vendor layout observed under `data/`, `premium_index/`, `spot_index/`,
`trading/` (and preserved verbatim by binance_btc_usd_downloader.py):

  data/<market>/<scope>/<dataType>/<symbol>/[interval/]<symbol>-<dataType-or-interval>-<date>.zip

`klines` and its derived-price siblings (indexPriceKlines/markPriceKlines/
premiumIndexKlines) use the INTERVAL in the filename, not the datatype name;
every other datatype uses the datatype name itself.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from config import DATA_ROOT, PREMIUM_INDEX_ROOT, SPOT_INDEX_ROOT, TRADING_ROOT

KLINE_FAMILY_DATATYPES = {"klines", "indexPriceKlines", "markPriceKlines", "premiumIndexKlines"}


def _daily_path(market: str, datatype: str, symbol: str, day: date, interval: str | None = None) -> Path:
    date_str = day.isoformat()
    base = DATA_ROOT / market / "daily" / datatype / symbol
    if interval is not None:
        base = base / interval
    name_key = interval if datatype in KLINE_FAMILY_DATATYPES else datatype
    return base / f"{symbol}-{name_key}-{date_str}.zip"


def _monthly_path(market: str, datatype: str, symbol: str, year: int, month: int, interval: str | None = None) -> Path:
    month_str = f"{year:04d}-{month:02d}"
    base = DATA_ROOT / market / "monthly" / datatype / symbol
    if interval is not None:
        base = base / interval
    name_key = interval if datatype in KLINE_FAMILY_DATATYPES else datatype
    return base / f"{symbol}-{name_key}-{month_str}.zip"


def agg_trades_path(market: str, symbol: str, day: date) -> Path:
    return _daily_path(f"futures/{market}" if market in ("um", "cm") else market, "aggTrades", symbol, day)


def trades_path(market: str, symbol: str, day: date) -> Path:
    return _daily_path(f"futures/{market}" if market in ("um", "cm") else market, "trades", symbol, day)


def klines_path(market: str, symbol: str, interval: str, day: date) -> Path:
    return _daily_path(f"futures/{market}" if market in ("um", "cm") else market, "klines", symbol, day, interval)


def index_price_klines_path(market: str, symbol: str, interval: str, day: date) -> Path:
    return _daily_path(f"futures/{market}", "indexPriceKlines", symbol, day, interval)


def mark_price_klines_path(market: str, symbol: str, interval: str, day: date) -> Path:
    return _daily_path(f"futures/{market}", "markPriceKlines", symbol, day, interval)


def premium_index_klines_path(market: str, symbol: str, interval: str, day: date) -> Path:
    return _daily_path(f"futures/{market}", "premiumIndexKlines", symbol, day, interval)


def book_ticker_path(market: str, symbol: str, day: date) -> Path:
    return _daily_path(f"futures/{market}", "bookTicker", symbol, day)


def book_depth_path(market: str, symbol: str, day: date) -> Path:
    return _daily_path(f"futures/{market}", "bookDepth", symbol, day)


def metrics_path(market: str, symbol: str, day: date) -> Path:
    return _daily_path(f"futures/{market}", "metrics", symbol, day)


def funding_rate_path(market: str, symbol: str, year: int, month: int) -> Path:
    return _monthly_path(f"futures/{market}", "fundingRate", symbol, year, month)


def liquidation_snapshot_path(symbol: str, day: date) -> Path:
    return _daily_path("futures/cm", "liquidationSnapshot", symbol, day)


def bvol_index_path(symbol: str, day: date) -> Path:
    """`data/option/daily/BVOLIndex/<symbol>/<symbol>-BVOLIndex-<date>.zip`
    — Binance's BTC implied-volatility index, published under the `option`
    market segment (sibling to `futures/um`, `futures/cm`, `spot`), keyed by
    its own symbol (e.g. BTCBVOLUSDT), not the underlying perpetual."""
    return _daily_path("option", "BVOLIndex", symbol, day)


def bybit_trading_path(symbol: str, day: date) -> Path:
    """`trading/<symbol>/<symbol><date>.csv.gz`. `symbol` here is Bybit's own
    contract identifier — a perpetual (BTCUSDT, BTCUSD, BTCPERP) or a dated/
    quarterly contract folder (e.g. "BTC-01DEC23", "BTCUSDH22"). One file
    exists per UTC day inside each contract's own folder."""
    return TRADING_ROOT / symbol / f"{symbol}{day.isoformat()}.csv.gz"


def bybit_premium_index_path(symbol: str, day: date) -> Path:
    return PREMIUM_INDEX_ROOT / symbol / f"{symbol}{day.isoformat()}_premium_index.csv.gz"


def bybit_spot_index_path(symbol: str, day: date) -> Path:
    return SPOT_INDEX_ROOT / symbol / f"{symbol}{day.isoformat()}_index_price.csv.gz"
