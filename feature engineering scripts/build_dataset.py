"""
Main entrypoint: for each day in a date range, load every available Binance
Vision (um/cm/spot) and Bybit source, build the primary 10s bar grid (plus
1m/5m context bars) from USDT-margined perpetual (BTCUSDT) trades, run the
full feature-engineering pipeline, attach labels, run a quick validation
pass, write a local parquet, and optionally upload it to the configured HF
dataset repo.

Usage:
    python build_dataset.py --start 2026-09-01 --end 2026-09-06 --upload
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

import config
import contracts
import paths
from common.feature_contract import COLUMN_DTYPES, conform_to_contract
from feature_engine.cross_exchange import build_bybit_bars
from feature_engine.pipeline import DaySources, compute_all_features, feature_manifest
from labeling import add_all_labels
from parsers.binance_parser import (
    MARKET_SPOT,
    MARKET_UM,
    read_agg_trades,
    read_book_depth,
    read_book_ticker,
    read_bvol_index,
    read_funding_rate_for_day,
    read_klines,
    read_liquidation_snapshot,
    read_metrics,
)
from parsers.bybit_parser import read_bybit_index, read_bybit_trading
from resample import build_trade_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_dataset")


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def build_day(day: date, conform: bool = True) -> pl.DataFrame | None:
    um_trades = read_agg_trades(paths.agg_trades_path(MARKET_UM, config.SYMBOL_UM, day), MARKET_UM)
    if um_trades is None or um_trades.is_empty():
        log.warning("No UM aggTrades for %s — skipping day", day)
        return None

    primary_bars = build_trade_bars(um_trades, config.PRIMARY_INTERVAL)
    bars_1m = build_trade_bars(um_trades, "1m")
    bars_5m = build_trade_bars(um_trades, "5m")

    book_depth = read_book_depth(paths.book_depth_path(MARKET_UM, config.SYMBOL_UM, day))
    book_ticker = read_book_ticker(paths.book_ticker_path(MARKET_UM, config.SYMBOL_UM, day))
    metrics = read_metrics(paths.metrics_path(MARKET_UM, config.SYMBOL_UM, day))
    funding_rate = read_funding_rate_for_day(MARKET_UM, config.SYMBOL_UM, day)

    index_klines = read_klines(paths.index_price_klines_path(MARKET_UM, config.SYMBOL_UM, "1m", day), MARKET_UM)
    mark_klines = read_klines(paths.mark_price_klines_path(MARKET_UM, config.SYMBOL_UM, "1m", day), MARKET_UM)
    premium_klines = read_klines(paths.premium_index_klines_path(MARKET_UM, config.SYMBOL_UM, "1m", day), MARKET_UM)

    spot_trades = read_agg_trades(paths.agg_trades_path(MARKET_SPOT, config.SYMBOL_SPOT, day), MARKET_SPOT)
    spot_bars = build_trade_bars(spot_trades, config.PRIMARY_INTERVAL) if spot_trades is not None else None

    liquidations = read_liquidation_snapshot(paths.liquidation_snapshot_path(config.SYMBOL_CM, day))

    bybit_trades = read_bybit_trading(paths.bybit_trading_path(config.BYBIT_SYMBOL, day))
    bybit_bars = build_bybit_bars(bybit_trades, config.PRIMARY_INTERVAL) if bybit_trades is not None else None

    bvol_index = read_bvol_index(paths.bvol_index_path(config.BVOL_SYMBOL, day))
    bybit_premium_index = read_bybit_index(paths.bybit_premium_index_path(config.BYBIT_INDEX_SYMBOL, day))
    bybit_spot_index = read_bybit_index(paths.bybit_spot_index_path(config.BYBIT_INDEX_SYMBOL, day))

    front_contract, back_contract = contracts.front_and_back_quarterly(config.QUARTERLY_BASE_UM, day)
    front_quarter_klines = read_klines(
        paths.mark_price_klines_path(MARKET_UM, front_contract.symbol, "1m", day), MARKET_UM
    )
    back_quarter_klines = read_klines(
        paths.mark_price_klines_path(MARKET_UM, back_contract.symbol, "1m", day), MARKET_UM
    )

    sources = DaySources(
        futures_bars=primary_bars,
        book_depth=book_depth,
        book_ticker=book_ticker,
        funding_rate=funding_rate,
        metrics=metrics,
        index_klines=index_klines,
        mark_klines=mark_klines,
        premium_klines=premium_klines,
        spot_bars=spot_bars,
        liquidations=liquidations,
        bars_1m=bars_1m,
        bars_5m=bars_5m,
        bybit_bars=bybit_bars,
        bvol_index=bvol_index,
        bybit_premium_index=bybit_premium_index,
        bybit_spot_index=bybit_spot_index,
        front_quarter_klines=front_quarter_klines,
        back_quarter_klines=back_quarter_klines,
    )

    featured = compute_all_features(sources)
    labeled = add_all_labels(featured)
    if not conform:
        return labeled
    extra = sorted(set(labeled.columns) - set(COLUMN_DTYPES))
    if extra:
        log.info("Day %s: dropping %d column(s) outside the feature contract: %s", day, len(extra), extra)
    return conform_to_contract(labeled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--upload", action="store_true", help="Upload each day's parquet to the HF dataset repo")
    parser.add_argument("--out-dir", default=str(config.OUTPUT_ROOT))
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for day in _daterange(start, end):
        log.info("Processing %s", day)
        try:
            df = build_day(day)
        except Exception:
            log.exception("Failed to build features for %s", day)
            continue
        if df is None:
            continue

        manifest = feature_manifest(df)
        log.info("Day %s: %d rows, %d columns", day, manifest["n_rows"], manifest["n_columns"])

        out_path = out_dir / f"features_{config.SYMBOL_UM}_{day.isoformat()}.parquet"
        df.write_parquet(out_path, compression="snappy")

        if args.upload:
            from publish_to_hf import upload_daily_features
            path_in_repo = upload_daily_features(df, config.SYMBOL_UM, day.isoformat())
            log.info("Uploaded to %s", path_in_repo)


if __name__ == "__main__":
    main()
