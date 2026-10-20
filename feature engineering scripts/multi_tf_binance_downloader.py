"""
feature engineering scripts/multi_tf_binance_downloader.py
High-throughput parallel downloader for https://data.binance.vision across all 13 native
kline intervals (1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d) plus
markPriceKlines, indexPriceKlines, premiumIndexKlines, metrics, fundingRate,
bookDepth, aggTrades, and BVOLIndex across BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT.

Supports both 'monthly' bulk archives (for 2019-2026 macro/meso training) and
'daily' archives (for recent days and high-frequency micro/tactical data), with
SHA-256 .CHECKSUM verification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import hashlib
import logging
from pathlib import Path
from typing import Iterable
import zipfile

import requests
from requests.adapters import HTTPAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multi_tf_downloader")

DOWNLOAD_BASE_URL = "https://data.binance.vision"

ALL_KLINE_INTERVALS_UM = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h", "1d",
]
ALL_KLINE_INTERVALS_SPOT = ["1s"] + ALL_KLINE_INTERVALS_UM

UM_KLINE_TYPES = [
    "klines",
    "markPriceKlines",
    "indexPriceKlines",
    "premiumIndexKlines",
]

CROSS_ASSET_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


def create_http_session(pool_size: int = 32) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def verify_sha256_checksum(zip_path: Path, checksum_path: Path) -> bool:
    """Verifies that the downloaded .zip matches Binance's published .CHECKSUM file."""
    if not zip_path.exists() or not checksum_path.exists():
        return False
    try:
        expected_hash = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
        sha256 = hashlib.sha256()
        with open(zip_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest().lower() == expected_hash
    except Exception:
        return False


def build_monthly_kline_urls(
    market: str,          # "futures/um", "futures/cm", or "spot"
    data_type: str,       # "klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines"
    symbol: str,          # e.g. "BTCUSDT"
    interval: str,        # e.g. "15m", "1h", "4h", "1d"
    year_months: Iterable[str],  # e.g. ["2023-01", "2023-02", ..., "2026-09"]
) -> list[str]:
    """Constructs relative S3 paths for monthly kline archives."""
    rel_paths = []
    for ym in year_months:
        fname = f"{symbol}-{interval}-{ym}.zip"
        rel = f"data/{market}/monthly/{data_type}/{symbol}/{interval}/{fname}"
        rel_paths.append(rel)
    return rel_paths


def build_daily_kline_urls(
    market: str,
    data_type: str,
    symbol: str,
    interval: str,
    days: Iterable[date],
) -> list[str]:
    """Constructs relative S3 paths for daily kline archives."""
    rel_paths = []
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        fname = f"{symbol}-{interval}-{ds}.zip"
        rel = f"data/{market}/daily/{data_type}/{symbol}/{interval}/{fname}"
        rel_paths.append(rel)
    return rel_paths


def build_auxiliary_urls(
    symbol: str,
    year_months: Iterable[str],
    days: Iterable[date],
    include_agg_trades: bool = True,
) -> list[str]:
    """Constructs URLs for fundingRate (monthly), metrics (daily), bookDepth (daily), and aggTrades."""
    rels = []
    for ym in year_months:
        rels.append(f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip")
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        rels.append(f"data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{ds}.zip")
        rels.append(f"data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{ds}.zip")
        if symbol == "BTCUSDT":
            rels.append(f"data/option/daily/BVOLIndex/BTCBVOLUSDT/BTCBVOLUSDT-BVOLIndex-{ds}.zip")
        if include_agg_trades:
            rels.append(f"data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{ds}.zip")
            rels.append(f"data/spot/daily/klines/{symbol}/1s/{symbol}-1s-{ds}.zip")
    return rels


def download_single_archive(
    rel_path: str,
    output_root: Path,
    session: requests.Session,
    verify_checksum: bool = True,
) -> tuple[str, str]:
    """Downloads a single .zip and its .CHECKSUM if not already valid on disk."""
    dest_zip = output_root / rel_path
    dest_chk = output_root / f"{rel_path}.CHECKSUM"
    dest_zip.parent.mkdir(parents=True, exist_ok=True)

    if dest_zip.exists() and dest_zip.stat().st_size > 0:
        if not verify_checksum or (dest_chk.exists() and verify_sha256_checksum(dest_zip, dest_chk)):
            return (rel_path, "SKIPPED_VALID")

    url_zip = f"{DOWNLOAD_BASE_URL}/{rel_path}"
    url_chk = f"{url_zip}.CHECKSUM"

    try:
        resp = session.get(url_zip, stream=True, timeout=30)
        if resp.status_code == 404:
            return (rel_path, "MISSING_404")
        if resp.status_code != 200:
            return (rel_path, f"HTTP_{resp.status_code}")

        tmp_zip = dest_zip.with_suffix(".zip.part")
        with open(tmp_zip, "wb") as f:
            for chunk in resp.iter_content(chunk_size=131072):
                if chunk:
                    f.write(chunk)
        tmp_zip.replace(dest_zip)

        if verify_checksum:
            r_chk = session.get(url_chk, timeout=15)
            if r_chk.status_code == 200:
                dest_chk.write_bytes(r_chk.content)
                if not verify_sha256_checksum(dest_zip, dest_chk):
                    dest_zip.unlink(missing_ok=True)
                    return (rel_path, "CHECKSUM_MISMATCH")
        return (rel_path, "DOWNLOADED_OK")
    except Exception as exc:
        return (rel_path, f"ERROR:{exc}")
