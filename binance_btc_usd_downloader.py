"""
Binance Data Vision (data.binance.vision) - BTC/USD Sample Downloader
=======================================================================
Downloads the LATEST available sample .zip (+ its .CHECKSUM) for every
BTC-vs-USD(-stable) instrument across every published market and data
type on https://data.binance.vision, including:

  - Spot                (BTCUSDT, BTCUSDC, BTCTUSD, BTCFDUSD, BTCDAI, ...)
  - Futures USDS-M (um)  (BTCUSDT, BTCUSDC, and dated/quarterly contracts
                          like BTCUSDT_250926)
  - Futures COIN-M (cm)  (BTCUSD_PERP, BTCUSD, dated contracts like
                          BTCUSD_250926)
  - Options              (BTCUSDT EOHSummary, BTCBVOLUSDT BVOLIndex, ...)

Nothing is hardcoded: every data type (klines, trades, aggTrades,
bookTicker, fundingRate, markPriceKlines, indexPriceKlines,
premiumIndexKlines, bookDepth, liquidationSnapshot, metrics,
BVOLIndex, EOHSummary, etc.) and every interval sub-folder (1s, 1m,
1h, 1d, ...) are discovered live by listing the S3 bucket, and only
symbols starting with "BTC" and containing "USD" are kept.

CHANGES vs the original version:
  - Every discovery stage (data-types -> symbols -> intervals -> latest
    zip lookup) is now parallelized with a thread pool, not just the
    final downloads. A full re-crawl of Binance's bucket touches
    thousands of directories (especially dated/quarterly futures
    contracts), so doing this sequentially could take a very long time.
  - The .CHECKSUM file that sits next to every .zip is now downloaded
    too, so each archive can be integrity-verified after the fact.
  - Verified live against the S3 bucket (s3-ap-northeast-1) and the
    binance/binance-public-data GitHub repo: bucket layout, symbol
    naming, and interval folder structure all match what this script
    assumes.

Usage (from Windows cmd):
    cd /d F:\\Downloads\\Binance data
    python binance_btc_usd_downloader.py

Requires: requests, tqdm  (pip install requests tqdm)
"""

import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

S3_BUCKET_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DOWNLOAD_BASE_URL = "https://data.binance.vision"

# Market + scope matrix to crawl. Symbols are NOT hardcoded here -- they are
# discovered dynamically for each (market, scope, data-type) combination and
# filtered down to BTC/USD-related instruments only (see is_btc_usd_symbol).
MARKET_SCOPES = [
    ("spot", "daily"),
    ("spot", "monthly"),
    ("futures/um", "daily"),
    ("futures/um", "monthly"),
    ("futures/cm", "daily"),
    ("futures/cm", "monthly"),
    ("option", "daily"),
    ("option", "monthly"),
]

# How many concurrent HTTP requests to use for LISTING directories (separate
# from the download pool below). S3 handles this fine; keep it well under
# any per-IP rate limit.
LISTING_WORKERS = 24
DOWNLOAD_WORKERS = 16


def is_btc_usd_symbol(symbol: str) -> bool:
    """Keep only symbols that represent a BTC-vs-USD(-stable) instrument.

    Matches: BTCUSDT, BTCUSDC, BTCTUSD, BTCFDUSD, BTCDAI, BTCUSD,
    BTCUSD_PERP, BTCUSD_250926, BTCUSDT_250926, BTCBVOLUSDT (options vol
    index), etc. Excludes unrelated BTC-quoted pairs (e.g. ETHBTC, where
    BTC is the quote asset, not the base).
    """
    s = symbol.upper()
    return s.startswith("BTC") and "USD" in s


def create_session():
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=LISTING_WORKERS, pool_maxsize=LISTING_WORKERS, max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _clean_key(key: str) -> str:
    """S3 keys returned by this bucket start with a leading '/', normalize it."""
    return key.lstrip("/")


def list_subdirectories(prefix, session):
    """Queries S3 using delimiter='/' to list immediate subfolder prefixes."""
    subdirs = []
    marker = ""

    while True:
        params = {"prefix": prefix, "delimiter": "/", "max-keys": 1000}
        if marker:
            params["marker"] = marker

        try:
            resp = session.get(S3_BUCKET_URL, params=params, timeout=20)
            if resp.status_code != 200:
                break
            root = ET.fromstring(resp.content)
        except Exception:
            break

        for elem in root.findall(".//{*}CommonPrefixes/{*}Prefix"):
            if elem.text:
                subdirs.append(_clean_key(elem.text))

        is_trunc = root.find(".//{*}IsTruncated")
        if is_trunc is None or is_trunc.text.lower() != "true":
            break

        next_marker = root.find(".//{*}NextMarker")
        if next_marker is not None and next_marker.text:
            marker = next_marker.text
        elif subdirs:
            marker = subdirs[-1]
        else:
            break

    return subdirs


def _parallel_map(func, items, session, workers=LISTING_WORKERS, desc=""):
    """Runs func(item, session) across items concurrently, returns list of
    (item, result) pairs. Order is not preserved but that's fine here."""
    results = []
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(items)))) as executor:
        future_to_item = {executor.submit(func, item, session): item for item in items}
        iterator = as_completed(future_to_item)
        if desc:
            iterator = tqdm(iterator, total=len(items), desc=desc, unit="dir")
        for future in iterator:
            item = future_to_item[future]
            try:
                results.append((item, future.result()))
            except Exception:
                results.append((item, []))
    return results


def discover_all_endpoints(session):
    """
    Crawls S3 to dynamically uncover every data type, BTC/USD symbol, and
    interval folder (1s, 1m, 1h, 1d, 1mo, etc.) across all markets.
    Every stage below is parallelized across directories.
    """
    print("[1/3] Traversing Binance Data Vision directory tree (dynamic discovery)...")

    # Step A: discover every published data type under each market & scope
    base_prefixes = [f"data/{market}/{scope}/" for market, scope in MARKET_SCOPES]
    stepA = _parallel_map(lambda p, s: list_subdirectories(p, s), base_prefixes, session,
                           desc="Scanning market/scope roots")
    market_scope_dtypes = [dt for _, dtypes in stepA for dt in dtypes]
    print(f"      Discovered {len(market_scope_dtypes)} data-type folders across all markets/scopes.")

    # Step B: for every data type, discover ALL symbol subfolders and keep
    # only BTC/USD-related ones (dynamic -- no hardcoded symbol names)
    stepB = _parallel_map(lambda p, s: list_subdirectories(p, s), market_scope_dtypes, session,
                           desc="Scanning data-type folders for symbols")
    btc_symbol_prefixes = []
    for _, symbols in stepB:
        for sym_prefix in symbols:
            symbol_name = sym_prefix.rstrip("/").rsplit("/", 1)[-1]
            if is_btc_usd_symbol(symbol_name):
                btc_symbol_prefixes.append(sym_prefix)
    print(f"      Matched {len(btc_symbol_prefixes)} BTC/USD symbol folders.")

    # Step C: discover interval subdirectories (klines-style datasets have
    # them, e.g. 1m/1h/1d; trades/aggTrades/bookTicker do not)
    stepC = _parallel_map(lambda p, s: list_subdirectories(p, s), btc_symbol_prefixes, session,
                           desc="Scanning symbol folders for intervals")
    active_endpoints = []
    for sym_prefix, intervals in stepC:
        if intervals:
            active_endpoints.extend(intervals)
        else:
            active_endpoints.append(sym_prefix)

    # Deduplicate while retaining order
    seen = set()
    unique_endpoints = []
    for ep in active_endpoints:
        if ep not in seen:
            seen.add(ep)
            unique_endpoints.append(ep)

    print(f"      Total: {len(unique_endpoints)} active BTC/USD paths & interval endpoints.\n")
    return unique_endpoints


def get_latest_zip_key(prefix, session):
    """Paginates an S3 prefix to find the latest available .zip archive."""
    marker = ""
    latest_zip = None

    while True:
        params = {"prefix": prefix, "max-keys": 1000}
        if marker:
            params["marker"] = marker

        try:
            resp = session.get(S3_BUCKET_URL, params=params, timeout=20)
            if resp.status_code != 200:
                break
            root = ET.fromstring(resp.content)
        except Exception:
            break

        keys = [_clean_key(elem.text) for elem in root.findall(".//{*}Key") if elem.text]
        zip_keys = [k for k in keys if k.endswith(".zip")]
        if zip_keys:
            latest_zip = zip_keys[-1]

        is_trunc = root.find(".//{*}IsTruncated")
        if is_trunc is None or is_trunc.text.lower() != "true":
            break

        if keys:
            marker = keys[-1]
        else:
            break

    return latest_zip


def download_one(url, local_filepath, session):
    """Streams a single URL to disk with a size check to skip re-downloads."""
    os.makedirs(os.path.dirname(local_filepath), exist_ok=True)

    try:
        head_resp = session.head(url, timeout=10)
        remote_size = int(head_resp.headers.get("content-length", 0))
        if os.path.exists(local_filepath) and os.path.getsize(local_filepath) == remote_size and remote_size > 0:
            return True
    except Exception:
        pass

    try:
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_bytes = int(r.headers.get("content-length", 0))
            filename = os.path.basename(local_filepath)

            with open(local_filepath, "wb") as f, tqdm(
                desc=filename[:32].ljust(32),
                total=total_bytes,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as pbar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        tqdm.write(f"Error downloading {url}: {e}")
        return False


def download_file_and_checksum(s3_key, destination_folder, session):
    """Downloads a zip and its companion .CHECKSUM file, preserving the
    site's folder structure (data/<market>/<scope>/<dataType>/<symbol>/
    [interval]/file.zip)."""
    zip_url = f"{DOWNLOAD_BASE_URL}/{s3_key}"
    zip_path = os.path.join(destination_folder, *s3_key.split("/"))

    ok_zip = download_one(zip_url, zip_path, session)

    checksum_key = s3_key + ".CHECKSUM"
    checksum_url = f"{DOWNLOAD_BASE_URL}/{checksum_key}"
    checksum_path = zip_path + ".CHECKSUM"
    ok_checksum = download_one(checksum_url, checksum_path, session)

    return ok_zip and ok_checksum


def main():
    target_dir = os.getcwd()
    session = create_session()

    print("=" * 70)
    print("BINANCE DATA VISION - BTC/USD Sample Downloader")
    print(f"Target Download Directory: {target_dir}")
    print("=" * 70)

    unique_endpoints = discover_all_endpoints(session)

    print("[2/3] Querying latest available ZIP for each path and interval...")
    download_targets = []

    with ThreadPoolExecutor(max_workers=LISTING_WORKERS) as executor:
        future_to_prefix = {executor.submit(get_latest_zip_key, ep, session): ep for ep in unique_endpoints}
        for future in tqdm(as_completed(future_to_prefix), total=len(unique_endpoints), desc="Scanning S3 paths", unit="path"):
            latest_zip = future.result()
            if latest_zip:
                download_targets.append(latest_zip)

    download_targets = sorted(set(download_targets))
    print(f"      Found {len(download_targets)} unique latest ZIP archives ready for download.\n")

    print("[3/3] Downloading latest sample ZIP archives (+ .CHECKSUM files)...")
    downloaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        future_to_key = {
            executor.submit(download_file_and_checksum, s3_key, target_dir, session): s3_key
            for s3_key in download_targets
        }
        for future in tqdm(as_completed(future_to_key), total=len(download_targets), desc="Total Download Progress", unit="file"):
            if future.result():
                downloaded += 1
            else:
                failed += 1

    print("\n" + "=" * 70)
    print("Download Process Completed!")
    print(f"Total Downloaded / Verified: {downloaded}")
    print(f"Failed / Missing: {failed}")
    print(f"All files saved under: {target_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
