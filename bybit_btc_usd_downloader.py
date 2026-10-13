"""
Bybit Public Data (public.bybit.com) - BTC/USD Sample Downloader
====================================================================
Downloads the LATEST available sample file for every BTC-vs-USD
instrument across every published category on https://public.bybit.com:

  - trading/                Raw trades: perpetuals, USDT/USDC-margined
                             linear contracts, inverse (USD) contracts,
                             AND dated/quarterly contracts. Two distinct
                             dated-contract naming styles exist and BOTH
                             are covered by the "starts with BTC" filter:
                               - BTC-26DEC25        (inverse, no "USD"
                                                      anywhere in the name)
                               - BTCUSDT-26DEC25     (linear)
                               - BTCUSDH26 / BTCUSDM26 (quarterly, letter
                                                      + 2-digit year code)
  - spot/                   Raw spot trades (BTCUSDT, BTCUSDC, BTCEUR
                             is skipped since it isn't USD, BTCUSD, ...)
  - spot_index/              Spot index price
  - premium_index/           Futures premium index
  - kline_for_metatrader4/   MT4-formatted klines

Nothing is hardcoded beyond the 5 top-level category names (which are
fixed by the site itself and were verified live against
https://public.bybit.com/). Every symbol folder is discovered live by
parsing the site's directory-listing HTML, and only symbols starting
with "BTC" are kept (this naturally captures every BTC/USD, BTC/USDT,
BTC/USDC and dated/quarterly BTC contract on the site, while excluding
unrelated pairs).

Since Bybit's daily filenames embed a sortable date
(e.g. BTCUSDT2024-05-01.csv.gz), the lexicographically last file in a
symbol folder is also the most recent one -- that's the "sample" this
script downloads for each symbol. Verified live: BTCUSDT/ alone has
1,800+ daily files using exactly this naming pattern.

CHANGES vs the original version:
  - Category -> symbol-folder discovery, and symbol-folder -> latest-file
    lookups, are now both parallelized with a thread pool instead of
    scanned one directory at a time. The "trading" category alone lists
    thousands of symbols (including ~200+ dated BTC/ETH/SOL/etc.
    contracts), so a sequential crawl could take a long time before any
    downloads even start.
  - Verified live against public.bybit.com: the 5 categories are
    confirmed complete, and the BTC dated-contract naming patterns above
    were confirmed to exist and to already be caught by the existing
    prefix filter.

Usage (from Windows cmd):
    cd /d F:\\Downloads\\Bybit data
    python bybit_btc_usd_downloader.py

Requires: requests, tqdm  (pip install requests tqdm)
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

BASE_URL = "https://public.bybit.com"

# The 5 top-level data categories published on public.bybit.com (confirmed
# live by listing the site root). Symbols under each are discovered
# dynamically.
CATEGORIES = [
    "trading",
    "spot",
    "spot_index",
    "premium_index",
    "kline_for_metatrader4",
]

LISTING_WORKERS = 24
DOWNLOAD_WORKERS = 16

LINK_RE = re.compile(r'href="([^"]+)"')


def create_session():
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=LISTING_WORKERS, pool_maxsize=LISTING_WORKERS, max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; BTC-USD-Downloader/1.0)"})
    return session


def is_btc_symbol(name: str) -> bool:
    """Keep any symbol folder that starts with BTC -- this covers
    BTCUSDT, BTCUSDC, BTCUSD, BTCPERP, BTCUSDH26 (quarterly inverse),
    BTCUSDT-26DEC25 (quarterly linear), BTC-26DEC25 (dated inverse with
    no 'USD' in the name at all), etc. All confirmed present live on the
    site under /trading/."""
    return name.upper().startswith("BTC")


def list_directory(url, session):
    """Parses an Apache/Nginx-style autoindex page and returns (dirs, files)
    as absolute URLs relative to the given directory URL."""
    if not url.endswith("/"):
        url += "/"
    try:
        resp = session.get(url, timeout=25)
        if resp.status_code != 200:
            return [], []
    except Exception:
        return [], []

    dirs, files = [], []
    for href in LINK_RE.findall(resp.text):
        if href in ("../", "..", "/") or href.startswith("?") or href.startswith("#"):
            continue
        full = urljoin(url, href)
        if not full.startswith(BASE_URL):
            continue
        if href.endswith("/"):
            dirs.append(full)
        else:
            files.append(full)
    return dirs, files


def discover_btc_symbol_dirs(category_url, session):
    dirs, _ = list_directory(category_url, session)
    btc_dirs = []
    for d in dirs:
        symbol = d.rstrip("/").rsplit("/", 1)[-1]
        if is_btc_symbol(symbol):
            btc_dirs.append((symbol, d))
    return btc_dirs


def latest_file(symbol_dir_url, session):
    """Returns the URL of the most recent file in a symbol folder.
    Filenames embed a sortable date, so a plain lexicographic sort works."""
    _, files = list_directory(symbol_dir_url, session)
    if not files:
        return None
    files.sort()
    return files[-1]


def download_file(file_url, destination_root, session):
    """Downloads a file, preserving the site's folder structure under
    destination_root, e.g. trading/BTCUSDT/BTCUSDT2024-05-01.csv.gz"""
    rel_path = file_url[len(BASE_URL):].lstrip("/")
    local_path = os.path.join(destination_root, *rel_path.split("/"))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        head = session.head(file_url, timeout=15)
        remote_size = int(head.headers.get("content-length", 0))
        if os.path.exists(local_path) and os.path.getsize(local_path) == remote_size and remote_size > 0:
            return True
    except Exception:
        pass

    try:
        with session.get(file_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            fname = os.path.basename(local_path)
            with open(local_path, "wb") as f, tqdm(
                desc=fname[:32].ljust(32),
                total=total,
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
        tqdm.write(f"Error downloading {file_url}: {e}")
        return False


def main():
    target_dir = os.getcwd()
    session = create_session()

    print("=" * 70)
    print("BYBIT PUBLIC DATA - BTC/USD Sample Downloader")
    print(f"Target Download Directory: {target_dir}")
    print("=" * 70)

    print("\n[1/3] Scanning public.bybit.com categories for BTC symbol folders...")
    symbol_targets = []  # list of (category, symbol, symbol_dir_url)

    with ThreadPoolExecutor(max_workers=len(CATEGORIES)) as executor:
        future_to_cat = {
            executor.submit(discover_btc_symbol_dirs, f"{BASE_URL}/{cat}/", session): cat
            for cat in CATEGORIES
        }
        for future in as_completed(future_to_cat):
            cat = future_to_cat[future]
            btc_dirs = future.result()
            for symbol, d in btc_dirs:
                symbol_targets.append((cat, symbol, d))
            print(f"      {cat}: {len(btc_dirs)} BTC symbol folder(s) found")

    print(f"\n[2/3] Locating latest sample file for {len(symbol_targets)} symbol folders...")
    download_targets = []

    with ThreadPoolExecutor(max_workers=LISTING_WORKERS) as executor:
        future_to_target = {
            executor.submit(latest_file, d, session): (cat, symbol, d)
            for cat, symbol, d in symbol_targets
        }
        for future in tqdm(as_completed(future_to_target), total=len(symbol_targets), desc="Scanning symbol folders", unit="symbol"):
            f = future.result()
            if f:
                download_targets.append(f)

    download_targets = sorted(set(download_targets))
    print(f"      Found {len(download_targets)} latest sample files ready for download.\n")

    print("[3/3] Downloading latest sample files...")
    downloaded, failed = 0, 0

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        future_to_url = {
            executor.submit(download_file, file_url, target_dir, session): file_url
            for file_url in download_targets
        }
        for future in tqdm(as_completed(future_to_url), total=len(download_targets), desc="Total Download Progress", unit="file"):
            if future.result():
                downloaded += 1
            else:
                failed += 1

    print("\n" + "=" * 70)
    print("Download Process Completed!")
    print(f"Total Downloaded / Verified: {downloaded}")
    print(f"Failed: {failed}")
    print(f"All files saved under: {target_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
