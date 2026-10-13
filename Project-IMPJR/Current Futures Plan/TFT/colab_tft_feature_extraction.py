"""Google Colab script (run top-to-bottom) to build a 10-second TFT feature dataset
from Binance Vision ZIPs and upload a Parquet file to a Hugging Face dataset repo.

Notes
- Uses Polars for vectorized, memory-efficient transforms.
- Avoids pandas .apply, uses group_by_dynamic/rolling expressions.
- Parallelizes downloads; extraction is streaming-friendly.
- DOES NOT hard-code any Hugging Face token.

In Colab
1) Runtime > Change runtime type > (recommended) High-RAM
2) Set HF_TOKEN as a secret/env var (do NOT paste token into notebooks you share)
   - Option A: `import os; os.environ['HF_TOKEN'] = '...';`
   - Option B: Colab Secrets / environment

Outputs
- Local: ./out/features_10s_BTCUSDT_YYYY-MM-DD.parquet
- Remote: uploaded into https://huggingface.co/datasets/p-j-r-1-2-3/Project-IMPJR-ModelTFT-dataset

"""

# =========================
# Cell 1: Install deps
# =========================

# In Colab, uncomment:
# !pip -q install "polars>=1.10.0" "pyarrow>=14.0.0" "huggingface_hub>=0.20.0" "tqdm>=4.66.0" "requests>=2.31.0"

from __future__ import annotations

import io
import os
import math
import zipfile
import shutil
import datetime as dt
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm.auto import tqdm

import polars as pl

from huggingface_hub import HfApi


# =========================
# Cell 2: Config
# =========================

@dataclass(frozen=True)
class Inputs:
    # Dates correspond to the ZIP names you listed
    futures_trades_date: str = "2025-12-31"
    futures_aggtrades_date: str = "2025-12-31"
    futures_bookdepth_date: str = "2025-12-31"
    futures_klines_1m_date: str = "2025-12-30"
    futures_mark_1m_date: str = "2025-12-30"
    futures_premium_1m_date: str = "2025-12-30"
    futures_metrics_date: str = "2025-12-31"
    futures_funding_month: str = "2025-12"  # monthly

    spot_klines_1s_date: str = "2025-12-31"
    spot_aggtrades_date: str = "2025-12-31"

    bvol_date: str = "2025-12-14"

    symbol_um: str = "BTCUSDT"
    symbol_spot: str = "BTCUSDT"
    symbol_bvol: str = "BTCBVOLUSDT"

    # Target dataset repo
    hf_repo_id: str = "p-j-r-1-2-3/Project-IMPJR-ModelTFT-dataset"
    hf_repo_path_in_repo: str = "data"  # folder inside the dataset repo

    # 10s grid settings
    bar_seconds: int = 10
    tz: str = "UTC"


CFG = Inputs()

BASE = "https://data.binance.vision/data"

WORKDIR = os.getcwd()
RAW_DIR = os.path.join(WORKDIR, "raw")
ZIP_DIR = os.path.join(WORKDIR, "zips")
OUT_DIR = os.path.join(WORKDIR, "out")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

pl.Config.set_tbl_rows(20)


# =========================
# Cell 3: URLs
# =========================

def _url(*parts: str) -> str:
    return "/".join([BASE.strip("/")] + [p.strip("/") for p in parts])


def build_download_plan(cfg: Inputs) -> list[tuple[str, str]]:
    """Returns list of (url, local_zip_path)."""

    items: list[tuple[str, str]] = []

    # Futures UM daily
    items.append((
        _url("futures/um/daily/trades", cfg.symbol_um, f"{cfg.symbol_um}-trades-{cfg.futures_trades_date}.zip"),
        os.path.join(ZIP_DIR, f"um_trades_{cfg.symbol_um}_{cfg.futures_trades_date}.zip"),
    ))
    items.append((
        _url("futures/um/daily/aggTrades", cfg.symbol_um, f"{cfg.symbol_um}-aggTrades-{cfg.futures_aggtrades_date}.zip"),
        os.path.join(ZIP_DIR, f"um_aggtrades_{cfg.symbol_um}_{cfg.futures_aggtrades_date}.zip"),
    ))
    items.append((
        _url("futures/um/daily/bookDepth", cfg.symbol_um, f"{cfg.symbol_um}-bookDepth-{cfg.futures_bookdepth_date}.zip"),
        os.path.join(ZIP_DIR, f"um_bookdepth_{cfg.symbol_um}_{cfg.futures_bookdepth_date}.zip"),
    ))

    # Futures UM daily 1m klines/mark/premium
    items.append((
        _url("futures/um/daily/klines", cfg.symbol_um, "1m", f"{cfg.symbol_um}-1m-{cfg.futures_klines_1m_date}.zip"),
        os.path.join(ZIP_DIR, f"um_klines1m_{cfg.symbol_um}_{cfg.futures_klines_1m_date}.zip"),
    ))
    items.append((
        _url("futures/um/daily/markPriceKlines", cfg.symbol_um, "1m", f"{cfg.symbol_um}-1m-{cfg.futures_mark_1m_date}.zip"),
        os.path.join(ZIP_DIR, f"um_mark1m_{cfg.symbol_um}_{cfg.futures_mark_1m_date}.zip"),
    ))
    items.append((
        _url("futures/um/daily/premiumIndexKlines", cfg.symbol_um, "1m", f"{cfg.symbol_um}-1m-{cfg.futures_premium_1m_date}.zip"),
        os.path.join(ZIP_DIR, f"um_premium1m_{cfg.symbol_um}_{cfg.futures_premium_1m_date}.zip"),
    ))

    items.append((
        _url("futures/um/daily/metrics", cfg.symbol_um, f"{cfg.symbol_um}-metrics-{cfg.futures_metrics_date}.zip"),
        os.path.join(ZIP_DIR, f"um_metrics_{cfg.symbol_um}_{cfg.futures_metrics_date}.zip"),
    ))

    # Futures UM monthly funding rate
    items.append((
        _url("futures/um/monthly/fundingRate", cfg.symbol_um, f"{cfg.symbol_um}-fundingRate-{cfg.futures_funding_month}.zip"),
        os.path.join(ZIP_DIR, f"um_funding_{cfg.symbol_um}_{cfg.futures_funding_month}.zip"),
    ))

    # Spot daily
    items.append((
        _url("spot/daily/klines", cfg.symbol_spot, "1s", f"{cfg.symbol_spot}-1s-{cfg.spot_klines_1s_date}.zip"),
        os.path.join(ZIP_DIR, f"spot_klines1s_{cfg.symbol_spot}_{cfg.spot_klines_1s_date}.zip"),
    ))
    items.append((
        _url("spot/daily/aggTrades", cfg.symbol_spot, f"{cfg.symbol_spot}-aggTrades-{cfg.spot_aggtrades_date}.zip"),
        os.path.join(ZIP_DIR, f"spot_aggtrades_{cfg.symbol_spot}_{cfg.spot_aggtrades_date}.zip"),
    ))

    # Options daily BVOL
    items.append((
        _url("option/daily/BVOLIndex", cfg.symbol_bvol, f"{cfg.symbol_bvol}-BVOLIndex-{cfg.bvol_date}.zip"),
        os.path.join(ZIP_DIR, f"bvol_{cfg.symbol_bvol}_{cfg.bvol_date}.zip"),
    ))

    return items


# =========================
# Cell 4: Fast download + extract
# =========================

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "feature-extractor/1.0"})


def download_file(url: str, dst_path: str, timeout: int = 120) -> None:
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return

    tmp_path = dst_path + ".partial"
    with SESSION.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, dst_path)


def download_many(plan: list[tuple[str, str]], max_workers: int = 6) -> None:
    """Parallel download (I/O bound)."""
    failures: list[tuple[str, str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(download_file, url, path): (url, path) for url, path in plan}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading ZIPs"):
            url, path = futs[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append((url, path, str(e)))

    if failures:
        msg = "\n".join([f"- {u} -> {p}: {err}" for u, p, err in failures])
        raise RuntimeError(f"Some downloads failed:\n{msg}")


def extract_zip(zip_path: str, dst_dir: str) -> list[str]:
    """Extracts all files; returns extracted file paths."""
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            out_path = os.path.join(dst_dir, os.path.basename(name))
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                extracted.append(out_path)
                continue
            with zf.open(name) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            extracted.append(out_path)
    return extracted


# =========================
# Cell 5: Robust CSV reading
# =========================


def _flex_read_csv(path: str, *, schema_overrides: dict[str, pl.DataType] | None = None) -> pl.LazyFrame:
    """Binance CSVs are typically comma-separated with headers; this tries , then \t."""
    schema_overrides = schema_overrides or {}

    # Try comma
    try:
        lf = pl.scan_csv(
            path,
            separator=",",
            infer_schema_length=10_000,
        )
        # quick check: if only 1 column but file should have more, retry with tab
        if len(lf.columns) <= 1:
            raise ValueError("Too few columns with comma separator")
        if schema_overrides:
            lf = lf.with_columns([pl.col(c).cast(t, strict=False).alias(c) for c, t in schema_overrides.items()])
        return lf
    except Exception:
        lf = pl.scan_csv(path, separator="\t", infer_schema_length=10_000)
        if schema_overrides:
            lf = lf.with_columns([pl.col(c).cast(t, strict=False).alias(c) for c, t in schema_overrides.items()])
        return lf


def normalize_epoch_to_ms(expr: pl.Expr) -> pl.Expr:
    """Normalize epoch timestamps that may be ms/us/ns into int64 ms.

    Heuristic by magnitude:
    - >= 1e17 : ns
    - >= 1e14 : us
    - else    : ms
    """
    x = expr.cast(pl.Int64, strict=False)
    return (
        pl.when(x >= 10**17).then((x // 10**6))
        .when(x >= 10**14).then((x // 10**3))
        .otherwise(x)
        .cast(pl.Int64)
    )


def ensure_bool(expr: pl.Expr) -> pl.Expr:
    # Binance is_buyer_maker can be TRUE/FALSE strings
    return (
        pl.when(expr.is_null()).then(None)
        .when(expr.cast(pl.Utf8, strict=False).str.to_uppercase() == "TRUE").then(True)
        .when(expr.cast(pl.Utf8, strict=False).str.to_uppercase() == "FALSE").then(False)
        .otherwise(expr.cast(pl.Boolean, strict=False))
    )


# =========================
# Cell 6: Dataset-specific loaders
# =========================


def load_um_trades_10s(csv_path: str, bar_seconds: int = 10) -> pl.DataFrame:
    # id, price, qty, quote_qty, time, is_buyer_maker
    lf = _flex_read_csv(
        csv_path,
        schema_overrides={
            "price": pl.Float64,
            "qty": pl.Float64,
            "quote_qty": pl.Float64,
        },
    )

    lf = lf.with_columns([
        normalize_epoch_to_ms(pl.col("time")).alias("ts_ms"),
        ensure_bool(pl.col("is_buyer_maker")).alias("is_buyer_maker"),
    ]).drop(["time"]).with_columns([
        (pl.col("ts_ms") * 1_000_000).cast(pl.Datetime("ns")).alias("ts"),
    ])

    # 95th percentile threshold for large trades
    qty_q95 = lf.select(pl.col("qty").quantile(0.95)).collect().item()

    every = f"{bar_seconds}s"
    agg = (
        lf.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="left")
        .agg([
            pl.col("price").first().alias("fut_open"),
            pl.col("price").max().alias("fut_high"),
            pl.col("price").min().alias("fut_low"),
            pl.col("price").last().alias("fut_close"),
            pl.col("qty").sum().alias("fut_volume"),
            pl.col("quote_qty").sum().alias("fut_quote_volume"),
            pl.len().alias("fut_trade_count"),
            pl.when(~pl.col("is_buyer_maker")).then(pl.col("qty")).otherwise(0.0).sum().alias("fut_buy_volume"),
            pl.when(pl.col("is_buyer_maker")).then(pl.col("qty")).otherwise(0.0).sum().alias("fut_sell_volume"),
            (pl.col("price") * pl.col("qty")).sum().alias("_px_qty"),
            pl.when(pl.col("qty") > qty_q95).then(1).otherwise(0).sum().alias("fut_large_trade_count"),
            pl.when(pl.col("qty") > qty_q95).then(pl.col("qty")).otherwise(0.0).sum().alias("fut_large_trade_volume"),
        ])
        .with_columns([
            (pl.col("_px_qty") / pl.col("fut_volume")).alias("fut_vwap"),
        ])
        .drop(["_px_qty"])
    )

    return agg.collect()


def load_um_aggtrades_10s(csv_path: str, bar_seconds: int = 10) -> pl.DataFrame:
    # agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
    lf = _flex_read_csv(
        csv_path,
        schema_overrides={
            "price": pl.Float64,
            "quantity": pl.Float64,
        },
    )
    lf = lf.with_columns([
        normalize_epoch_to_ms(pl.col("transact_time")).alias("ts_ms"),
        ensure_bool(pl.col("is_buyer_maker")).alias("is_buyer_maker"),
        (pl.col("ts_ms") * 1_000_000).cast(pl.Datetime("ns")).alias("ts"),
    ])

    qty_q95 = lf.select(pl.col("quantity").quantile(0.95)).collect().item()

    every = f"{bar_seconds}s"
    agg = (
        lf.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="left")
        .agg([
            pl.col("quantity").sum().alias("agg_qty"),
            pl.len().alias("agg_count"),
            pl.when(~pl.col("is_buyer_maker")).then(pl.col("quantity")).otherwise(0.0).sum().alias("agg_buy_qty"),
            pl.when(pl.col("is_buyer_maker")).then(pl.col("quantity")).otherwise(0.0).sum().alias("agg_sell_qty"),
            pl.when(pl.col("quantity") > qty_q95).then(1).otherwise(0).sum().alias("agg_large_count"),
            pl.when(pl.col("quantity") > qty_q95).then(pl.col("quantity")).otherwise(0.0).sum().alias("agg_large_qty"),
        ])
    )

    return agg.collect()


def load_spot_klines_1s_to_10s(csv_path: str, bar_seconds: int = 10) -> pl.DataFrame:
    # open_time, open, high, low, close, volume, close_time, quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore
    lf = _flex_read_csv(
        csv_path,
        schema_overrides={
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "quote_volume": pl.Float64,
            "count": pl.Int64,
            "taker_buy_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
        },
    )

    lf = lf.with_columns([
        normalize_epoch_to_ms(pl.col("open_time")).alias("ts_ms"),
        (pl.col("ts_ms") * 1_000_000).cast(pl.Datetime("ns")).alias("ts"),
    ])

    every = f"{bar_seconds}s"
    agg = (
        lf.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="left")
        .agg([
            pl.col("open").first().alias("spot_open"),
            pl.col("high").max().alias("spot_high"),
            pl.col("low").min().alias("spot_low"),
            pl.col("close").last().alias("spot_close"),
            pl.col("volume").sum().alias("spot_volume"),
            pl.col("quote_volume").sum().alias("spot_quote_volume"),
            pl.col("count").sum().alias("spot_trade_count"),
            (pl.col("close") * pl.col("volume")).sum().alias("_px_qty"),
        ])
        .with_columns([
            (pl.col("_px_qty") / pl.col("spot_volume")).alias("spot_vwap"),
        ])
        .drop(["_px_qty"])
    )

    return agg.collect()


def load_spot_aggtrades_10s(csv_path: str, bar_seconds: int = 10) -> pl.DataFrame:
    # spot aggTrades have 8 columns in sample; Binance spot aggTrades usually:
    # agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker, is_best_match
    lf = _flex_read_csv(
        csv_path,
        schema_overrides={
            "price": pl.Float64,
            "quantity": pl.Float64,
        },
    )

    # Some spot aggTrades use different header names. Try to map if needed.
    cols = set(lf.columns)
    time_col = "transact_time" if "transact_time" in cols else ("time" if "time" in cols else None)
    qty_col = "quantity" if "quantity" in cols else ("qty" if "qty" in cols else None)

    if time_col is None or qty_col is None:
        raise ValueError(f"Unexpected spot aggTrades columns: {lf.columns}")

    lf = lf.with_columns([
        normalize_epoch_to_ms(pl.col(time_col)).alias("ts_ms"),
        ensure_bool(pl.col("is_buyer_maker")).alias("is_buyer_maker"),
        (pl.col("ts_ms") * 1_000_000).cast(pl.Datetime("ns")).alias("ts"),
        pl.col(qty_col).cast(pl.Float64, strict=False).alias("qty"),
    ])

    every = f"{bar_seconds}s"
    agg = (
        lf.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="left")
        .agg([
            pl.col("qty").sum().alias("spot_agg_qty"),
            pl.len().alias("spot_agg_count"),
            pl.when(~pl.col("is_buyer_maker")).then(pl.col("qty")).otherwise(0.0).sum().alias("spot_agg_buy_qty"),
            pl.when(pl.col("is_buyer_maker")).then(pl.col("qty")).otherwise(0.0).sum().alias("spot_agg_sell_qty"),
        ])
    )
    return agg.collect()


def load_bookdepth_pivot(csv_path: str) -> pl.DataFrame:
    # timestamp, percentage, depth, notional
    lf = _flex_read_csv(
        csv_path,
        schema_overrides={
            "percentage": pl.Int64,
            "depth": pl.Float64,
            "notional": pl.Float64,
        },
    )

    # Parse timestamp like "12/17/2025 0:00". Assume UTC.
    # (If your file uses different format, adjust this strptime.)
    lf = lf.with_columns([
        pl.col("timestamp").cast(pl.Utf8).str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M", strict=False).alias("ts"),
    ])

    # Pivot depth and notional by percentage levels.
    df = lf.select(["ts", "percentage", "depth", "notional"]).collect()

    depth_w = df.pivot(index="ts", columns="percentage", values="depth", aggregate_function="first")
    depth_w = depth_w.rename({c: f"depth_{c}" for c in depth_w.columns if c != "ts"})

    notional_w = df.pivot(index="ts", columns="percentage", values="notional", aggregate_function="first")
    notional_w = notional_w.rename({c: f"notional_{c}" for c in notional_w.columns if c != "ts"})

    return depth_w.join(notional_w, on="ts", how="inner").sort("ts")


def load_1m_kline_asof(csv_path: str, prefix: str) -> pl.DataFrame:
    # For futures klines/mark/premium: open_time + OHLC...
    lf = _flex_read_csv(csv_path)
    if "open_time" not in lf.columns:
        raise ValueError(f"Expected open_time in {csv_path}, got {lf.columns}")

    # Cast numerics to float32/float64 as needed
    numeric_cols = [c for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"] if c in lf.columns]

    lf = lf.with_columns([
        normalize_epoch_to_ms(pl.col("open_time")).alias("ts_ms"),
        (pl.col("ts_ms") * 1_000_000).cast(pl.Datetime("ns")).alias("ts"),
        *[pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric_cols],
    ])

    # Keep only a minimal set; will be joined as-of to 10s grid
    keep = ["ts"] + [c for c in ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"] if c in lf.columns]
    df = lf.select(keep).collect().sort("ts")

    # Prefix columns to avoid collisions
    ren = {c: f"{prefix}_{c}" for c in df.columns if c != "ts"}
    return df.rename(ren)


def load_metrics_5m_asof(csv_path: str) -> pl.DataFrame:
    lf = _flex_read_csv(csv_path)

    # create_time like "12/17/2025 0:05" (UTC assumed)
    lf = lf.with_columns([
        pl.col("create_time").cast(pl.Utf8).str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M", strict=False).alias("ts"),
    ])

    keep = [
        "ts",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    keep = [c for c in keep if c in lf.columns]
    df = lf.select(keep).with_columns([pl.exclude("ts").cast(pl.Float64, strict=False)]).collect().sort("ts")
    return df


def load_funding_monthly_asof(csv_path: str) -> pl.DataFrame:
    lf = _flex_read_csv(csv_path)

    # calc_time epoch (often ms)
    lf = lf.with_columns([
        normalize_epoch_to_ms(pl.col("calc_time")).alias("ts_ms"),
        (pl.col("ts_ms") * 1_000_000).cast(pl.Datetime("ns")).alias("ts"),
        pl.col("last_funding_rate").cast(pl.Float64, strict=False).alias("funding_rate"),
        pl.col("funding_interval_hours").cast(pl.Int64, strict=False).alias("funding_interval_hours"),
    ])

    df = lf.select(["ts", "funding_rate", "funding_interval_hours"]).collect().sort("ts")
    return df


def load_bvol_asof(csv_path: str) -> pl.DataFrame:
    lf = _flex_read_csv(csv_path)
    lf = lf.with_columns([
        normalize_epoch_to_ms(pl.col("calc_time")).alias("ts_ms"),
        (pl.col("ts_ms") * 1_000_000).cast(pl.Datetime("ns")).alias("ts"),
        pl.col("index_value").cast(pl.Float64, strict=False).alias("bvol_index"),
    ])
    df = lf.select(["ts", "bvol_index"]).collect().sort("ts")
    return df


# =========================
# Cell 7: Feature engineering (vectorized)
# =========================


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.abs() > 0).then(num / den).otherwise(None)


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    # Known future inputs can be derived later for horizon; here we compute per-timestamp encodings.
    ts = pl.col("ts")
    hour = ts.dt.hour()
    minute = ts.dt.minute()
    weekday = ts.dt.weekday()

    return df.with_columns([
        (hour * 2 * math.pi / 24).sin().alias("hour_sin"),
        (hour * 2 * math.pi / 24).cos().alias("hour_cos"),
        (minute * 2 * math.pi / 60).sin().alias("minute_sin"),
        (minute * 2 * math.pi / 60).cos().alias("minute_cos"),
        (weekday * 2 * math.pi / 7).sin().alias("day_of_week_sin"),
        (weekday * 2 * math.pi / 7).cos().alias("day_of_week_cos"),

        (weekday >= 5).cast(pl.Int8).alias("weekend"),
        (weekday < 5).cast(pl.Int8).alias("weekday_flag"),

        # Simple session bins (UTC)
        ((hour >= 0) & (hour < 8)).cast(pl.Int8).alias("hour_0_8_asian"),
        ((hour >= 8) & (hour < 16)).cast(pl.Int8).alias("hour_8_16_europe"),
        ((hour >= 16) & (hour < 24)).cast(pl.Int8).alias("hour_16_24_us"),
    ])


def add_core_features(df: pl.DataFrame) -> pl.DataFrame:
    # Assumes columns exist from joins (fut_close, spot_close, depth_-2/+2 etc may be null)
    # Rolling windows are in 10s bars: 6=1m, 30=5m, 60=10m, 120=20m

    df = df.sort("ts")

    # Returns
    df = df.with_columns([
        (pl.col("fut_close").log().diff()).alias("futures_log_return"),
        (pl.col("spot_close").log().diff()).alias("spot_log_return"),

        safe_div(pl.col("fut_high") - pl.col("fut_low"), pl.col("fut_close")).alias("hl_ratio"),
        safe_div(pl.col("fut_close") - pl.col("fut_open"), pl.col("fut_open")).alias("oc_ratio"),
        ((pl.col("fut_high") + pl.col("fut_low") + pl.col("fut_close")) / 3.0).alias("typical_price"),
        ((pl.col("fut_high") + pl.col("fut_low") + 2.0 * pl.col("fut_close")) / 4.0).alias("weighted_close"),
        ((pl.col("fut_high") + pl.col("fut_low")) / 2.0).alias("median_price"),
    ])

    # Basis / cross-market
    df = df.with_columns([
        safe_div(pl.col("fut_close") - pl.col("spot_close"), pl.col("spot_close")).alias("spot_futures_basis"),
        (pl.col("spot_log_return") - pl.col("futures_log_return")).alias("spot_lead_indicator"),
        (pl.col("fut_close") / pl.col("spot_close")).alias("relative_strength"),
        (pl.col("fut_close") - pl.col("spot_close")).abs().alias("price_convergence"),
    ])

    # Volume / imbalance
    df = df.with_columns([
        (pl.col("fut_buy_volume") + pl.col("fut_sell_volume")).alias("total_volume"),
        safe_div(pl.col("fut_buy_volume"), pl.col("fut_sell_volume")).alias("buy_sell_ratio"),
        safe_div(pl.col("fut_buy_volume") - pl.col("fut_sell_volume"), pl.col("fut_buy_volume") + pl.col("fut_sell_volume")).alias("volume_imbalance"),
        (pl.col("fut_buy_volume") - pl.col("fut_sell_volume")).alias("net_volume"),
        safe_div(pl.col("fut_volume"), pl.col("fut_trade_count")).alias("avg_trade_size"),
        safe_div(pl.col("spot_volume"), pl.col("fut_volume")).alias("spot_futures_vol_ratio"),
    ])

    # Order book depth levels (from bookDepth pivot)
    # We expect depth_-2 and depth_2 etc; compute imbalances if present
    for pct in [2, 3, 5]:
        bid = f"depth_-{pct}"
        ask = f"depth_{pct}"
        if bid in df.columns and ask in df.columns:
            df = df.with_columns([
                safe_div(pl.col(bid) - pl.col(ask), pl.col(bid) + pl.col(ask)).alias(f"depth_imbalance_{pct}pct"),
                safe_div(pl.col(bid), pl.col(bid) + pl.col(ask)).alias(f"bid_pressure_{pct}pct"),
                safe_div(pl.col(ask), pl.col(bid) + pl.col(ask)).alias(f"ask_pressure_{pct}pct"),
            ])

    # Rolling stats (vectorized)
    df = df.with_columns([
        pl.col("futures_log_return").rolling_mean(10).alias("log_return_ma_10"),
        pl.col("futures_log_return").rolling_mean(30).alias("log_return_ma_30"),
        pl.col("futures_log_return").rolling_mean(60).alias("log_return_ma_60"),
        pl.col("futures_log_return").rolling_mean(120).alias("log_return_ma_120"),

        pl.col("futures_log_return").rolling_std(10).alias("return_std_10"),
        pl.col("futures_log_return").rolling_std(30).alias("return_std_30"),
        pl.col("futures_log_return").rolling_std(60).alias("return_std_60"),
        pl.col("futures_log_return").rolling_std(120).alias("return_std_120"),

        # Realized vol (sqrt(sum(r^2)))
        (pl.col("futures_log_return").pow(2).rolling_sum(10).sqrt()).alias("realized_vol_10"),
        (pl.col("futures_log_return").pow(2).rolling_sum(30).sqrt()).alias("realized_vol_30"),
        (pl.col("futures_log_return").pow(2).rolling_sum(60).sqrt()).alias("realized_vol_60"),
        (pl.col("futures_log_return").pow(2).rolling_sum(120).sqrt()).alias("realized_vol_120"),

        # Volume MA/std/zscore
        pl.col("fut_volume").rolling_mean(6).alias("volume_ma_6"),
        pl.col("fut_volume").rolling_mean(30).alias("volume_ma_30"),
        pl.col("fut_volume").rolling_mean(60).alias("volume_ma_60"),
        pl.col("fut_volume").rolling_std(30).alias("volume_std_30"),
        pl.col("fut_volume").rolling_std(60).alias("volume_std_60"),
        safe_div(pl.col("fut_volume") - pl.col("fut_volume").rolling_mean(30), pl.col("fut_volume").rolling_std(30)).alias("volume_zscore_30"),

        # Basis MA/velocity
        pl.col("spot_futures_basis").rolling_mean(6).alias("basis_ma_6"),
        (pl.col("spot_futures_basis") - pl.col("spot_futures_basis").shift(6)).alias("basis_velocity"),

        # VWAP distance
        safe_div(pl.col("fut_close") - pl.col("fut_vwap"), pl.col("fut_vwap")).alias("fut_vwap_distance"),
        safe_div(pl.col("spot_close") - pl.col("spot_vwap"), pl.col("spot_vwap")).alias("spot_vwap_distance"),
    ])

    # Futures-specific joins (mark, premium, funding, metrics)
    if "mark_close" in df.columns:
        df = df.with_columns([
            safe_div(pl.col("mark_close") - pl.col("fut_close"), pl.col("fut_close")).alias("mark_last_divergence"),
        ])
    if "premium_close" in df.columns:
        df = df.with_columns([
            pl.col("premium_close").alias("premium_index"),
            pl.col("premium_close").rolling_mean(6).alias("premium_ma_6"),
            pl.col("premium_close").rolling_mean(60).alias("premium_ma_60"),
            pl.col("premium_close").rolling_std(30).alias("premium_std_30"),
        ])

    # Metrics-derived (forward-filled)
    if "sum_open_interest" in df.columns:
        df = df.with_columns([
            pl.col("sum_open_interest").alias("open_interest"),
            pl.col("sum_open_interest_value").alias("open_interest_value"),
            (pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(1)).alias("oi_change"),
            safe_div(pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(1), pl.col("sum_open_interest").shift(1)).alias("oi_change_pct"),
        ])

    # Cast down to reduce memory
    float_cols = [c for c, t in df.schema.items() if t in (pl.Float64, pl.Float32)]
    df = df.with_columns([pl.col(float_cols).cast(pl.Float32)])

    return df


# =========================
# Cell 8: Build the unified 10s table
# =========================


def build_features(cfg: Inputs) -> tuple[pl.DataFrame, str]:
    plan = build_download_plan(cfg)
    download_many(plan, max_workers=6)

    # Extract each zip into RAW_DIR; keep first CSV found
    extracted_files: dict[str, str] = {}
    for url, zpath in plan:
        files = extract_zip(zpath, RAW_DIR)
        # Choose the largest file as the primary CSV (some zips include README)
        files = [f for f in files if os.path.isfile(f)]
        files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        if not files:
            raise RuntimeError(f"No files extracted from {zpath}")
        extracted_files[os.path.basename(zpath)] = files[0]

    # Map extracted paths
    um_trades_csv = extracted_files[f"um_trades_{cfg.symbol_um}_{cfg.futures_trades_date}.zip"]
    um_agg_csv = extracted_files[f"um_aggtrades_{cfg.symbol_um}_{cfg.futures_aggtrades_date}.zip"]
    um_book_csv = extracted_files[f"um_bookdepth_{cfg.symbol_um}_{cfg.futures_bookdepth_date}.zip"]
    um_klines_csv = extracted_files[f"um_klines1m_{cfg.symbol_um}_{cfg.futures_klines_1m_date}.zip"]
    um_mark_csv = extracted_files[f"um_mark1m_{cfg.symbol_um}_{cfg.futures_mark_1m_date}.zip"]
    um_premium_csv = extracted_files[f"um_premium1m_{cfg.symbol_um}_{cfg.futures_premium_1m_date}.zip"]
    um_metrics_csv = extracted_files[f"um_metrics_{cfg.symbol_um}_{cfg.futures_metrics_date}.zip"]
    um_funding_csv = extracted_files[f"um_funding_{cfg.symbol_um}_{cfg.futures_funding_month}.zip"]

    spot_1s_csv = extracted_files[f"spot_klines1s_{cfg.symbol_spot}_{cfg.spot_klines_1s_date}.zip"]
    spot_agg_csv = extracted_files[f"spot_aggtrades_{cfg.symbol_spot}_{cfg.spot_aggtrades_date}.zip"]
    bvol_csv = extracted_files[f"bvol_{cfg.symbol_bvol}_{cfg.bvol_date}.zip"]

    # Base 10s bars from futures trades
    fut_10s = load_um_trades_10s(um_trades_csv, bar_seconds=cfg.bar_seconds)
    agg_10s = load_um_aggtrades_10s(um_agg_csv, bar_seconds=cfg.bar_seconds)
    spot_10s = load_spot_klines_1s_to_10s(spot_1s_csv, bar_seconds=cfg.bar_seconds)
    spot_agg_10s = load_spot_aggtrades_10s(spot_agg_csv, bar_seconds=cfg.bar_seconds)

    # BookDepth pivot (not necessarily 10s; will be asof-joined)
    book_pivot = load_bookdepth_pivot(um_book_csv)

    # 1m as-of frames
    kl_1m = load_1m_kline_asof(um_klines_csv, prefix="kline")
    mark_1m = load_1m_kline_asof(um_mark_csv, prefix="mark")
    prem_1m = load_1m_kline_asof(um_premium_csv, prefix="premium")

    metrics_5m = load_metrics_5m_asof(um_metrics_csv)
    funding = load_funding_monthly_asof(um_funding_csv)
    bvol = load_bvol_asof(bvol_csv)

    # Create a 10s grid that covers futures trades
    start_ts = fut_10s.select(pl.col("ts").min()).item()
    end_ts = fut_10s.select(pl.col("ts").max()).item() + dt.timedelta(seconds=cfg.bar_seconds)

    grid = pl.DataFrame({
        "ts": pl.datetime_range(start=start_ts, end=end_ts, interval=f"{cfg.bar_seconds}s", eager=True, time_unit="ns")
    })

    # Join everything onto the grid
    out = grid.join(fut_10s, on="ts", how="left")
    out = out.join(agg_10s, on="ts", how="left")
    out = out.join(spot_10s, on="ts", how="left")
    out = out.join(spot_agg_10s, on="ts", how="left")

    # As-of joins (backward)
    out = out.join_asof(book_pivot, on="ts", strategy="backward")
    out = out.join_asof(kl_1m, on="ts", strategy="backward")
    out = out.join_asof(mark_1m, on="ts", strategy="backward")
    out = out.join_asof(prem_1m, on="ts", strategy="backward")
    out = out.join_asof(metrics_5m, on="ts", strategy="backward")
    out = out.join_asof(funding, on="ts", strategy="backward")
    out = out.join_asof(bvol, on="ts", strategy="backward")

    # Fill small gaps for trade-based bars (if no trades, keep null close; you may choose forward-fill)
    out = out.sort("ts")

    # Forward fill key stateful columns
    ffill_cols = [
        "fut_close", "fut_open", "fut_high", "fut_low", "fut_volume", "fut_quote_volume",
        "spot_close", "spot_open", "spot_high", "spot_low", "spot_volume", "spot_quote_volume",
        "funding_rate",
        "bvol_index",
        "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    ]
    ffill_cols = [c for c in ffill_cols if c in out.columns]
    if ffill_cols:
        out = out.with_columns([pl.col(ffill_cols).fill_null(strategy="forward")])

    # Add features
    out = add_time_features(out)
    out = add_core_features(out)

    # Final cleanup
    out = out.with_columns([
        pl.lit(cfg.symbol_um).alias("symbol"),
    ])

    out_file = os.path.join(OUT_DIR, f"features_10s_{cfg.symbol_um}_{cfg.futures_trades_date}.parquet")
    out.write_parquet(out_file, compression="snappy")

    return out, out_file


# =========================
# Cell 9: Upload to Hugging Face
# =========================


def upload_to_hf(cfg: Inputs, local_parquet_path: str) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. In Colab: set os.environ['HF_TOKEN'] or use Secrets.")

    api = HfApi(token=token)

    # Put parquet into a stable path inside repo
    filename = os.path.basename(local_parquet_path)
    path_in_repo = f"{cfg.hf_repo_path_in_repo}/{filename}".strip("/")

    api.upload_file(
        path_or_fileobj=local_parquet_path,
        path_in_repo=path_in_repo,
        repo_id=cfg.hf_repo_id,
        repo_type="dataset",
    )

    return f"https://huggingface.co/datasets/{cfg.hf_repo_id}/blob/main/{path_in_repo}"


# =========================
# Cell 10: Run
# =========================

if __name__ == "__main__":
    df, parquet_path = build_features(CFG)
    print(df.head(5))
    print(f"Wrote: {parquet_path}")

    # Upload (requires HF_TOKEN)
    url = upload_to_hf(CFG, parquet_path)
    print(f"Uploaded to: {url}")
