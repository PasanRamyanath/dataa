"""Shared low-level helpers: epoch normalization, safe arithmetic, downcast,
checksum verification. Kept dependency-free (polars + stdlib only)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from schemas import detect_timestamp_unit, timestamp_unit_multiplier_to_ns


def normalize_epoch_column_to_ns(df: pl.DataFrame, col: str, out_col: str | None = None) -> pl.DataFrame:
    """Convert an integer epoch column of unknown unit (s/ms/us/ns) to int64 ns.

    Unit is detected per-file from the magnitude of the first non-null value —
    never assumed from the datatype name, since Binance spot silently switched
    epoch units (ms -> us) partway through its history.
    """
    out_col = out_col or col
    sample = df.select(pl.col(col).drop_nulls().first()).item()
    if sample is None:
        return df.with_columns(pl.col(col).cast(pl.Int64).alias(out_col))
    unit = detect_timestamp_unit(sample)
    mult = timestamp_unit_multiplier_to_ns(unit)
    return df.with_columns((pl.col(col).cast(pl.Int64) * mult).alias(out_col))


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Division that returns null (never inf/NaN) when the denominator is ~0."""
    return pl.when(den.abs() > 1e-12).then(num / den).otherwise(None)


def downcast_floats_to_f32(df: pl.DataFrame) -> pl.DataFrame:
    float_cols = [c for c, t in df.schema.items() if t in (pl.Float64,)]
    if not float_cols:
        return df
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])


def clean_inf_nan(df: pl.DataFrame, cols: list[str] | None = None) -> pl.DataFrame:
    """Replace +-inf and NaN with null across float columns. Does NOT fill null with 0 —
    that decision belongs to the caller (imputation strategy is feature-specific
    and must never leak future information via bfill)."""
    cols = cols or [c for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
    exprs = [
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in cols
    ]
    return df.with_columns(exprs)


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(zip_path: Path) -> bool | None:
    """Verify a Binance Vision .zip against its sibling .CHECKSUM file.

    Returns True/False if a checksum file exists, or None if there is none
    to check against (checksum files are not published for every endpoint).
    """
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".CHECKSUM")
    if not checksum_path.exists():
        return None
    expected = checksum_path.read_text().strip().split()[0].lower()
    actual = sha256_of_file(zip_path).lower()
    return expected == actual


def is_zero_byte(path: Path) -> bool:
    return path.exists() and path.stat().st_size == 0
