"""
Hugging Face Hub publishing, mirroring the existing project's SAFEST verified
pattern (from colab_tft_feature_extraction.py / tft_feature_extraction_2_(2).py
Config, not the antigravity/ variant that regressed to a hardcoded-token
fallback):

  * token ONLY from the HF_TOKEN environment variable (Kaggle Secrets in
    production) — never hardcoded, never printed, never logged,
  * huggingface_hub.HfApi().upload_file (never push_to_hub/upload_folder),
  * repo_type="dataset", flat `data/` prefix, one parquet file per calendar
    day: `data/features_{symbol}_{date}.parquet`,
  * create_repo(..., exist_ok=True) called once, with a defensive retry
    inside the upload path in case the repo didn't exist yet.

Security note: several files elsewhere in this project (see project audit)
were found to carry a literal HF_TOKEN value in source. This module and the
rest of `feature engineering scripts/` never do that — if HF_TOKEN is unset,
functions here raise rather than silently falling back to a hardcoded value.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi

from config import CACHE_ROOT, HF_DATA_PREFIX, HF_REPO_ID, HF_REPO_TYPE, get_hf_token


def get_api() -> HfApi:
    return HfApi(token=get_hf_token())


def ensure_repo(api: HfApi | None = None, repo_id: str = HF_REPO_ID) -> None:
    api = api or get_api()
    api.create_repo(repo_id=repo_id, repo_type=HF_REPO_TYPE, exist_ok=True)


def upload_daily_features(
    df: pl.DataFrame,
    symbol: str,
    date_str: str,
    api: HfApi | None = None,
    repo_id: str = HF_REPO_ID,
    cache_dir: Path = CACHE_ROOT,
    keep_local: bool = False,
) -> str:
    """Write `df` to a local parquet file and upload it to the dataset repo
    at `data/features_{symbol}_{date_str}.parquet`. Returns the path_in_repo."""
    api = api or get_api()
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"features_{symbol}_{date_str}.parquet"
    local_path = cache_dir / filename
    path_in_repo = f"{HF_DATA_PREFIX}/{filename}"

    df.write_parquet(local_path, compression="snappy")

    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=HF_REPO_TYPE,
            commit_message=f"Add features for {symbol} {date_str}",
        )
    except Exception:
        ensure_repo(api, repo_id)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=HF_REPO_TYPE,
            commit_message=f"Add features for {symbol} {date_str}",
        )
    finally:
        if not keep_local and local_path.exists():
            os.remove(local_path)

    return path_in_repo
