"""
feature engineering scripts/inference/download_models_from_hf.py
Utility script to cleanly download all 16 Project-IMPJR v4.2 retrained models
from Hugging Face repositories into local checkpoints.
"""
import os
import shutil
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = (Path(__file__).resolve().parent.parent / "models_v3" / "checkpoints").resolve()
TOKEN = os.environ.get("HF_TOKEN")

REPOS = {
    "tier4_macro": "p-j-r-1-2-3/Project-IMPJR-v4-Models-tier4_macro",
    "tier3_meso": "p-j-r-1-2-3/Project-IMPJR-v4-Models-tier3_meso",
    "tier2_tactical": "p-j-r-1-2-3/Project-IMPJR-v4-Models-tier2_tactical",
    "tier1_micro": "p-j-r-1-2-3/Project-IMPJR-v4-Models-tier1_micro",
    "stage5_fusion": "p-j-r-1-2-3/Project-IMPJR-v4-Models-fusion",
}


def download_all_models(force_clean: bool = True):
    print(f"--- 1. Target Directory: {BASE_DIR} ---")
    if force_clean and BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
        print("Cleared existing checkpoints directory.")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- 2. Downloading models from Hugging Face Hub ---")
    for folder, repo_id in REPOS.items():
        dest = BASE_DIR / folder
        dest.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"Downloading {repo_id} -> {folder}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            token=TOKEN,
            ignore_patterns=["*training_data_snapshot*"],
        )
        elapsed = time.time() - t0
        size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"  [OK] {folder}: {size_mb:.2f} MB downloaded in {elapsed:.1f}s")

    print("\n--- 3. Verifying all 16 models in LiveMultiHorizonV4ModelRegistry ---")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry

    registry = LiveMultiHorizonV4ModelRegistry()
    summary = registry.summary()

    print(f"Total Loaded Models: {summary['loaded_count']} / 16")
    print(f"Stage 5 Stacker Fitted: {summary.get('stage5_fitted')}")
    for m in summary["loaded_members"]:
        print(f"  [OK] {m}")

    assert summary["loaded_count"] == 16, f"Expected 16 models, got {summary['loaded_count']}"
    assert summary.get("stage5_fitted") is True, "Stage 5 Stacker is not fitted!"
    print("\n>>> ALL 16 MODELS DOWNLOADED & VERIFIED SUCCESSFULLY <<<")


if __name__ == "__main__":
    download_all_models()
