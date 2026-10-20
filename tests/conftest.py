import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"

for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
