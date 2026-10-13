"""
Regenerates col_schema.json — the empirical output-schema snapshot that
common/feature_contract.py loads at import time.

Run this any time feature_engine/*.py or labeling.py changes the set of
output columns, then re-check the counts asserted in feature_contract.py.

Usage:
    python scripts/dump_schema.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_dataset import build_day  # noqa: E402


def main() -> None:
    day = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 9, 6)
    df = build_day(day, conform=False)  # raw pipeline output, not the current contract
    if df is None:
        raise SystemExit(f"build_day({day}) returned None — pick a day with local raw data available")

    schema = {name: str(dtype) for name, dtype in df.schema.items()}
    out_path = Path(__file__).resolve().parent.parent / "col_schema.json"
    out_path.write_text(json.dumps(schema, indent=1), encoding="utf-8")
    print(f"Wrote {len(schema)} columns to {out_path}")


if __name__ == "__main__":
    main()
