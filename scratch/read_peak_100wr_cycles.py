import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open("D:/PrivateProjects/BTC Futures/changelog.md", "r", encoding="utf-8") as f:
    text = f.read()

import re
for c in [31, 35, 43]:
    m = re.search(rf"## Cycle {c}\b.*?(?=## Cycle \d+|\Z)", text, re.DOTALL)
    if m:
        print("="*100)
        print(f"CYCLE {c} CHANGELOG ENTRY:")
        print("="*100)
        print(m.group(0)[:3000])
