import re
import os
import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open("changelog.md", "r", encoding="utf-8") as f:
    text = f.read()

# Pattern to find all cycle table rows
# | **Cycle X** | ...
pattern = r"\|\s*\*\*Cycle\s+(\d+)\*\*\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|\n]+)"

matches = re.findall(pattern, text)
print(f"Total cycle rows found in master table: {len(matches)}")

cycles = []
for m in matches:
    c_num = int(m[0])
    dur = m[1].strip()
    bars = m[2].strip()
    trades = m[3].strip()
    wr = m[4].replace("**", "").replace("`", "").strip()
    net = m[5].replace("**", "").replace("`", "").strip()
    pf = m[6].replace("**", "").replace("`", "").strip()
    summary = m[7].replace("**", "").replace("`", "").strip()[:80]
    cycles.append({
        "cycle": c_num,
        "dur": dur,
        "bars": bars,
        "trades": trades,
        "wr": wr,
        "net": net,
        "pf": pf,
        "summary": summary
    })

# Also parse all ### Cycle X sections for any cycles not in master table
section_pattern = r"### Cycle\s+(\d+)\s*\((.*?)\).*?- \*\*Results:\*\*.*?- Net Realized PnL:\s*`?([+\-\d\.]+)\s*bps`?"

print("\n--- ALL HISTORICAL CYCLES CHRONOLOGY ---")
for c in sorted(cycles, key=lambda x: x["cycle"]):
    print(f"Cycle {c['cycle']:<2} | Tr: {c['trades']:<3} | WR: {c['wr']:<8} | Net: {c['net']:<12} | PF: {c['pf']:<7} | {c['summary']}")
