import re
import os

with open("changelog.md", "r", encoding="utf-8") as f:
    text = f.read()

# Parse the master summary table at the top of changelog.md
table_match = re.search(r"## Master Tournament Performance Summary.*?\n\| :---.*?\n((?:\| \*\*Cycle \d+\*\*.*?\n)+)", text, re.DOTALL)

if table_match:
    rows = table_match.group(1).strip().split("\n")
    print(f"Found {len(rows)} cycle rows in Master Summary Table:")
    print(f"{'Cycle':<12} | {'Duration':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Net PnL':<15} | {'Profit Factor':<15}")
    print("-" * 80)
    for r in rows:
        cols = [c.strip() for c in r.split("|")[1:-1]]
        if len(cols) >= 6:
            cycle = cols[0].replace("**", "")
            dur = cols[1]
            trades = cols[3]
            wr = cols[4].replace("**", "")
            net = cols[5].replace("**", "").replace("`", "")
            pf = cols[6].replace("**", "").replace("`", "")
            print(f"{cycle:<12} | {dur:<10} | {trades:<8} | {wr:<10} | {net:<15} | {pf:<15}")
else:
    print("Table match not found, parsing sections...")
