import re
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

changelog_path = "changelog.md"
with open(changelog_path, "r", encoding="utf-8") as f:
    text = f.read()

# Pattern for table rows
row_pattern = re.compile(r'\|\s*\*\*Cycle\s+(\d+)\*\*\s*\|\s*([^\|]+)\|\s*([^\|]+)\|\s*([^\|]+)\|\s*([^\|]+)\|\s*([^\|]+)\|\s*([^\|]+)\|\s*([^\|]+)\|')

matches = row_pattern.findall(text)
print(f"Extracted {len(matches)} cycle summary rows.")

cycles = []
for m in matches:
    c_num = int(m[0])
    dur = m[1].strip()
    bars = m[2].strip()
    trades_str = m[3].strip()
    wr_str = m[4].strip()
    raw_net = m[5].strip().replace("`", "").replace("*", "").replace("bps", "").replace("+", "").strip()
    pf_str = m[6].strip()
    desc = m[7].strip()
    try:
        trades = int(re.search(r'\d+', trades_str).group())
    except:
        trades = 0
    try:
        net = float(raw_net)
    except:
        net = 0.0
    cycles.append({
        "cycle": c_num,
        "trades": trades,
        "wr": wr_str,
        "net": net,
        "pf": pf_str,
        "desc": desc
    })

print("\n" + "="*80)
print("ALL CYCLES PERFORMANCE SUMMARY")
print("="*80)
for c in cycles:
    print(f"Cycle {c['cycle']:2d}: Trades={c['trades']:2d}, WR={c['wr']:>6}, Net={c['net']:+7.2f} bps | {c['desc'][:65]}...")

# Group by performance
profitable = [c for c in cycles if c['net'] > 0]
flat = [c for c in cycles if c['net'] == 0]
losing = [c for c in cycles if c['net'] < 0]

print("\n" + "="*80)
print(f"TOTAL: {len(cycles)} cycles. Profitable: {len(profitable)}, Flat (0 PnL): {len(flat)}, Losing: {len(losing)}")
print("="*80)

print("\nTOP 10 MOST PROFITABLE CYCLES:")
for c in sorted(profitable, key=lambda x: x['net'], reverse=True)[:10]:
    print(f"  Cycle {c['cycle']:2d}: Net = {c['net']:+6.2f} bps | Trades = {c['trades']} | WR = {c['wr']} | {c['desc'][:80]}")

print("\nWORST 5 LOSING CYCLES:")
for c in sorted(losing, key=lambda x: x['net'])[:5]:
    print(f"  Cycle {c['cycle']:2d}: Net = {c['net']:+6.2f} bps | Trades = {c['trades']} | WR = {c['wr']} | {c['desc'][:80]}")
