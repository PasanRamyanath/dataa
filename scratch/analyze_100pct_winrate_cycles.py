import sys
import re
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

changelog_path = Path("D:/PrivateProjects/BTC Futures/changelog.md")
with open(changelog_path, "r", encoding="utf-8") as f:
    text = f.read()

# Parse all completed cycles
cycle_sections = re.findall(r"(## Cycle (\d+).*?)(?=## Cycle \d+|$)", text, re.DOTALL)

records = []
for sec_text, c_num_str in cycle_sections:
    c_num = int(c_num_str)
    
    # Extract trades
    trades_m = re.search(r"(\d+)\s*trades?", sec_text, re.IGNORECASE)
    trades = int(trades_m.group(1)) if trades_m else 0
    
    # Extract win rate
    wr_m = re.search(r"Win Rate:?\s*\*?\*?(\d+(?:\.\d+)?)%", sec_text, re.IGNORECASE)
    if not wr_m:
        wr_m = re.search(r"(\d+(?:\.\d+)?)%\s*win rate", sec_text, re.IGNORECASE)
    wr = float(wr_m.group(1)) if wr_m else 0.0
    
    # Extract net pnl
    pnl_m = re.search(r"Net Realized Cash Profit:?\s*\*?\*?`?([+-]?\d+(?:\.\d+)?)\s*bps", sec_text, re.IGNORECASE)
    if not pnl_m:
        pnl_m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*bps\s*Net", sec_text, re.IGNORECASE)
    if not pnl_m:
        pnl_m = re.search(r"net_pnl_bps\s*=\s*([+-]?\d+(?:\.\d+)?)", sec_text, re.IGNORECASE)
    net_pnl = float(pnl_m.group(1)) if pnl_m else 0.0
    
    # Extract winner/champion name
    winner_m = re.search(r"Tournament Winner:\s*\*?\*?(.*?)(?:\n|\*)", sec_text)
    if not winner_m:
        winner_m = re.search(r"NEW CHAMPION CROWNED:\s*\*?\*?`?(.*?)`?\*?\*?", sec_text)
    winner = winner_m.group(1).strip() if winner_m else "N/A"
    
    # Extract accuracies
    macro_m = re.search(r"15m Macro:?\s*\*?\*?`?(\d+(?:\.\d+)?)%", sec_text)
    meso_m = re.search(r"5m Meso:?\s*\*?\*?`?(\d+(?:\.\d+)?)%", sec_text)
    tact_m = re.search(r"1m Tactical:?\s*\*?\*?`?(\d+(?:\.\d+)?)%", sec_text)
    micro_m = re.search(r"10s Micro:?\s*\*?\*?`?(\d+(?:\.\d+)?)%", sec_text)
    
    macro_acc = float(macro_m.group(1)) if macro_m else 0.0
    meso_acc = float(meso_m.group(1)) if meso_m else 0.0
    tact_acc = float(tact_m.group(1)) if tact_m else 0.0
    micro_acc = float(micro_m.group(1)) if micro_m else 0.0
    
    records.append({
        "cycle": c_num,
        "trades": trades,
        "win_rate": wr,
        "net_pnl": net_pnl,
        "winner": winner,
        "macro_acc": macro_acc,
        "meso_acc": meso_acc,
        "tact_acc": tact_acc,
        "micro_acc": micro_acc,
        "raw_text": sec_text[:1000]
    })

print("="*110)
print("ALL 100% WIN RATE CYCLES (WITH > 0 TRADES)")
print("="*110)
win100 = [r for r in records if r["win_rate"] == 100.0 and r["trades"] > 0]

# Sort by trades descending, then net pnl descending
win100_sorted = sorted(win100, key=lambda x: (x["trades"], x["net_pnl"]), reverse=True)

for r in win100_sorted:
    avg_trade = r["net_pnl"] / r["trades"] if r["trades"] > 0 else 0.0
    print(f"Cycle {r['cycle']:2d} | Trades: {r['trades']:2d} | Net PnL: {r['net_pnl']:+7.2f} bps | Avg/Trade: {avg_trade:+6.2f} bps | Acc: 15m={r['macro_acc']:.1f}% 5m={r['meso_acc']:.1f}% 1m={r['tact_acc']:.1f}% 10s={r['micro_acc']:.1f}%")
    print(f"   Winner: {r['winner']}")

print("\n" + "="*110)
print("TOP CYCLES BY NET PNL (WITH WIN RATE >= 80% AND >= 2 TRADES)")
print("="*110)
high_wr = [r for r in records if r["win_rate"] >= 80.0 and r["trades"] >= 2]
high_wr_sorted = sorted(high_wr, key=lambda x: x["net_pnl"], reverse=True)
for r in high_wr_sorted:
    avg_trade = r["net_pnl"] / r["trades"] if r["trades"] > 0 else 0.0
    print(f"Cycle {r['cycle']:2d} | WR: {r['win_rate']:5.1f}% | Trades: {r['trades']:2d} | Net PnL: {r['net_pnl']:+7.2f} bps | Avg/Trade: {avg_trade:+6.2f} bps | Acc: 15m={r['macro_acc']:.1f}% 5m={r['meso_acc']:.1f}% 10s={r['micro_acc']:.1f}%")
    print(f"   Winner: {r['winner']}")
