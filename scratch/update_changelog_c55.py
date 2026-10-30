fpath = "changelog.md"
with open(fpath, "r", encoding="utf-8") as f:
    text = f.read()

# Add Cycle 55 to the summary table
old_table_end = """| **Cycle 54** | 60.07 min | 360 | 0 | **N/A** | **`+0.00 bps`** | **1.00** | 👑 **REIGNING CHAMPION DEFENDS TITLE VIA CAPITAL PRESERVATION; FORENSIC CHOKE BUG ISOLATED: `DualTierSteppedRatchetScalper` retains crown with 0 trades, 0 losses, and `+0.00 bps` net draw!** Subdued $202 range ($80,033 - $80,235). Code-level audit isolated root cause of zero trade frequency in C50-C54: unintended `meso_dir_max >= 0.45` over-correction. All-time peak forward accuracies: **97.04% 15m Macro (92.45% Active Precision), 95.76% 5m Meso (88.98% Active Precision, 89.63% Meso tier), 87.47% 10s Micro**. Absolute Gate enforced. **Cumulative Peak Realized Net Cash Profit 100% preserved at `+109.63 bps Net`**! |"""

new_table_end = """| **Cycle 54** | 60.07 min | 360 | 0 | **N/A** | **`+0.00 bps`** | **1.00** | 👑 **REIGNING CHAMPION DEFENDS TITLE VIA CAPITAL PRESERVATION; FORENSIC CHOKE BUG ISOLATED: `DualTierSteppedRatchetScalper` retains crown with 0 trades, 0 losses, and `+0.00 bps` net draw!** Subdued $202 range ($80,033 - $80,235). Code-level audit isolated root cause of zero trade frequency in C50-C54: unintended `meso_dir_max >= 0.45` over-correction. All-time peak forward accuracies: **97.04% 15m Macro (92.45% Active Precision), 95.76% 5m Meso (88.98% Active Precision, 89.63% Meso tier), 87.47% 10s Micro**. Absolute Gate enforced. **Cumulative Peak Realized Net Cash Profit 100% preserved at `+109.63 bps Net`**! |
| **Cycle 55** | 36.00 min | 216 | 2 | **0.0%** | `-21.51 bps` | 0.00 | 👑 **ABSOLUTE PROFIT GATE PROTECTS REIGNING CROWN & TRIGGERS PURE LONG-ONLY REVERSION:** Interrupted at T+36m per user command. All engines attempting short-selling suffered losses by selling local bottoms ($80,019 and $80,010) directly into the major $80,000 support floor before a +$173 upward bounce. Challenger 5 achieved Rank 1 (0 trades, 100% capital preserved). Absolute Profitability Gate permanently rejected all negative mutants; **Cumulative Peak Realized Net Cash Profit 100% defended and preserved at `+109.63 bps Net`**! Replay audit uncovered **Bar 50 Peak Discovery** (+14.33 bps net run missed due to flat prior split), isolating the exact unchoked directional spread formula for Cycle 56. |"""

if old_table_end in text:
    text = text.replace(old_table_end, new_table_end)
    print("Added Cycle 55 to changelog summary table successfully!")
else:
    print("Warning: old_table_end not found!")

with open(fpath, "w", encoding="utf-8") as f:
    f.write(text)
print("Changelog update complete!")
