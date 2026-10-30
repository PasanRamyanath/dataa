import sys, os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

changelog_path = 'changelog.md'
with open(changelog_path, 'r', encoding='utf-8') as f:
    text = f.read()

c58_marker = "| **Cycle 58** | 60.07 min | 359 | 1 | 0.0% | `-12.01 bps` | 0.00 |"
assert c58_marker in text, "Cycle 58 marker not found!"

c59_row = """| **Cycle 59** | 60.06 min | 360 | 3 | **66.7%** | **`+3.26 bps`** | **`1.21`** | **CYCLE 35 REVERSION PROTOCOL & $6 ENVELOPE PROFIT WIN:** Under Cycle 35 Reversion Protocol, Challenger 3 (`Cycle35_TightRangeReversionScalper ($6 Envelope)`) captured **`+3.26 bps Net Realized Profit`** (PF 1.21, 3 trades: 2 Wins / 1 Loss). Standout Trade 1 captured **`+14.58 bps Net Maker Win`** with zero MAE and $+20.58\text{ bps}$ MFE! Absolute Gate preserved Reigning Champion; **Defended Cumulative Peak Realized Net Cash Profit remains defended at `+111.63 bps Net`**! Accuracies: **93.70% Macro, 89.39% Meso, 84.96% Micro, 81.64% Tactical**. |"""

c60_row = """| **Cycle 60** | 60.06 min | 360 | 5 | **80.0%** | **`+62.20 bps`** | **`4.99`** | **PEAK REALIZED CASH SURGE (+173.83 BPS CUMULATIVE DEFENDED):** Absolute Profit Gate strictly enforced; **Reigning Champion (`Cycle35_ReversionPureScalper`) defended title** with **`+62.20 bps Net Realized Profit`**, **`80.0% Win Rate` (4 Wins / 1 Loss)**, **`4.99` Profit Factor**, and **`+12.44 bps` average profit per trade** across 5 active trades (5.0 trades/hr)! Standout Trade 3 captured **`+44.43 bps Net Maker Profit`** with zero MAE and $+50.43\text{ bps}$ MFE! Peak multi-horizon forward accuracies: **92.28% (5m Meso, 94.3% Active Precision)**, **92.22% (15m Macro, 94.3% Active Precision)**, **91.64% (10s Micro)**. Cumulative peak defended cash profit reached **`+173.83 bps Net Cash Profit`**! |"""

lines = text.splitlines()
new_lines = []
for l in lines:
    new_lines.append(l)
    if c58_marker in l:
        new_lines.append(c59_row)
        new_lines.append(c60_row)

text_with_table = "\n".join(new_lines)

cycle60_section = """

### Cycle 60 (Champion-Challenger Tournament Session 37: 56-Cycle Historical Root-Cause Synthesis & Record Cash Surge) - COMPLETED
- **Session Window:** 2026-10-30 01:50:49 – 2026-10-30 02:50:49 UTC (60.06 min / 360 bars)
- **Deployment Mode:** Live Binance Unified Futures Stream (100% Maker Orders at BBO)
- **Tournament Winner:** **👑 Reigning Champion: Cycle 35 Reversion Pure Scalper**
- **Session Net Realized Cash Profit:** **`+62.20 bps`** (Gross: $+85.20\text{ bps}$, Total Fees: $23.0\text{ bps}$)
- **Profit Factor:** **`4.99`**
- **Win Rate:** **`80.0%` (4 Wins / 1 Loss)**
- **Trade Frequency:** **`5.0 trades/hr` (5 total trades)**
- **Average Profit Per Trade:** **`+12.44 bps`**
- **Defended Cumulative Peak Realized Net Cash Profit:** **`+173.83 bps Net Cash Profit`** (Surged from $+111.63\text{ bps}$)
- **Multi-Horizon Forward Accuracy & Active Precision:**
  - **15m Macro:** **`92.22%`** Overall Directional Accuracy | **`94.31%`** Active Precision (116 / 123 correct directional calls)
  - **5m Meso:** **`92.28%`** Overall Directional Accuracy | **`94.27%`** Active Precision (148 / 157 correct directional calls)
  - **1m Tactical:** **`83.24%`** Overall Accuracy | **`88.00%`** Active Precision (44 / 50 correct calls)
  - **30s Micro:** **`81.87%`** Overall Accuracy | **`88.24%`** Active Precision (45 / 51 correct calls)
  - **10s Micro:** **`91.64%`** Overall Accuracy
- **Champion Trade Log:**
  1. `SELL @ 79,920.8 -> 79,761.2` | Gross: $+19.97\text{b}$ | Fee: $4.0\text{b}$ | **`Net: +15.97 bps`** | MFE: $+20.73\text{b}$ | MAE: $0.00\text{b}$ | Dur: 60s | Exit: `MODEL_HARVEST_MAKER_TP`
  2. `SELL @ 79,800.4 -> 79,696.6` | Gross: $+13.01\text{b}$ | Fee: $4.0\text{b}$ | **`Net: +9.01 bps`** | MFE: $+13.98\text{b}$ | MAE: $-1.22\text{b}$ | Dur: 50s | Exit: `MODEL_HARVEST_MAKER_TP`
  3. `SELL @ 78,756.5 -> 78,375.1` | Gross: $+48.43\text{b}$ | Fee: $4.0\text{b}$ | **`Net: +44.43 bps`** | MFE: $+50.43\text{b}$ | MAE: $0.00\text{b}$ | Dur: 80s | Exit: `MICRO_TRAILING_MAKER_TP` (Monster swing winner!)
  4. `SELL @ 79,027.7 -> 78,929.8` | Gross: $+12.39\text{b}$ | Fee: $4.0\text{b}$ | **`Net: +8.39 bps`** | MFE: $+14.25\text{b}$ | MAE: $-2.82\text{b}$ | Dur: 90s | Exit: `MODEL_HARVEST_MAKER_TP`
  5. `SELL @ 78,812.4 -> 78,880.2` | Gross: $-8.60\text{b}$ | Fee: $7.0\text{b}$ | **`Net: -15.60 bps`** | MFE: $+1.93\text{b}$ | MAE: $-8.60\text{b}$ | Dur: 30s | Exit: `DUAL_HORIZON_INVALIDATION_TAKER`
- **Tournament Verdict:** The Reigning Champion exceeded the Absolute Profitability Gate (`net_pnl > 0.0` and `WR >= 75.0%`). No challenger strictly exceeded the Champion's net PnL, so the **Reigning Champion defended its title**, elevating the protected cumulative peak realized profit to **`+173.83 bps Net Cash Profit`**. Archived in `scratch/archive_cycle60/`.
"""

final_text = text_with_table + cycle60_section

with open(changelog_path, 'w', encoding='utf-8') as f:
    f.write(final_text)

print("changelog.md successfully updated with Cycle 59 and Cycle 60!")
