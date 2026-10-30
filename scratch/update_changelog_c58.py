import re

with open('changelog.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Add Cycle 58 row to summary table
c57_marker = "| **Cycle 57** | 60.06 min | 359 | 1 | **100.0%** | **`+2.00 bps`** | **`999.00`** | 🏆 **NEW CHAMPION CROWNED & ALL-TIME PROFIT RECORD EXPANDS: `Challenger 5` (`OrderbookLiquidityImbalanceScalper`) wins crown with 100.0% Win Rate (1 Win / 0 Losses) and `+2.00 bps` Net Realized Cash Profit!** Fleet fired Maker BUY orders at $81,728.50 on concordant breakout; Stepped Ratchet locked +2.00 bps net profit floor at +6.96b MFE. Former Champion and Challengers 1-4 suffered -16.7 bps stop-out by re-entering post-win pullback, while Challenger 5's L1/L2 Orderbook Liquidity Imbalance Guard (depth ratio >= 1.25) vetoed the trap entry! Absolute Gate crowns Challenger 5. **Cumulative Peak Realized Net Cash Profit officially expands to `+111.63 bps Net`**! Accuracies: **93.68% Macro, 87.62% Meso, 84.70% Tactical (85.0% Precision), 87.99% Micro**. |"

c58_row = "| **Cycle 58** | 60.07 min | 359 | 1 | 0.0% | `-12.01 bps` | 0.00 | 👑 **ABSOLUTE PROFIT GATE DEFENDS TITLE & ACTIVATES CYCLE 35 REVERSION PROTOCOL:** BTC suffered -$437 breakdown from $81,368 to $80,931. Competitors 1-4 & Champion entered Long at $81,368; Responsive Model Maker Exit capped gross loss to -8.01 bps, saving +35.7 bps drawdown. Challenger 5 (`OrderbookLiquidity_HighConvictionScalper`) achieved Rank 1 (0 trades, 100% capital preserved) via high-conviction imbalance guard (>= 1.35). Absolute Gate preserved Reigning Champion; **Cumulative Peak Realized Net Cash Profit 100% defended at `+111.63 bps Net`**! Standing user directive activated: system initiates Cycle 35 Reversion Protocol. Accuracies: **91.79% Meso, 86.59% Micro, 83.27% Macro, 80.75% Tactical**. |"

if c57_marker in content and c58_row not in content:
    content = content.replace(c57_marker, c57_marker + "\n" + c58_row)
    print("Added Cycle 58 row to summary table.")

c58_detail = """

### Cycle 58 (Champion-Challenger Tournament Session 35: -$437 Cascade Defense & Cycle 35 Reversion Trigger) - COMPLETED
- **Session Window:** 2026-10-29 16:16:18 – 2026-10-29 17:16:29 UTC (21:46:18 – 22:46:29 Local Time) (60.07 min / 359 bars)
- **Tournament Verdict:** 👑 **ABSOLUTE PROFITABILITY GATE DEFENDS TITLE & ACTIVATES CYCLE 35 REVERSION PROTOCOL** (0 challenger qualified with net > 0 & WR >= 75%).
- **Cumulative Peak Realized Net Cash Profit:** **`+111.63 bps Net Cash Profit`** (100% Defended and Preserved).
- **Competitors Breakdown:**
  - 🥇 **1. Challenger 5 (`OrderbookLiquidity_HighConvictionScalper`):** 0 Trades, 0 Losses, **`+0.00 bps Net`**, PF 1.00 -> **100% Capital Preserved** (Rank 1).
  - 👑 **2. Champion (`OrderbookLiquidityImbalanceScalper`):** 1 Trade, 0 Wins / 1 Loss, `-12.01 bps Net` -> Retained Crown under Gate.
  - 3. **Challenger 1 (`OrderbookLiquidity_RetraceScalper`):** 1 Trade, 0 Wins / 1 Loss, `-12.01 bps Net` -> Disqualified by Gate.
  - 4. **Challenger 2 (`OrderbookLiquidity_Tier05RatchetScalper`):** 1 Trade, 0 Wins / 1 Loss, `-12.01 bps Net` -> Disqualified by Gate.
  - 5. **Challenger 3 (`OrderbookLiquidity_AdaptiveSpreadScalper`):** 1 Trade, 0 Wins / 1 Loss, `-12.01 bps Net` -> Disqualified by Gate.
  - 6. **Challenger 4 (`OrderbookLiquidity_TightEnvelopeScalper`):** 1 Trade, 0 Wins / 1 Loss, `-12.01 bps Net` -> Disqualified by Gate.
- **Model Forward Accuracies (359 bars / 60 minutes):**
  - **5m Meso Swing:** **`91.79%`** overall, **`67.80% Active Precision`** (59 samples)
  - **10s Micro Velocity:** **`86.59%`** overall directional accuracy (359 samples)
  - **15m Macro Trend:** **`83.27%`** overall, **`60.00% Active Precision`** (50 samples)
  - **30s Micro-Tactical:** **`81.20%`** overall, **`80.36% Active Precision`** (56 samples)
  - **1m Tactical Wave:** **`80.75%`** overall, **`80.36% Active Precision`** (56 samples)
- **Deep Architectural Findings:**
  1. **High-Conviction Orderbook Guard Defends Capital:**
     Challenger 5 required `imbalance_ratio >= 1.35`. It strictly vetoed the entry at $81,368.80, completely dodging the -$437 cascade and preserving 100% of capital.
  2. **Responsive Model Maker Exit Capped Drawdown:**
     When Meso swung adverse during the cascade, the responsive exit closed at $81,303.60 (-8.01 bps gross), saving +35.7 bps compared to holding into $80,931.
  3. **Activation of Contractual Cycle 35 Reversion Mandate:**
     In accordance with user directive (*"if the trade frequency is improved and the win rate and the accuracy of the trades and predictions starts to fall off, get back to the cycle 35"*), the tournament executes the Cycle 35 Reversion Protocol for Cycle 59.
"""

if "### Cycle 58 (" not in content:
    content += c58_detail
    print("Appended Cycle 58 detail to changelog.")

with open('changelog.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("changelog.md written successfully.")
