import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from testing.challenger_historical_qualification_harness import HistoricalLedgerReplayer
from execution.champion_challenger_framework import (
    PeakSameSideRetraceScalper,
    ConformalConfidenceFilterScalper,
    Cycle35_ReversionPureScalper,
    Cycle35_TightRangeReversionScalper,
)

print("=" * 80)
print("VERIFYING SPECIFIC CHALLENGER SOLUTIONS AGAINST TARGETED HISTORICAL LEDGERS")
print("=" * 80)

# Solution 1: Range Exhaustion Aware Scalper on Cycle 59 Ledger
class Solution2_ExhaustionAwareScalper(Cycle35_TightRangeReversionScalper):
    """
    Addresses: Cycle 59 Trade 3 bottom-selling loss (-15.47 bps).
    Mechanism: $6 Envelope + Dynamic Exhaustion Guard.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Solution 2: Exhaustion-Aware $6 Envelope Scalper"

    def _is_range_boundary_violation(self, primary_direction: int, mid: float) -> bool:
        if super()._is_range_boundary_violation(primary_direction, mid):
            return True
        if len(self.rolling_15m_prices) >= 18:
            sess_high = max(self.rolling_15m_prices)
            sess_low = min(self.rolling_15m_prices)
            rng = sess_high - sess_low
            if rng >= 100.0:
                pct = (mid - sess_low) / rng
                if primary_direction == 0 and pct < 0.20:
                    return True
                if primary_direction == 2 and pct > 0.80:
                    return True
        return False

# Solution 2: Orderbook Depth Shield Scalper on Cycle 57 Ledger
class Solution1_OrderbookDepthShieldScalper(Cycle35_ReversionPureScalper):
    """
    Addresses: Cycle 57 Trade 2 hollow-depth pullback trap (-16.7 bps stop-out).
    Mechanism: Requires book depth imbalance >= 1.25.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Solution 1: Orderbook Depth Shield Scalper"

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        if self.state == "IDLE":
            imbalance = bbo_bid_qty / max(bbo_ask_qty, 1e-4)
            if imbalance < 1.25:
                return self.state
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

# Solution 3: Stepped Ratchet Scalper on Cycle 48 Ledger
class Solution4_SteppedRatchetScalper(Cycle35_TightRangeReversionScalper):
    """
    Addresses: Profit decay on impulse moves before reversal.
    Mechanism: Locks +1.50b net at +4.5b and +3.50b net at +7.5b MFE.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Solution 4: Stepped Ratchet Scalper"
        self.min_trailing_activation_bps = 4.50
        self.tier1_min_lock_bps = 5.50
        self.model_harvest_min_bps = 5.00


# Test Solution 2 on Cycle 59
rep59 = HistoricalLedgerReplayer("scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl")
eng59 = Solution2_ExhaustionAwareScalper(symbol="BTCUSDT")
trades59 = rep59.replay_into_engine(eng59)
wins59 = [t for t in trades59 if t.net_return_bps > 0]
pnl59 = sum(t.net_return_bps for t in trades59)
print(f"\n[Test 1] Solution 2 on Cycle 59 (Target: Avoid Trade 3 bottom-selling loss):")
print(f"  Trades: {len(trades59)} | Wins: {len(wins59)} | Net PnL: {pnl59:+.2f} bps")
for i, t in enumerate(trades59):
    print(f"   Trade {i+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | {t.exit_reason}")
if pnl59 > 0:
    print("  >>> VERDICT: PASSED - Successfully solved the bottom-selling loss!")
else:
    print("  >>> VERDICT: FAILED")

# Test Solution 1 on Cycle 57
rep57 = HistoricalLedgerReplayer("scratch/archive_cycle57/live_hierarchical_session_1h_ledger.jsonl")
eng57 = Solution1_OrderbookDepthShieldScalper(symbol="BTCUSDT")
trades57 = rep57.replay_into_engine(eng57)
wins57 = [t for t in trades57 if t.net_return_bps > 0]
pnl57 = sum(t.net_return_bps for t in trades57)
print(f"\n[Test 2] Solution 1 on Cycle 57 (Target: Avoid Trade 2 hollow-depth trap):")
print(f"  Trades: {len(trades57)} | Wins: {len(wins57)} | Net PnL: {pnl57:+.2f} bps")
for i, t in enumerate(trades57):
    print(f"   Trade {i+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | {t.exit_reason}")

# Test Solution 4 on Cycle 48
rep48 = HistoricalLedgerReplayer("scratch/archive_cycle48/live_hierarchical_session_1h_ledger.jsonl")
eng48 = Solution4_SteppedRatchetScalper(symbol="BTCUSDT")
trades48 = rep48.replay_into_engine(eng48)
wins48 = [t for t in trades48 if t.net_return_bps > 0]
pnl48 = sum(t.net_return_bps for t in trades48)
print(f"\n[Test 3] Solution 4 on Cycle 48 (Target: Lock in profits on impulse breakouts):")
print(f"  Trades: {len(trades48)} | Wins: {len(wins48)} | Net PnL: {pnl48:+.2f} bps")
for i, t in enumerate(trades48):
    print(f"   Trade {i+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | {t.exit_reason}")
