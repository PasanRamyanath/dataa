import json
import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('feature engineering scripts'))

from execution.champion_challenger_framework import (
    TournamentCompetitorRecord,
    PeakSameSideRetraceScalper,
    ConformalConfidenceFilterScalper,
    Cycle35_ReversionPureScalper,
    Cycle35_TightRangeReversionScalper,
)
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

# Let's define the candidate engines for Cycle 60:

class Challenger3_ExhaustionAware(Cycle35_TightRangeReversionScalper):
    """
    Winning $6 Envelope + 15m Range Exhaustion Guard:
    Prevents shorting into the bottom 15% of a wide (> $200) 15m drop,
    and prevents buying into the top 15% of a wide (> $200) 15m rally.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: Exhaustion-Aware $6 Envelope Scalper"

    def _is_range_boundary_violation(self, primary_direction: int, mid: float) -> bool:
        # Base $6 envelope
        if super()._is_range_boundary_violation(primary_direction, mid):
            return True
        # Exhaustion check: if range > $100, don't enter in the extreme 20% tail
        if len(self.rolling_15m_prices) >= 18:
            sess_high = max(self.rolling_15m_prices)
            sess_low = min(self.rolling_15m_prices)
            rng = sess_high - sess_low
            if rng >= 100.0:
                pct = (mid - sess_low) / rng
                if primary_direction == 0 and pct < 0.20: # Do not short bottom 20%
                    return True
                if primary_direction == 2 and pct > 0.80: # Do not buy top 20%
                    return True
        return False


class Challenger4_ActiveFrequencySniper(Cycle35_TightRangeReversionScalper):
    """
    High-frequency $6 envelope scalper with responsive 8.0 bps net edge trigger.
    Targets 3-6 trades per hour with strict maker execution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Active Frequency $6 Envelope Sniper"
        self.min_net_edge_bps = 3.5


class Challenger5_FastSteppedRatchet(Challenger3_ExhaustionAware):
    """
    Exhaustion-aware $6 envelope + Fast Stepped Ratchet (+1.5b net lock at +4.0b MFE).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Fast Stepped Ratchet Scalper"
        self.tier1_min_lock_bps = 4.0


# Run simulation on Cycle 59 ledger
records = []
with open('scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} bars.")

engines = {
    "Champion": Cycle35_ReversionPureScalper(symbol="BTCUSDT"),
    "C1_C59Winner": Cycle35_TightRangeReversionScalper(symbol="BTCUSDT"),
    "C3_ExhaustionAware": Challenger3_ExhaustionAware(symbol="BTCUSDT"),
    "C4_ActiveFreq": Challenger4_ActiveFrequencySniper(symbol="BTCUSDT"),
    "C5_FastRatchet": Challenger5_FastSteppedRatchet(symbol="BTCUSDT"),
}

for r in records:
    mid = r["bbo_mid"]
    ts_ms = r["ts_ms"]
    bid = r["bbo_bid"]
    ask = r["bbo_ask"]
    
    # Fake decision from ledger fields
    class DummyDecision:
        def __init__(self, d):
            self.ts_ms = d["ts_ms"]
            self.macro_probs = d["macro_probs"]
            self.meso_probs = d["meso_probs"]
            self.tactical_probs = d["tactical_probs"]
            self.micro_probs = d["micro_probs"]
            self.primary_direction = d["primary_direction"]
            self.expected_net_edge_bps = d["expected_net_edge_bps"]
            self.expected_gross_move_bps = d["expected_net_edge_bps"] + 4.0
            self.target_holding_seconds = 300.0
            self.take_profit_bps = 15.0
            self.stop_loss_bps = 25.0
            self.kelly_position_fraction = 0.10
            self.fused_swing_probs = d["meso_probs"]
            self.cross_horizon_alignment = 0.8
            self.meta_act_probability = 0.8
            self.micro_sniper_trigger_prob = 0.8
            self.execution_mode = "SNIPER_MAKER"
            
    dec = DummyDecision(r)
    
    for name, eng in engines.items():
        eng.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.0, bbo_ask_qty=1.0)
        eng.on_hierarchical_decision(
            decision=dec,
            bbo_bid=bid,
            bbo_ask=ask,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            equity_usdt=10000.0,
            intra_bar_open=mid
        )

print("\n--- SIMULATION RESULTS ON CYCLE 59 ---")
for name, eng in engines.items():
    trades = eng.closed_trades
    wins = [t for t in trades if t.net_return_bps > 0]
    wr = len(wins) / len(trades) * 100.0 if trades else 0.0
    net_pnl = sum(t.net_return_bps for t in trades)
    print(f"[{name}] Trades: {len(trades)} | Wins: {len(wins)} | WR: {wr:.1f}% | Net PnL: {net_pnl:+.2f} bps")
    for i, t in enumerate(trades):
        print(f"   Trade {i+1}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | {t.exit_reason}")
