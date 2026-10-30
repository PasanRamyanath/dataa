import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('feature engineering scripts'))

from execution.champion_challenger_framework import (
    PeakSameSideRetraceScalper,
    ConformalConfidenceFilterScalper,
    Cycle35_ReversionPureScalper,
    Cycle35_TightRangeReversionScalper,
)

# Test defining candidate classes
class Cycle60_ExhaustionAwareScalper(Cycle35_TightRangeReversionScalper):
    """
    ⚔️ Challenger 2: Exhaustion-Aware $6 Envelope Scalper
    Winning $6 Envelope + Range Exhaustion Guard:
    Prevents shorting into the bottom 20% of a wide (> $100) range,
    and prevents buying into the top 20% of a wide (> $100) range.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Exhaustion-Aware $6 Envelope Scalper"

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


class Cycle60_ActiveFrequencyScalper(Cycle35_TightRangeReversionScalper):
    """
    ⚔️ Challenger 3: Active Frequency $5 Envelope Scalper
    $5 envelope with responsive net edge hurdle (>= 8.0 bps) to maximize trade frequency (3-6 trades/hr).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 5.0
        self.min_net_edge_bps = 3.5
        self.strategy_profile = "Challenger 3: Active Frequency $5 Envelope Scalper"


class Cycle60_SteppedRatchetScalper(Cycle35_TightRangeReversionScalper):
    """
    ⚔️ Challenger 4: Stepped Ratchet $6 Envelope Scalper
    Winning $6 envelope + dual-tier stepped profit locks (+2.0b net at +5.0b MFE, +4.0b net at +8.0b MFE).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tier1_min_lock_bps = 4.25
        self.model_harvest_min_bps = 5.50
        self.strategy_profile = "Challenger 4: Stepped Ratchet $6 Envelope Scalper"


class Cycle60_DynamicAsymmetricScalper(Cycle60_ExhaustionAwareScalper):
    """
    ⚔️ Challenger 5: Dynamic Asymmetric Scalper
    Combines the winning $6 envelope, exhaustion guard, and fast trailing locks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 6.0
        self.tier1_min_lock_bps = 4.00
        self.strategy_profile = "Challenger 5: Dynamic Asymmetric Scalper"

print("All classes compiled cleanly.")
