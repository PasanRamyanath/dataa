import sys, os

framework_path = 'feature engineering scripts/execution/champion_challenger_framework.py'
with open(framework_path, 'r', encoding='utf-8') as f:
    text = f.read()

old_block = '''class Cycle60_ExhaustionAwareScalper(Cycle35_TightRangeReversionScalper):
    """
    ⚔️ Challenger 2: Exhaustion-Aware $6 Envelope Scalper
    Takes Challenger 1's winning $6 envelope and adds a dynamic Range Exhaustion Guard:
    Prevents shorting into the bottom 20% of a wide (> $100) range,
    and prevents buying into the top 20% of a wide (> $100) range.
    Eliminates extended tail-chasing while maximizing trade execution.
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
    Calibrated $5 envelope buffer with responsive edge hurdle (min_net_edge_bps = 3.5).
    Designed to maximize trade frequency (3-6 trades/hr) without being overly conservative.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 5.0
        self.min_net_edge_bps = 3.5
        self.strategy_profile = "Challenger 3: Active Frequency $5 Envelope Scalper"


class Cycle60_SteppedRatchetScalper(Cycle35_TightRangeReversionScalper):
    """
    ⚔️ Challenger 4: Stepped Ratchet $6 Envelope Scalper
    Winning $6 envelope + dual-tier stepped profit locks (+1.50b net at +4.5b MFE, +3.5b net at +7.5b MFE).
    Guarantees that profitable impulse moves are locked in before market reversion.
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
        self.strategy_profile = "Challenger 5: Dynamic Asymmetric Scalper"'''

new_block = '''class Cycle60_ExhaustionAwareScalper(Cycle35_TightRangeReversionScalper):
    """
    ⚔️ Challenger 2: Exhaustion-Aware $6 Envelope Scalper
    Takes Challenger 1's winning $6 envelope and adds a dynamic Range Exhaustion Guard:
    Prevents shorting into the bottom 25% of a range (rng >= $35),
    and prevents buying into the top 25% of a range (rng >= $35).
    Eliminates extended tail-chasing while maximizing trade execution.
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
            if rng >= 35.0:
                pct = (mid - sess_low) / rng
                if primary_direction == 0 and pct < 0.25:
                    return True
                if primary_direction == 2 and pct > 0.75:
                    return True
        return False


class Cycle60_ActiveFrequencyScalper(Cycle35_TightRangeReversionScalper):
    """
    ⚔️ Challenger 3: Active Frequency $5 Envelope Scalper
    Calibrated $5 envelope buffer with responsive edge hurdle (min_net_edge_bps = 3.5).
    Designed to maximize trade frequency (3-6 trades/hr) without being overly conservative.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 5.0
        self.min_net_edge_bps = 3.5
        self.strategy_profile = "Challenger 3: Active Frequency $5 Envelope Scalper"


class Cycle60_SteppedRatchetScalper(Cycle35_TightRangeReversionScalper):
    """
    ⚔️ Challenger 4: Stepped Ratchet $6 Envelope Scalper
    Winning $6 envelope + dual-tier stepped profit locks (+1.25b net at +4.0b MFE, +3.0b net at +6.5b MFE).
    Guarantees that profitable impulse moves are locked in before market reversion.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_trailing_activation_bps = 4.00
        self.tier1_min_lock_bps = 5.25  # +1.25 bps net profit guaranteed after 4.0 bps Maker fees
        self.model_harvest_min_bps = 5.50
        self.strategy_profile = "Challenger 4: Stepped Ratchet $6 Envelope Scalper"


class Cycle60_DynamicAsymmetricScalper(Cycle60_ExhaustionAwareScalper):
    """
    ⚔️ Challenger 5: Dynamic Asymmetric Scalper
    Combines the winning $6 envelope, exhaustion guard, fast trailing locks, and
    Asymmetric Maker Risk Bounding (unwinds at BBO Maker when underwater <= -4.50 bps).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 6.0
        self.min_trailing_activation_bps = 4.00
        self.tier1_min_lock_bps = 5.25
        self.strategy_profile = "Challenger 5: Dynamic Asymmetric Scalper"

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4

        # Risk Bounding: If underwater <= -4.50 bps, unwind immediately via BBO Maker
        if unrealized_bps <= -4.50:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(
                exit_price=maker_exit_p,
                exit_ts_ms=now_ms,
                exit_fee_bps=self.maker_fee_leg,
                reason="ASYMMETRIC_RISK_BOUNDING_MAKER",
            )
            return

        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)'''

text_norm = text.replace('\r\n', '\n')
old_block_norm = old_block.replace('\r\n', '\n')
new_block_norm = new_block.replace('\r\n', '\n')

assert old_block_norm in text_norm, 'old_block_norm not found in file!'
updated = text_norm.replace(old_block_norm, new_block_norm)

with open(framework_path, 'w', encoding='utf-8') as f:
    f.write(updated)

print("Successfully replaced block in champion_challenger_framework.py!")
