import sys, os, glob, json, copy
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import (
    PeakSameSideRetraceScalper,
    HierarchicalDecisionOutput,
    Cycle35_ReversionPureScalper,
)
from scratch.test_champ_56_cycles import RealisticLedgerReplayer, all_ledgers, get_cycle_num

# ==============================================================================
# CHALLENGER 1: Range Exhaustion Guard Scalper
# ==============================================================================
class Challenger1_RangeExhaustionGuardScalper(PeakSameSideRetraceScalper):
    """
    Addresses Flaw 1: The Top-Buying / Bottom-Selling Range Exhaustion Trap.
    Vetoes BUY if price is in the top 35% of the 15m range (range_pct > 0.65).
    Vetoes SELL if price is in the bottom 35% of the 15m range (range_pct < 0.35).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 1: Range Exhaustion Guard Scalper"
        self.range_boundary_buffer_dollars = 8.0
        self.min_net_edge_bps = 8.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        self.ticks_seen += 1
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)

        if self.state == "IDLE":
            if len(self.rolling_15m_prices) < 20:
                return self.state

            r_high = max(self.rolling_15m_prices)
            r_low = min(self.rolling_15m_prices)
            rng = r_high - r_low
            range_pct = (mid - r_low) / rng if rng > 0 else 0.5

            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]

            cand_dir = None
            if d_tact >= 0.12 and d_meso >= 0.02 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if range_pct <= 0.65:  # Forbid buying in upper 35%
                    cand_dir = 2
            elif d_tact <= -0.12 and d_meso <= -0.02 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if range_pct >= 0.35:  # Forbid shorting in lower 35%
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_range_boundary_violation(cand_dir, mid):
                    return self.state
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 60.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 15.0:
                        return self.state

                import copy
                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state
            return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# CHALLENGER 2: Anti-Chasing Retrace Scalper
# ==============================================================================
class Challenger2_AntiChasingRetraceScalper(PeakSameSideRetraceScalper):
    """
    Addresses Flaw 2: The Parabolic Vacuum Chase Trap.
    Enforces a strict pullback before buying (mid <= 3m high - $3.00)
    and a strict bounce before selling (mid >= 3m low + $3.00).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Anti-Chasing Retrace Scalper"
        self.max_parabolic_surge_30s_bps = 4.0
        self.min_net_edge_bps = 8.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        self.ticks_seen += 1
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)

        if self.state == "IDLE":
            if len(self.recent_mid_prices) < 18:
                return self.state

            p_3m = [p for _, p in list(self.recent_mid_prices)[-18:]]
            high_3m = max(p_3m)
            low_3m = min(p_3m)

            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]

            cand_dir = None
            if d_tact >= 0.12 and d_meso >= 0.02 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if mid <= high_3m - 3.0:  # Must pull back at least $3 from recent high
                    cand_dir = 2
            elif d_tact <= -0.12 and d_meso <= -0.02 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if mid >= low_3m + 3.0:   # Must bounce at least $3 from recent low
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 60.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 15.0:
                        return self.state

                import copy
                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state
            return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# CHALLENGER 3: High-Conviction Resonance Scalper
# ==============================================================================
class Challenger3_HighConvictionResonanceScalper(PeakSameSideRetraceScalper):
    """
    Addresses Flaw 3: Flat-Regime Whipsaw Bleed.
    Requires minimum edge >= 12.0 bps and strong cross-horizon directional alignment.
    Rejects FLAT dominance bars.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 3: High-Conviction Resonance Scalper"
        self.min_net_edge_bps = 12.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        self.ticks_seen += 1
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)

        if self.state == "IDLE":
            if self.ticks_seen < 18:
                return self.state

            # Reject Flat/Chop
            tact_dir_max = max(decision.tactical_probs[0], decision.tactical_probs[2])
            meso_dir_max = max(decision.meso_probs[0], decision.meso_probs[2])
            if tact_dir_max < 0.38 or meso_dir_max < 0.38:
                return self.state

            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]

            cand_dir = None
            if d_tact >= 0.16 and d_meso >= 0.05 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                cand_dir = 2
            elif d_tact <= -0.16 and d_meso <= -0.05 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 60.0:
                        return self.state

                import copy
                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state
            return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


# ==============================================================================
# CHALLENGER 4: Two-Tier Micro-Ratchet Scalper
# ==============================================================================
class Challenger4_TwoTierMicroRatchetScalper(PeakSameSideRetraceScalper):
    """
    Addresses Flaw 4: Premature Profit Decay on Favorable Moves.
    Arms Tier 1 Profit Shield at +4.00 bps MFE, guaranteeing +1.25 bps net profit.
    Arms Tier 2 at +6.50 bps MFE, guaranteeing +3.00 bps net profit.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 4: Two-Tier Micro-Ratchet Scalper"
        self.min_trailing_activation_bps = 4.00
        self.tier1_min_lock_bps = 5.25  # +1.25 bps net after 4.0 bps Maker fees
        self.model_harvest_min_bps = 5.50
        self.range_boundary_buffer_dollars = 6.0


# ==============================================================================
# CHALLENGER 5: Asymmetric Risk-Bounding Scalper
# ==============================================================================
class Challenger5_AsymmetricRiskBoundingScalper(PeakSameSideRetraceScalper):
    """
    Addresses Flaw 5: Severe Invalidation & Underwater Drag.
    Caps underwater loss via aggressive Maker Unwind at -4.0 bps gross,
    preventing catastrophic -15 bps taker invalidations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 5: Asymmetric Risk-Bounding Scalper"
        self.range_boundary_buffer_dollars = 6.0

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

        super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)


# Test suite on selected benchmark cycles (Cycles 35, 48, 57, 58, 59)
benchmark_ledgers = [
    l for l in all_ledgers if get_cycle_num(l) in (35, 48, 57, 58, 59)
]

candidates = [
    ("Champion", Cycle35_ReversionPureScalper),
    ("Challenger 1", Challenger1_RangeExhaustionGuardScalper),
    ("Challenger 2", Challenger2_AntiChasingRetraceScalper),
    ("Challenger 3", Challenger3_HighConvictionResonanceScalper),
    ("Challenger 4", Challenger4_TwoTierMicroRatchetScalper),
    ("Challenger 5", Challenger5_AsymmetricRiskBoundingScalper),
]

print("=" * 95)
print(f"BENCHMARK EVALUATION ACROSS BENCHMARK SUITE (CYCLES 35, 48, 57, 58, 59)")
print("=" * 95)
print(f"{'Competitor':15s} | {'Trades':6s} | {'Wins':4s} | {'Losses':6s} | {'Win Rate':8s} | {'Net PnL':10s} | {'Avg/Trade':10s}")
print("-" * 95)

for name, cls in candidates:
    trades_all = []
    for l in benchmark_ledgers:
        rep = RealisticLedgerReplayer(l)
        eng = cls(symbol="BTCUSDT")
        t_list = rep.replay(eng)
        trades_all.extend(t_list)
    
    n = len(trades_all)
    w = sum(1 for t in trades_all if t.net_return_bps > 0)
    l_cnt = sum(1 for t in trades_all if t.net_return_bps <= 0)
    wr = (w / n * 100.0) if n > 0 else 0.0
    pnl = sum(t.net_return_bps for t in trades_all)
    avg_t = (pnl / n) if n > 0 else 0.0
    print(f"{name:15s} | {n:6d} | {w:4d} | {l_cnt:6d} | {wr:7.1f}% | {pnl:+9.2f}b | {avg_t:+9.2f}b")
print("=" * 95)
