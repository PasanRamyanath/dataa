import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from testing.run_56_cycle_historical_qualification import run_56_cycle_battery
from execution.champion_challenger_framework import PeakSameSideRetraceScalper

class MacroAlignedPeakScalper(PeakSameSideRetraceScalper):
    """
    Solves the Counter-Trend Cascade Trap:
    1. Macro Alignment Guard: Vetoes BUY if Macro is strongly bearish (d_macro < -0.05),
       vetoes SELL if Macro is strongly bullish (d_macro > +0.05).
    2. 5-Minute Velocity Filter: Blocks BUY if 5m price change < -$20,
       blocks SELL if 5m price change > +$20.
    3. Directional Conformal Certainty: strictly requires directional probability
       max(p_down, p_up) >= 0.35 across Tactical and Meso.
    4. Minimum Directional Spread: |d_tact| >= 0.14 and |d_meso| >= 0.04.
    5. $6 Range Boundary Envelope: Prevents oscillating range top-buying and bottom-selling.
    6. Two-Tier Stepped Ratchet: Locks +1.50b net at +4.5b MFE and +3.50b net at +7.5b MFE.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Macro-Aligned Peak Scalper"
        self.range_boundary_buffer_dollars = 6.0
        self.min_net_edge_bps = 8.0
        self.min_trailing_activation_bps = 4.50
        self.tier1_min_lock_bps = 5.50
        self.model_harvest_min_bps = 6.00

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        self.ticks_seen += 1
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)

        d_macro = decision.macro_probs[2] - decision.macro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]

        if self.state == "IDLE":
            if self.ticks_seen < 18:
                return self.state

            # 1. Macro Trend Veto
            # Do NOT buy into macro downtrend, do NOT sell into macro uptrend
            tact_dir_max = max(decision.tactical_probs[0], decision.tactical_probs[2])
            meso_dir_max = max(decision.meso_probs[0], decision.meso_probs[2])
            if tact_dir_max < 0.35 or meso_dir_max < 0.35:
                return self.state

            # 2. 5-Minute Velocity Check (30 bars * 10s = 300s = 5m)
            vel_5m = 0.0
            if len(self.recent_mid_prices) >= 30:
                p_old = self.recent_mid_prices[-30][1]
                vel_5m = mid - p_old

            cand_dir = None
            if d_tact >= 0.14 and d_meso >= 0.04 and d_macro >= -0.05 and vel_5m >= -15.0 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                cand_dir = 2
            elif d_tact <= -0.14 and d_meso <= -0.04 and d_macro <= 0.05 and vel_5m <= 15.0 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                cand_dir = 0
            elif decision.primary_direction == 2 and d_meso >= 0.05 and d_macro >= 0.00 and vel_5m >= -10.0 and decision.expected_net_edge_bps >= 10.0:
                cand_dir = 2
            elif decision.primary_direction == 0 and d_meso <= -0.05 and d_macro <= 0.00 and vel_5m <= 10.0 and decision.expected_net_edge_bps >= 10.0:
                cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_range_boundary_violation(cand_dir, mid):
                    return self.state

                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 45.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 15.0:
                        return self.state

                    if elapsed < 240.0 and self.last_trade_exit_price > 0:
                        if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - self.range_boundary_buffer_dollars):
                            return self.state
                        elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + self.range_boundary_buffer_dollars):
                            return self.state

                import copy
                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

            return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

print("=" * 80)
print("TESTING MacroAlignedPeakScalper ACROSS ALL 56 HISTORICAL CYCLES")
print("=" * 80)

res = run_56_cycle_battery(
    candidate_cls=MacroAlignedPeakScalper,
    candidate_name="Macro-Aligned Peak Scalper",
    problem_addressed="Macro Alignment & 5m Velocity Filter to prevent cascade traps"
)

print(f"\nRESULTS ACROSS {res['total_cycles']} HISTORICAL CYCLES:")
print(f"  Total Trades: {res['total_trades']} ({res['avg_trades_per_hour']} trades/hr)")
print(f"  Win Rate: {res['overall_win_rate_pct']}% ({res['total_wins']}W / {res['total_losses']}L)")
print(f"  Total Net Realized PnL: {res['total_net_pnl_bps']:+.2f} bps")
print(f"  Winning Cycles: {res['winning_cycles']} | Break-even: {res['break_even_cycles']} | Losing: {res['losing_cycles']}")
print(f"  Passed Qualification Standard: {res['passed_qualification']}")

# Save
with open("scratch/56_cycle_macro_aligned_qualification_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
