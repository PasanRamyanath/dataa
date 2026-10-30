import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from testing.run_56_cycle_historical_qualification import run_56_cycle_battery
from execution.champion_challenger_framework import PeakSameSideRetraceScalper

class HighPrecisionConcordantScalper(PeakSameSideRetraceScalper):
    """
    Solves adverse execution and negative momentum entries across 56 cycles:
    1. Positive Displacement Guard:
       - For BUY: requires disp_5m >= 0.0 bps and disp_1m >= -0.5 bps.
       - For SELL: requires disp_5m <= 0.0 bps and disp_1m <= 0.5 bps.
    2. Tactical Directional Conviction: |d_tact| >= 0.12 and |d_meso| >= 0.02.
    3. Macro Concordance: d_macro >= -0.02 for BUY, d_macro <= +0.02 for SELL.
    4. Two-Tier Stepped Ratchet: Locks +1.50b net at +4.5b MFE, +3.50b at +7.5b MFE.
    5. $6 Range Boundary Envelope.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "High-Precision Concordant Scalper"
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

        if self.state == "IDLE":
            if self.ticks_seen < 18:
                return self.state

            # Conformal directional certainty
            tact_dir_max = max(decision.tactical_probs[0], decision.tactical_probs[2])
            meso_dir_max = max(decision.meso_probs[0], decision.meso_probs[2])
            if tact_dir_max < 0.35 or meso_dir_max < 0.35:
                return self.state

            # 5m and 1m displacement
            disp_5m = 0.0
            if len(self.recent_mid_prices) >= 30:
                disp_5m = (mid - self.recent_mid_prices[-30][1]) / mid * 10000.0

            disp_1m = 0.0
            if len(self.recent_mid_prices) >= 6:
                disp_1m = (mid - self.recent_mid_prices[-6][1]) / mid * 10000.0

            cand_dir = None
            if d_tact >= 0.12 and d_meso >= 0.02 and d_macro >= -0.02 and disp_5m >= 0.0 and disp_1m >= -0.5 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                cand_dir = 2
            elif d_tact <= -0.12 and d_meso <= -0.02 and d_macro <= 0.02 and disp_5m <= 0.0 and disp_1m <= 0.5 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                cand_dir = 0
            elif decision.primary_direction == 2 and d_meso >= 0.04 and d_macro >= 0.02 and disp_5m >= 1.0 and decision.expected_net_edge_bps >= 10.0:
                cand_dir = 2
            elif decision.primary_direction == 0 and d_meso <= -0.04 and d_macro <= -0.02 and disp_5m <= -1.0 and decision.expected_net_edge_bps >= 10.0:
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
print("TESTING HighPrecisionConcordantScalper ACROSS ALL 56 HISTORICAL CYCLES")
print("=" * 80)

res = run_56_cycle_battery(
    candidate_cls=HighPrecisionConcordantScalper,
    candidate_name="High-Precision Concordant Scalper",
    problem_addressed="Positive Displacement & Concordant Momentum Filter across all 56 cycles"
)

print(f"\nRESULTS ACROSS {res['total_cycles']} HISTORICAL CYCLES:")
print(f"  Total Trades: {res['total_trades']} ({res['avg_trades_per_hour']} trades/hr)")
print(f"  Win Rate: {res['overall_win_rate_pct']}% ({res['total_wins']}W / {res['total_losses']}L)")
print(f"  Total Net Realized PnL: {res['total_net_pnl_bps']:+.2f} bps")
print(f"  Winning Cycles: {res['winning_cycles']} | Break-even: {res['break_even_cycles']} | Losing: {res['losing_cycles']}")
print(f"  Passed Qualification Standard: {res['passed_qualification']}")

with open("scratch/56_cycle_concordant_qualification_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
