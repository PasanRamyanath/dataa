import sys, os, glob, json, copy, re
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from scratch.test_champ_56_cycles import RealisticLedgerReplayer, all_ledgers, get_cycle_num

class QuadResonanceStrictScalper(PeakSameSideRetraceScalper):
    """
    Quad-Horizon Resonance Architecture:
    Enters ONLY when Macro, Meso, Tactical, and Micro ALL confirm the exact same trend.
    Applies the Two-Tier Stepped Ratchet to lock wins immediately.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Quad Resonance Strict Scalper"
        self.range_boundary_buffer_dollars = 8.0
        self.min_net_edge_bps = 10.0
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

            cand_dir = None
            # Bullish Quad-Resonance
            if (d_macro >= 0.05 and d_meso >= 0.05 and d_tact >= 0.12 and d_micro > 0.01 and 
                decision.expected_net_edge_bps >= self.min_net_edge_bps):
                cand_dir = 2

            # Bearish Quad-Resonance
            elif (d_macro <= -0.05 and d_meso <= -0.05 and d_tact <= -0.12 and d_micro < -0.01 and 
                  decision.expected_net_edge_bps >= self.min_net_edge_bps):
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

                    if elapsed < 200.0 and self.last_trade_exit_price > 0:
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

all_trades = []
for l in all_ledgers:
    c_num = get_cycle_num(l)
    rep = RealisticLedgerReplayer(l)
    eng = QuadResonanceStrictScalper(symbol="BTCUSDT")
    trades = rep.replay(eng)
    for t in trades:
        all_trades.append((c_num, t))

n_trades = len(all_trades)
wins = sum(1 for _, t in all_trades if t.net_return_bps > 0)
losses = sum(1 for _, t in all_trades if t.net_return_bps <= 0)
total_pnl = sum(t.net_return_bps for _, t in all_trades)
wr = (wins / n_trades * 100) if n_trades > 0 else 0

print("=" * 80)
print(f"QUAD RESONANCE STRICT SCALPER ACROSS ALL {len(all_ledgers)} CYCLES:")
print(f"  Total Trades: {n_trades} ({n_trades / len(all_ledgers):.2f} trades/hr)")
print(f"  Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
print(f"  Total Net Realized PnL: {total_pnl:+.2f} bps")
