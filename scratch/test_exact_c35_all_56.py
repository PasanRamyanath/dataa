import sys, os, glob, json, copy, re
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from scratch.test_champ_56_cycles import RealisticLedgerReplayer, all_ledgers, get_cycle_num

class ExactCycle35Challenger5(PeakSameSideRetraceScalper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Exact Cycle 35 Challenger 5"
        self.range_boundary_buffer_dollars = 10.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        conf_tact = max(decision.tactical_probs)
        conf_meso = max(decision.meso_probs)
        if conf_tact < 0.45 or conf_meso < 0.45:
            return self.state

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.03) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= -0.03)
            )

            cand_dir = tact_dir if tact_confirmed else None
            if cand_dir is None and decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.03:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= -0.03:
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
                        if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 10.0):
                            return self.state
                        elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 10.0):
                            return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

all_trades = []
for l in all_ledgers:
    c_num = get_cycle_num(l)
    rep = RealisticLedgerReplayer(l)
    eng = ExactCycle35Challenger5(symbol="BTCUSDT")
    trades = rep.replay(eng)
    for t in trades:
        all_trades.append((c_num, t))

n_trades = len(all_trades)
wins = sum(1 for _, t in all_trades if t.net_return_bps > 0)
losses = sum(1 for _, t in all_trades if t.net_return_bps <= 0)
total_pnl = sum(t.net_return_bps for _, t in all_trades)
wr = (wins / n_trades * 100) if n_trades > 0 else 0

print("=" * 80)
print(f"EXACT CYCLE 35 CHALLENGER 5 ACROSS ALL {len(all_ledgers)} CYCLES:")
print(f"  Total Trades: {n_trades} ({n_trades / len(all_ledgers):.2f} trades/hr)")
print(f"  Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
print(f"  Total Net Realized PnL: {total_pnl:+.2f} bps")
