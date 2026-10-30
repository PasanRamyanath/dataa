import sys, os, glob, json, copy
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from scratch.test_champ_56_cycles import RealisticLedgerReplayer

class ExactCycle35Challenger5(PeakSameSideRetraceScalper):
    """
    Exact implementation of Challenger 5 from commit 37b3b7d4 that achieved
    +44.64 bps net, 100% win rate in live Cycle 35.
    """
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

        # Conformal confidence check on tactical and meso
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

# Test on Cycle 35
rep = RealisticLedgerReplayer('scratch/archive_cycle35/live_hierarchical_session_1h_ledger.jsonl')
eng = ExactCycle35Challenger5(symbol='BTCUSDT')
trades = rep.replay(eng)
print(f"Cycle 35 Replay with Exact C35 Challenger 5: {len(trades)} trades")
for t in trades:
    print(f"  {t.side} @ {t.entry_price} -> {t.exit_price} | Net: {t.net_return_bps:+.2f}b | MFE: {t.peak_mfe_bps:.2f}b | {t.exit_reason}")
total_pnl = sum(t.net_return_bps for t in trades)
wins = sum(1 for t in trades if t.net_return_bps > 0)
wr = (wins / len(trades) * 100) if trades else 0
print(f"Cycle 35 Replay PnL: {total_pnl:+.2f} bps | Win Rate: {wr:.1f}%")
