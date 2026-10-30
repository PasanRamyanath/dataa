import sys, os, glob, json, copy, re
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from scratch.test_champ_56_cycles import RealisticLedgerReplayer, all_ledgers, get_cycle_num

# We will test across a representative 10-cycle benchmark suite covering all regimes
# Cycles 5 (bull win), 6 (bear chop), 11 (bear), 12 (bull win), 25 (mega bull win),
# 35 (peak champion), 48 (coronation), 57 (pure win), 58 (bear), 59 (reversion win)
test_cycles = [5, 6, 11, 12, 25, 35, 48, 57, 58, 59]
suite_ledgers = [l for l in all_ledgers if get_cycle_num(l) in test_cycles]
print(f"Testing parametric sweep across {len(suite_ledgers)} diverse benchmark cycles...")

def run_simulation(edge_thresh, range_buffer, retrace_dollars, trail_act, tier1_lock, risk_bound_bps):
    class ParamEngine(PeakSameSideRetraceScalper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.min_net_edge_bps = edge_thresh
            self.range_boundary_buffer_dollars = range_buffer
            self.min_trailing_activation_bps = trail_act
            self.tier1_min_lock_bps = tier1_lock
            self.model_harvest_min_bps = trail_act + 1.0

        def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
            if self.active_position is None:
                return
            pos = self.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            mid = 0.5 * (bbo_bid + bbo_ask)
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4

            # Risk Bounding
            if risk_bound_bps > 0 and unrealized_bps <= -risk_bound_bps:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(
                    exit_price=maker_exit_p,
                    exit_ts_ms=now_ms,
                    exit_fee_bps=self.maker_fee_leg,
                    reason="ASYMMETRIC_RISK_BOUNDING_MAKER",
                )
                return

            super()._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)

        def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
            self.ticks_seen += 1
            now_ms = decision.ts_ms
            mid = 0.5 * (bbo_bid + bbo_ask)
            self.recent_mid_prices.append((now_ms, mid))
            self.rolling_15m_prices.append(mid)

            if self.state == "IDLE":
                if len(self.recent_mid_prices) < 18 or len(self.rolling_15m_prices) < 18:
                    return self.state

                d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
                d_meso = decision.meso_probs[2] - decision.meso_probs[0]

                # Pullback check
                p_3m = [p for _, p in list(self.recent_mid_prices)[-18:]]
                high_3m = max(p_3m)
                low_3m = min(p_3m)

                cand_dir = None
                if d_tact >= 0.12 and d_meso >= 0.02 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                    if retrace_dollars <= 0.0 or (mid <= high_3m - retrace_dollars):
                        cand_dir = 2
                elif d_tact <= -0.12 and d_meso <= -0.02 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                    if retrace_dollars <= 0.0 or (mid >= low_3m + retrace_dollars):
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

                    import copy
                    tact_dec = copy.copy(decision)
                    object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                    self.stalking_decision = tact_dec
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state
                return self.state

            return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

    trades_all = []
    for l in suite_ledgers:
        rep = RealisticLedgerReplayer(l)
        eng = ParamEngine(symbol="BTCUSDT")
        t_list = rep.replay(eng)
        trades_all.extend(t_list)

    n = len(trades_all)
    w = sum(1 for t in trades_all if t.net_return_bps > 0)
    l_cnt = sum(1 for t in trades_all if t.net_return_bps <= 0)
    wr = (w / n * 100.0) if n > 0 else 0.0
    pnl = sum(t.net_return_bps for t in trades_all)
    tr_hr = n / len(suite_ledgers)
    return tr_hr, n, w, l_cnt, wr, pnl

print("\n--- PARAMETRIC SWEEP RESULTS ---")
print(f"{'Config':40s} | {'Tr/Hr':5s} | {'Trades':6s} | {'Wins':4s} | {'Win Rate':8s} | {'Net PnL':10s}")
print("-" * 85)

configs = [
    # (edge, range_buf, retrace_$, trail_act, tier1_lock, risk_bound)
    ("Baseline (Edge 6, $6 buf, No Retrace)", 6.0, 6.0, 0.0, 4.5, 5.5, 0.0),
    ("Edge 8, $6 buf, No Retrace", 8.0, 6.0, 0.0, 4.5, 5.5, 0.0),
    ("Edge 8, $6 buf, Retrace $2.0", 8.0, 6.0, 2.0, 4.5, 5.5, 0.0),
    ("Edge 8, $6 buf, Retrace $2.0, RiskBound 5b", 8.0, 6.0, 2.0, 4.5, 5.5, 5.0),
    ("Edge 8, $6 buf, Retrace $2.0, Trail 4.0b, Lock 5.25b", 8.0, 6.0, 2.0, 4.0, 5.25, 5.0),
    ("Edge 8, $8 buf, Retrace $2.5, Trail 4.0b, Lock 5.25b", 8.0, 8.0, 2.5, 4.0, 5.25, 4.5),
    ("Edge 10, $6 buf, Retrace $2.0, Trail 4.0b, Lock 5.25b", 10.0, 6.0, 2.0, 4.0, 5.25, 4.5),
    ("Edge 10, $8 buf, Retrace $2.0, Trail 4.0b, Lock 5.25b", 10.0, 8.0, 2.0, 4.0, 5.25, 4.5),
]

for label, edge, rbuf, retr, tact, t1lock, rbound in configs:
    tr_hr, n, w, l_cnt, wr, pnl = run_simulation(edge, rbuf, retr, tact, t1lock, rbound)
    print(f"{label:40s} | {tr_hr:5.1f} | {n:6d} | {w:4d} | {wr:7.1f}% | {pnl:+9.2f}b")
