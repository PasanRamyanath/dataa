import sys, os, glob, json, copy
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from scratch.test_champ_56_cycles import RealisticLedgerReplayer, all_ledgers, get_cycle_num

class CombinedPeakSynthesisScalper(PeakSameSideRetraceScalper):
    """
    Synthesizes all 4 proven guards:
    1. Range Exhaustion Guard: Forbids buying upper 35% of 15m range, forbids shorting lower 35%.
    2. Retrace Entry Rule: Buys only on pullbacks from 3m peak; shorts only on bounces from 3m trough.
    3. Two-Tier Ratchet: Locks +1.25 bps net at +4.0 bps MFE, +3.00 bps net at +6.5 bps MFE.
    4. Asymmetric Maker Risk Bounding: Exits at BBO Maker when underwater <= -4.5 bps.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Combined Peak Synthesis Scalper"
        self.min_trailing_activation_bps = 4.00
        self.tier1_min_lock_bps = 5.25  # +1.25 bps net after 4.0 bps Maker fees
        self.model_harvest_min_bps = 5.50
        self.min_net_edge_bps = 8.0
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

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        self.ticks_seen += 1
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)

        if self.state == "IDLE":
            if len(self.recent_mid_prices) < 20 or len(self.rolling_15m_prices) < 20:
                return self.state

            r_high = max(self.rolling_15m_prices)
            r_low = min(self.rolling_15m_prices)
            rng = r_high - r_low
            range_pct = (mid - r_low) / rng if rng > 0 else 0.5

            p_3m = [p for _, p in list(self.recent_mid_prices)[-18:]]
            high_3m = max(p_3m)
            low_3m = min(p_3m)

            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]

            cand_dir = None
            if d_tact >= 0.12 and d_meso >= 0.02 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if range_pct <= 0.65 and (mid <= high_3m - 2.5):
                    cand_dir = 2
            elif d_tact <= -0.12 and d_meso <= -0.02 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if range_pct >= 0.35 and (mid >= low_3m + 2.5):
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

# Benchmark on the 5-cycle suite
benchmark_ledgers = [l for l in all_ledgers if get_cycle_num(l) in (35, 48, 57, 58, 59)]
rep_trades = []
for l in benchmark_ledgers:
    rep = RealisticLedgerReplayer(l)
    eng = CombinedPeakSynthesisScalper(symbol="BTCUSDT")
    t_list = rep.replay(eng)
    rep_trades.extend(t_list)

n = len(rep_trades)
w = sum(1 for t in rep_trades if t.net_return_bps > 0)
l_cnt = sum(1 for t in rep_trades if t.net_return_bps <= 0)
wr = (w / n * 100.0) if n > 0 else 0.0
pnl = sum(t.net_return_bps for t in rep_trades)
avg_t = (pnl / n) if n > 0 else 0.0

print("=" * 80)
print(f"COMBINED PEAK SYNTHESIS SCALPER (BENCHMARK SUITE):")
print(f"  Trades: {n} | Wins: {w} | Losses: {l_cnt} | Win Rate: {wr:.1f}%")
print(f"  Net PnL: {pnl:+.2f} bps | Avg/Trade: {avg_t:+.2f} bps")
print("=" * 80)
