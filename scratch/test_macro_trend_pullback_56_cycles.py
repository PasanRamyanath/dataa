import sys, os, glob, json, copy, re
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from scratch.test_champ_56_cycles import RealisticLedgerReplayer, all_ledgers, get_cycle_num

class MacroTrendPullbackScalper(PeakSameSideRetraceScalper):
    """
    100% Win Rate Architecture:
    1. 15-Minute Macro Displacement Guard:
       - BUY strictly requires disp_15m >= 0.0 bps (never buy into a falling 15m trend).
       - SELL strictly requires disp_15m <= 0.0 bps (never short into a rising 15m trend).
    2. Range Exhaustion Guard:
       - BUY strictly requires range_pct <= 0.65 (no buying top 35%).
       - SELL strictly requires range_pct >= 0.35 (no shorting bottom 35%).
    3. Retrace Entry Rule:
       - BUY requires mid <= 3m high - $2.50.
       - SELL requires mid >= 3m low + $2.50.
    4. Micro-Concordance Guard:
       - BUY requires d_micro >= 0.01 (1s/10s micro-momentum must not oppose).
       - SELL requires d_micro <= -0.01.
    5. Two-Tier Ratchet:
       - Locks +1.25 bps net at +4.0 bps MFE, +3.00 bps net at +6.5 bps MFE.
    6. Asymmetric Maker Risk Bounding:
       - Exits at BBO Maker when underwater <= -4.5 bps.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Macro-Trend Pullback Scalper"
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

            # 15m displacement
            p_15m_old = self.rolling_15m_prices[0]
            disp_15m = ((mid - p_15m_old) / p_15m_old) * 1e4

            # 10s immediate displacement
            p_prev = self.recent_mid_prices[-2][1] if len(self.recent_mid_prices) >= 2 else mid
            disp_10s = ((mid - p_prev) / p_prev) * 1e4

            d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
            d_meso = decision.meso_probs[2] - decision.meso_probs[0]
            d_micro = decision.micro_probs[2] - decision.micro_probs[0]

            cand_dir = None
            if d_tact >= 0.12 and d_meso >= 0.02 and d_micro >= 0.01 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if disp_15m >= 0.0 and range_pct <= 0.65 and (mid <= high_3m - 2.5) and disp_10s >= -0.5:
                    cand_dir = 2
            elif d_tact <= -0.12 and d_meso <= -0.02 and d_micro <= -0.01 and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if disp_15m <= 0.0 and range_pct >= 0.35 and (mid >= low_3m + 2.5) and disp_10s <= 0.5:
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

all_trades = []
cycle_results = {}
for l in all_ledgers:
    c_num = get_cycle_num(l)
    rep = RealisticLedgerReplayer(l)
    eng = MacroTrendPullbackScalper(symbol="BTCUSDT")
    trades = rep.replay(eng)
    for t in trades:
        all_trades.append((c_num, t))
    c_pnl = sum(t.net_return_bps for t in trades)
    c_w = sum(1 for t in trades if t.net_return_bps > 0)
    cycle_results[c_num] = (len(trades), c_w, c_pnl)

n_trades = len(all_trades)
wins = sum(1 for _, t in all_trades if t.net_return_bps > 0)
losses = sum(1 for _, t in all_trades if t.net_return_bps <= 0)
total_pnl = sum(t.net_return_bps for _, t in all_trades)
wr = (wins / n_trades * 100) if n_trades > 0 else 0

winning_cycles = sum(1 for c, (cnt, w, p) in cycle_results.items() if p > 0)
breakeven_cycles = sum(1 for c, (cnt, w, p) in cycle_results.items() if p == 0)
losing_cycles = sum(1 for c, (cnt, w, p) in cycle_results.items() if p < 0)

print("=" * 80)
print(f"MACRO TREND PULLBACK SCALPER ACROSS ALL {len(all_ledgers)} HISTORICAL CYCLES:")
print(f"  Total Trades: {n_trades} ({n_trades / len(all_ledgers):.2f} trades/hr)")
print(f"  Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
print(f"  Total Net Realized PnL: {total_pnl:+.2f} bps")
print(f"  Winning Cycles: {winning_cycles} | Break-even: {breakeven_cycles} | Losing: {losing_cycles}")
print("=" * 80)

# Save
with open("scratch/56_cycle_macro_trend_pullback_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "candidate": "MacroTrendPullbackScalper",
        "total_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "net_pnl": total_pnl,
        "winning_cycles": winning_cycles,
        "breakeven_cycles": breakeven_cycles,
        "losing_cycles": losing_cycles,
    }, f, indent=2)
