import sys, os, glob, json, copy
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import PeakSameSideRetraceScalper
from scratch.test_champ_56_cycles import RealisticLedgerReplayer, all_ledgers, get_cycle_num

class PullbackTrendScalper(PeakSameSideRetraceScalper):
    """
    True Dip-Buyer / Rally-Seller Scalper:
    1. Trend Qualification: Meso & Macro confirm structural trend direction.
    2. Pullback Qualification: Vetoes buying at the top of green surges;
       strictly buys on pullbacks from the 3-minute high (at least $4 pullback).
    3. Rally Qualification: Vetoes shorting at the bottom of red dumps;
       strictly shorts on bounces from the 3-minute low (at least $4 bounce).
    4. Two-Tier Stepped Ratchet: Locks +1.50b net at +4.5b MFE, +3.50b net at +7.0b MFE.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Pullback Trend Scalper"
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
            if len(self.recent_mid_prices) < 20:
                return self.state

            # 3-minute (18 bars * 10s = 180s) prices
            p_3m = [p for _, p in list(self.recent_mid_prices)[-18:]]
            high_3m = max(p_3m)
            low_3m = min(p_3m)

            # 15-minute range
            r_high = max(self.rolling_15m_prices)
            r_low = min(self.rolling_15m_prices)
            rng = r_high - r_low
            range_pct = (mid - r_low) / rng if rng > 0 else 0.5

            # 30s displacement
            disp_30s = 0.0
            if len(self.recent_mid_prices) >= 4:
                disp_30s = (mid - list(self.recent_mid_prices)[-4][1]) / list(self.recent_mid_prices)[-4][1] * 1e4

            cand_dir = None
            # BUY on Pullback:
            # 1. Structural uptrend: Meso bullish, Macro not bearish, Edge >= 10.0
            # 2. Must be pulled back from 3m peak by at least $4.0 (buying the discount)
            # 3. Not surging in the last 30s (disp_30s <= 2.5 bps)
            # 4. Range position <= 70%
            if (d_meso >= 0.03 and d_macro >= 0.00 and decision.expected_net_edge_bps >= self.min_net_edge_bps):
                if (mid <= high_3m - 4.0) and (disp_30s <= 2.5) and (range_pct <= 0.70):
                    cand_dir = 2

            # SELL on Rally:
            # 1. Structural downtrend: Meso bearish, Macro not bullish, Edge >= 10.0
            # 2. Must be bounced up from 3m trough by at least $4.0 (selling the premium)
            # 3. Not plunging in the last 30s (disp_30s >= -2.5 bps)
            # 4. Range position >= 30%
            elif (d_meso <= -0.03 and d_macro <= 0.00 and decision.expected_net_edge_bps >= self.min_net_edge_bps):
                if (mid >= low_3m + 4.0) and (disp_30s >= -2.5) and (range_pct >= 0.30):
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_range_boundary_violation(cand_dir, mid):
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
    eng = PullbackTrendScalper(symbol="BTCUSDT")
    trades = rep.replay(eng)
    for t in trades:
        all_trades.append((c_num, t))

n_trades = len(all_trades)
wins = sum(1 for _, t in all_trades if t.net_return_bps > 0)
losses = sum(1 for _, t in all_trades if t.net_return_bps <= 0)
total_pnl = sum(t.net_return_bps for _, t in all_trades)
wr = (wins / n_trades * 100) if n_trades > 0 else 0

print("=" * 80)
print(f"PULLBACK TREND SCALPER ACROSS ALL {len(all_ledgers)} CYCLES:")
print(f"  Total Trades: {n_trades} ({n_trades / len(all_ledgers):.2f} trades/hr)")
print(f"  Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
print(f"  Total Net Realized PnL: {total_pnl:+.2f} bps")
