import os
import glob
import json
import sys
import types

sys.path.append('feature engineering scripts')
from test_quantile_envelope_tuning import QuantileTunedEngine

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

class PeakEngineWith900sStaleCut(QuantileTunedEngine):
    def __init__(self):
        super().__init__(
            short_min_pct=0.33,
            long_max_pct=0.75,
            min_dir_bars=2,
            trend_threshold_bps=12.0,
            min_net_edge_bps=7.5
        )
        self.cooldown_sec = 45.0
        self.max_entry_tick = 335
        self.hard_stop_bps = 45.0

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4

        if not hasattr(pos, 'mfe'):
            pos.mfe = max(0.0, unrealized_bps)
            pos.mae = min(0.0, unrealized_bps)
        else:
            pos.mfe = max(pos.mfe, unrealized_bps)
            pos.mae = min(pos.mae, unrealized_bps)
        mfe = pos.mfe

        maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        # 1. Target TP hit (Maker Limit Fill)
        if unrealized_bps >= self.target_tp_gross:
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="TARGET_MAKER_TP")
            return

        # 2. Unchoked Stepped Ratchet Ladder
        active_floor = None
        ratchet_reason = None
        if mfe >= self.ratchet3_mfe_hurdle:
            active_floor = self.ratchet3_floor_gross
            ratchet_reason = "TIER3_RUNNER_EXPANSION_TP"
        elif mfe >= self.ratchet2_mfe_hurdle:
            active_floor = self.ratchet2_floor_gross
            ratchet_reason = "TIER2_RUNNER_RATCHET_TP"
        elif mfe >= self.ratchet1_mfe_hurdle:
            active_floor = self.ratchet1_floor_gross
            ratchet_reason = "TIER1_STEPPED_RATCHET_TP"
        elif mfe >= self.ratchet0_5_mfe_hurdle:
            active_floor = self.ratchet0_5_floor_gross
            ratchet_reason = "TIER0_5_CYCLE48_CROWN_RATCHET_TP"
        elif mfe >= self.shield_mfe_hurdle:
            active_floor = self.shield_floor_gross
            ratchet_reason = "TIER0_BREAKEVEN_MAKER_SHIELD"

        if active_floor is not None and unrealized_bps <= active_floor:
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason=ratchet_reason)
            return

        # 3. Time Unwind:
        # A) Profitable time unwind at >= 900s
        if elapsed_sec >= 900.0 and unrealized_bps >= 4.0:
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
            return

        # B) Stale Trapped Cut at >= 900s if opposing meso confirmed
        if elapsed_sec >= 900.0:
            meso_opposing = (sign * d_meso) < -0.10
            if meso_opposing and unrealized_bps <= -15.0:
                taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
                self._close_position(exit_price=taker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.taker_fee_leg, reason="STALE_TRAP_REGIME_UNWIND")
                return

        # 4. Hard Emergency Stop Loss (Taker)
        if unrealized_bps <= -self.hard_stop_bps:
            taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
            self._close_position(exit_price=taker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.taker_fee_leg, reason="HARD_STOP_LOSS_TAKER")
            return

all_closed_trades = []
long_trades = []
short_trades = []
active_sessions = 0
wr100_sessions = 0
profitable_sessions = 0

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]

    engine = PeakEngineWith900sStaleCut()

    for idx, bar in enumerate(bars):
        mid = bar.get('bbo_mid', bar.get('mid', 0.0))
        bid = bar.get('bbo_bid', mid - 0.05)
        ask = bar.get('bbo_ask', mid + 0.05)
        bid_qty = bar.get('bbo_bid_qty', 10.0)
        ask_qty = bar.get('bbo_ask_qty', 10.0)
        ts_ms = bar.get('ts_ms', idx * 10000)

        tact = bar.get('tactical_probs', [0.33, 0.33, 0.33])
        meso = bar.get('meso_probs', [0.33, 0.33, 0.33])
        macro = bar.get('macro_probs', [0.33, 0.33, 0.33])
        micro = bar.get('micro_probs', [0.33, 0.33, 0.33])
        p_dir = bar.get('primary_direction', bar.get('direction', 1))
        edge = bar.get('expected_net_edge_bps', bar.get('edge', 0.0))

        decision = types.SimpleNamespace(
            ts_ms=ts_ms, primary_direction=p_dir, expected_net_edge_bps=edge,
            tactical_probs=tact, meso_probs=meso, macro_probs=macro, micro_probs=micro,
            kelly_position_fraction=0.15, execution_mode='PASSIVE_MAKER_SNIPER',
            target_holding_seconds=900, take_profit_bps=35.0, stop_loss_bps=45.0,
            micro_sniper_trigger_prob=0.85
        )

        engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)
        engine.on_hierarchical_decision(decision, bid, ask, bid_qty, ask_qty, 10000.0, mid)
        if engine.state == 'RESTING_ENTRY_QUEUE':
            engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

    session_trades = engine.closed_trades
    if session_trades:
        active_sessions += 1
        w = sum(1 for t in session_trades if t.net_return_bps > 0)
        sess_net = sum(t.net_return_bps for t in session_trades)
        if w == len(session_trades): wr100_sessions += 1
        if sess_net > 0: profitable_sessions += 1

        for t in session_trades:
            item = {'cycle': arch, 'side': t.side, 'net': t.net_return_bps, 'reason': t.exit_reason}
            all_closed_trades.append(item)
            if t.side == 'BUY': long_trades.append(item)
            else: short_trades.append(item)

n = len(all_closed_trades)
w = sum(1 for t in all_closed_trades if t['net'] > 0)
l = n - w
wr = (w / n * 100.0) if n > 0 else 0.0
tot_pnl = sum(t['net'] for t in all_closed_trades)
avg_pnl = (tot_pnl / n) if n > 0 else 0.0

n_l = len(long_trades)
w_l = sum(1 for t in long_trades if t['net'] > 0)
wr_l = (w_l / n_l * 100.0) if n_l > 0 else 0.0
pnl_l = sum(t['net'] for t in long_trades)

n_s = len(short_trades)
w_s = sum(1 for t in short_trades if t['net'] > 0)
wr_s = (w_s / n_s * 100.0) if n_s > 0 else 0.0
pnl_s = sum(t['net'] for t in short_trades)

gw = sum(t['net'] for t in all_closed_trades if t['net'] > 0)
gl = abs(sum(t['net'] for t in all_closed_trades if t['net'] <= 0))
pf = (gw / gl) if gl > 0 else 999.0

losses_list = [t for t in all_closed_trades if t['net'] <= 0]
loss_strs = [f"{x['cycle']}_{x['side']}({x['net']:+.1f}b, {x['reason']})" for x in losses_list]

print("=" * 100)
print(f"CYCLE 65 PEAK BIDIRECTIONAL SYSTEM (WITH 900s STALE TRAP UNWIND)")
print("=" * 100)
print(f"Total Closed Trades: {n} across {active_sessions} active sessions ({n/max(1,active_sessions):.2f}/session)")
print(f"Overall: Wins={w} | Losses={l} | Win Rate={wr:.2f}% | Net Realized Profit={tot_pnl:+.2f} bps | Avg={avg_pnl:+.2f}b | PF={pf:.2f}")
print(f"  LONGS : {n_l} trades | Wins={w_l} Losses={n_l-w_l} | Win Rate={wr_l:.1f}% | Net Realized PnL={pnl_l:+.2f} bps (avg {pnl_l/max(1,n_l):+.2f}b)")
print(f"  SHORTS: {n_s} trades | Wins={w_s} Losses={n_s-w_s} | Win Rate={wr_s:.1f}% | Net Realized PnL={pnl_s:+.2f} bps (avg {pnl_s/max(1,n_s):+.2f}b)")
print(f"100% WR Sessions: {wr100_sessions}/{active_sessions} ({wr100_sessions/max(1,active_sessions)*100:.1f}%) | Profitable Sessions: {profitable_sessions}/{active_sessions} ({profitable_sessions/max(1,active_sessions)*100:.1f}%)")
print(f"Losses: {loss_strs}")
