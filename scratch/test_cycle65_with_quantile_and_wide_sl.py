import os
import glob
import json
import io
import sys
import copy
import types
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('feature engineering scripts')

from execution.champion_challenger_framework import Cycle65_UnifiedPeakReversionEngine
from execution.maker_node import AdverseSelectionGuard, QueuePriorityTracker
from execution.hierarchical_sniper_engine import ActiveSwingPosition

class Cycle65_UnifiedPeakReversionEngineFixed(Cycle65_UnifiedPeakReversionEngine):
    def _is_range_boundary_violation(self, primary_direction: int, mid: float) -> bool:
        # Quantile Range Envelope (0.25 to 0.75): Prevents buying range tops and selling range bottoms
        if len(self.rolling_15m_prices) >= 30:
            sess_high = max(self.rolling_15m_prices)
            sess_low = min(self.rolling_15m_prices)
            rng = sess_high - sess_low
            if rng >= 20.0:
                pct = (mid - sess_low) / rng
                if primary_direction == 0 and pct <= 0.25:
                    return True
                if primary_direction == 2 and pct >= 0.75:
                    return True
        return False

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        self.ticks_seen += 1
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        # UNIVERSAL IDLE GATE: Only evaluate entries when in IDLE state
        if self.state == "IDLE":
            # Warmup check
            if self.ticks_seen < 10:
                return self.state

            # FIX 3: Session Runway Cutoff (Bars > 270 have insufficient runway for ratchets)
            if self.ticks_seen > self.max_entry_tick:
                return self.state

            # FIX 4: Directional Discipline & High Conviction Edge Gate
            if decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                cand_dir = None
                if decision.primary_direction == 2 and d_tact >= 0.05 and d_meso >= 0.01:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_tact <= -0.05 and d_meso <= -0.01:
                    cand_dir = 0

                if cand_dir in (0, 2):
                    # Range Boundary Quantile Violation Guard
                    if self._is_range_boundary_violation(cand_dir, mid):
                        return self.state

                    # Impulse Exhaustion Guard
                    if self._is_parabolic_surge(cand_dir, now_ms, mid):
                        return self.state

                    # Post-Trade Settlement & Same-Side Retrace Protection
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

                    tact_dec = copy.copy(decision)
                    if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False):
                        object.__setattr__(tact_dec, "primary_direction", cand_dir)
                    else:
                        setattr(tact_dec, "primary_direction", cand_dir)
                    self.stalking_decision = tact_dec
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state

        # Pass active position management through to base engine unimpeded
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        sign = 1.0 if pos.side == 'BUY' else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4

        if not hasattr(pos, 'mfe'):
            pos.mfe = max(0.0, unrealized_bps)
            pos.mae = min(0.0, unrealized_bps)
        else:
            pos.mfe = max(pos.mfe, unrealized_bps)
            pos.mae = min(pos.mae, unrealized_bps)
        mfe = pos.mfe

        active_floor = None
        ratchet_reason = None
        if mfe >= self.ratchet3_mfe_hurdle:
            active_floor = self.ratchet3_floor_gross
            ratchet_reason = 'TIER3_RUNNER_EXPANSION_TP'
        elif mfe >= self.ratchet2_mfe_hurdle:
            active_floor = self.ratchet2_floor_gross
            ratchet_reason = 'TIER2_RUNNER_RATCHET_TP'
        elif mfe >= self.ratchet1_mfe_hurdle:
            active_floor = self.ratchet1_floor_gross
            ratchet_reason = 'TIER1_STEPPED_RATCHET_TP'
        elif mfe >= self.ratchet0_5_mfe_hurdle:
            active_floor = self.ratchet0_5_floor_gross
            ratchet_reason = 'TIER0_5_CYCLE48_CROWN_RATCHET_TP'
        elif mfe >= self.shield_mfe_hurdle:
            active_floor = self.shield_floor_gross
            ratchet_reason = 'TIER0_BREAKEVEN_MAKER_SHIELD'

        # 1. Full Target TP (+35.00 bps gross / +31.00 bps net)
        if unrealized_bps >= self.target_tp_gross:
            exit_p = pos.entry_price * (1.0 + sign * self.target_tp_gross * 1e-4)
            self._close_position(exit_price=exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason='TARGET_MAKER_TP')
            return

        # 2. Retrace Floor Execution via resting Maker limit order at floor price
        if active_floor is not None and unrealized_bps <= active_floor:
            exit_p = pos.entry_price * (1.0 + sign * active_floor * 1e-4)
            self._close_position(exit_price=exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason=ratchet_reason)
            return

        # 3. Profitable Time Unwind (elapsed >= 900s and gross >= 4.0b)
        if elapsed_sec >= self.max_holding_sec and unrealized_bps >= 4.0:
            maker_exit_p = bbo_ask if pos.side == 'BUY' else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason='PASSIVE_MAKER_TIME_UNWIND')
            return

        # 4. FIX 1: Catastrophic Emergency Stop Breaker (Aggressive Taker Stop Market at -45.0 bps)
        if unrealized_bps <= -45.0:
            taker_exit_p = bbo_bid if pos.side == 'BUY' else bbo_ask
            self._close_position(exit_price=taker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.taker_fee_leg, reason='HARD_STOP_LOSS_TAKER')
            return

    def on_market_micro_tick(self, ts_ms: int, bbo_bid: float, bbo_ask: float, bbo_bid_qty: float, bbo_ask_qty: float, *args, **kwargs):
        # 1. Fill resting entry order at BBO
        if self.state == "RESTING_ENTRY_QUEUE" and self.stalking_decision is not None:
            side = self.resting_entry_side
            limit_p = self.resting_entry_price
            bar_high = kwargs.get("bar_high", bbo_ask)
            bar_low = kwargs.get("bar_low", bbo_bid)
            eff_high = max(bbo_ask, bar_high)
            eff_low = min(bbo_bid, bar_low)

            filled = (eff_low <= limit_p) if side == "BUY" else (eff_high >= limit_p)
            if filled:
                ep = limit_p
                tp_bps = 35.0
                sl_bps = 45.0
                tp_p = ep * (1.0 + tp_bps * 1e-4) if side == "BUY" else ep * (1.0 - tp_bps * 1e-4)
                sl_p = ep * (1.0 - sl_bps * 1e-4) if side == "BUY" else ep * (1.0 + sl_bps * 1e-4)

                self.active_position = ActiveSwingPosition(
                    symbol=self.symbol,
                    side=side,
                    entry_price=ep,
                    quantity=self.resting_entry_qty,
                    entry_ts_ms=ts_ms,
                    target_holding_seconds=900,
                    take_profit_bps=tp_bps,
                    stop_loss_bps=sl_bps,
                    tp_price=tp_p,
                    sl_price=sl_p,
                    maker_entry_fee_bps=self.maker_fee_leg,
                )
                self.state = "FILLED_SWING_HOLDING"
                self.queue_tracker = None
                self.adverse_guard = None
                return self.state

        # 2. Position Management during micro-tick
        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and self.active_position is not None:
            pos = self.active_position
            elapsed_sec = (ts_ms - pos.entry_ts_ms) / 1000.0
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

            # Determine Active Ratchet Floor (Unchoked Multi-Tier Ratchet Ladder)
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

            # 1. Full Target TP
            if unrealized_bps >= self.target_tp_gross:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.maker_fee_leg, reason="TARGET_MAKER_TP")
                return self.state

            # 2. Retrace Floor Execution via resting Maker limit order at floor price
            if active_floor is not None and unrealized_bps <= active_floor:
                exit_p = pos.entry_price * (1.0 + sign * active_floor * 1e-4)
                self._close_position(exit_price=exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.maker_fee_leg, reason=ratchet_reason)
                return self.state

            # 3. Profitable Time Unwind (elapsed >= 900s and gross >= 4.0b)
            if elapsed_sec >= self.max_holding_sec and unrealized_bps >= 4.0:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
                return self.state

            # 4. FIX 1: Catastrophic Emergency Stop Breaker (Aggressive Taker Stop Market at -45.0 bps)
            if unrealized_bps <= -45.0:
                taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
                self._close_position(
                    exit_price=taker_exit_p,
                    exit_ts_ms=ts_ms,
                    exit_fee_bps=self.taker_fee_leg,
                    reason="HARD_STOP_LOSS_TAKER"
                )
                return self.state

        return self.state

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

all_closed_trades = []
active_sessions = 0
wr100_sessions = 0
profitable_sessions = 0
exit_reasons = Counter()

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
    if not ledgers:
        continue
    with open(ledgers[0], "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f if line.strip().startswith("{")]

    engine = Cycle65_UnifiedPeakReversionEngineFixed(symbol="BTCUSDT")

    for idx, bar in enumerate(bars):
        mid = bar.get("bbo_mid", bar.get("mid", 0.0))
        bid = bar.get("bbo_bid", mid - 0.05)
        ask = bar.get("bbo_ask", mid + 0.05)
        ts_ms = bar.get("ts_ms", idx * 10000)
        p_dir = bar.get("primary_direction", bar.get("direction", 1))
        edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))
        tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
        meso = bar.get("meso_probs", [0.33, 0.33, 0.33])

        decision = types.SimpleNamespace(
            ts_ms=ts_ms,
            primary_direction=p_dir,
            expected_net_edge_bps=edge,
            tactical_probs=tact,
            meso_probs=meso,
            macro_probs=[0.33, 0.33, 0.33],
            micro_probs=[0.33, 0.33, 0.33],
            kelly_position_fraction=0.15,
            execution_mode="PASSIVE_MAKER_SNIPER",
            target_holding_seconds=900,
            take_profit_bps=35.0,
            stop_loss_bps=45.0,
            micro_sniper_trigger_prob=0.85
        )

        engine.on_hierarchical_decision(decision, bid, ask, 10.0, 10.0, 10000.0, mid)
        engine.on_market_micro_tick(ts_ms, bid, ask, 10.0, 10.0, bar_high=ask + 0.05, bar_low=bid - 0.05, current_decision=decision)

    session_trades = engine.closed_trades
    if session_trades:
        active_sessions += 1
        wins = sum(1 for t in session_trades if t.net_return_bps > 0)
        sess_net = sum(t.net_return_bps for t in session_trades)
        if wins == len(session_trades):
            wr100_sessions += 1
        if sess_net > 0:
            profitable_sessions += 1

        for t in session_trades:
            exit_reasons[t.exit_reason] += 1
            all_closed_trades.append({
                "cycle": arch,
                "side": t.side,
                "gross": t.gross_return_bps,
                "net": t.net_return_bps,
                "fee": t.total_fee_bps,
                "dur": t.holding_duration_sec,
                "reason": t.exit_reason
            })

n = len(all_closed_trades)
w = sum(1 for t in all_closed_trades if t["net"] > 0)
l = n - w
wr = (w / n * 100.0) if n > 0 else 0.0
tot_pnl = sum(t["net"] for t in all_closed_trades)
avg_pnl = (tot_pnl / n) if n > 0 else 0.0

print("=" * 100)
print("64-CYCLE PRODUCTION REPLAY VERIFICATION (USING OBJECT ORIENTED ENGINE INSTANCES)")
print("=" * 100)
print(f"Total Closed Trades: {n} across {active_sessions} active sessions ({n/max(1, active_sessions):.2f} trades/session)")
print(f"Wins: {w} | Losses: {l} | Win Rate: {wr:.2f}%")
print(f"Total Net PnL: {tot_pnl:+.2f} bps NET REALIZED PROFIT")
print(f"Average Profit per Trade: {avg_pnl:+.2f} bps/trade")
print(f"100% Win Rate Sessions: {wr100_sessions} / {active_sessions} ({wr100_sessions/max(1, active_sessions)*100:.1f}%)")
print(f"Profitable Sessions: {profitable_sessions} / {active_sessions} ({profitable_sessions/max(1, active_sessions)*100:.1f}%)")
print("\nExit Mechanism Distribution:")
for r, cnt in exit_reasons.most_common():
    sub_pnl = sum(t["net"] for t in all_closed_trades if t["reason"] == r)
    print(f"  {r:<35}: {cnt:>3} trades | Net PnL: {sub_pnl:>+8.2f} bps | Avg: {sub_pnl/cnt:>+5.2f} bps")
print("=" * 100)
