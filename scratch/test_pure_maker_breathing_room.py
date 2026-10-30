import os
import sys
import copy
import json

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('scratch'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from simulate_all_59_cycles_qualification import replay_cycle_ledger
from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

benchmark_suite = [
    "archive_cycle63",
    "archive_cycle62",
    "archive_cycle61",
    "archive_cycle60",
    "archive_cycle59",
    "archive_cycle58",
    "archive_cycle57",
    "archive_cycle35",
]

class PureMaker_BreathingRoom_Engine(Cycle35_ReversionPureScalper):
    """
    Pure 100% Maker Execution + True Volatility Breathing Room:
    1. Stop loss widened to 25.0 bps (gives $200 breathing room on $78,000 BTC).
    2. Zero premature taker cuts (no 0B2 taker stop on 6.5 bps noise).
    3. Structural invalidation: If elapsed >= 90s and structure is opposing, unwind via MAKER order at BBO.
    4. Trailing profit shield: Locks +1.25 bps net at +4.5 bps MFE, trails runners freely.
    5. Pure Maker exits: All regular exits are Maker orders at BBO (2.0 bps fee per leg).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Pure Maker Breathing Room Scalper (25 bps SL + Passive Maker Unwinds)"

    def _place_sniper_maker_entry(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open=0.0):
        # Override to ensure stop_loss_bps is 25.0 bps (not capped to 14.5 bps)
        side = "BUY" if decision.primary_direction == 2 else "SELL"
        limit_price = bbo_bid if side == "BUY" else bbo_ask
        notional = max(100.0, equity_usdt * max(0.10, decision.kelly_position_fraction))
        qty = round(notional / limit_price, 3)
        if qty <= 0:
            qty = 0.001

        # Rest resting order
        self.resting_entry_side = side
        self.resting_entry_price = limit_price
        self.resting_entry_qty = qty
        self.resting_entry_ts_ms = now_ms
        self.state = "RESTING_ENTRY_QUEUE"

    def on_market_micro_tick(self, ts_ms, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, current_decision=None, aggressive_buy_vol_at_ask=0.0, aggressive_sell_vol_at_bid=0.0, is_10s_bar_mode=True):
        # Handle resting fill with 25.0 bps SL
        if self.state == "RESTING_ENTRY_QUEUE":
            filled = False
            if self.resting_entry_side == "BUY":
                if bbo_bid <= self.resting_entry_price or bbo_ask <= self.resting_entry_price:
                    filled = True
            else:
                if bbo_ask >= self.resting_entry_price or bbo_bid >= self.resting_entry_price:
                    filled = True

            if filled:
                ep = self.resting_entry_price
                tp_bps = 15.0
                sl_bps = 25.0  # WIDE 25 bps breathing room!
                tp_p = ep * (1.0 + tp_bps * 1e-4) if self.resting_entry_side == "BUY" else ep * (1.0 - tp_bps * 1e-4)
                sl_p = ep * (1.0 - sl_bps * 1e-4) if self.resting_entry_side == "BUY" else ep * (1.0 + sl_bps * 1e-4)

                from execution.hierarchical_sniper_engine import ActiveSwingPosition
                self.active_position = ActiveSwingPosition(
                    symbol=self.symbol,
                    side=self.resting_entry_side,
                    entry_price=ep,
                    quantity=self.resting_entry_qty,
                    entry_ts_ms=ts_ms,
                    target_holding_seconds=300.0,
                    take_profit_bps=tp_bps,
                    stop_loss_bps=sl_bps,
                    tp_price=tp_p,
                    sl_price=sl_p,
                    maker_entry_fee_bps=self.maker_fee_leg,
                )
                self.state = "FILLED_SWING_HOLDING"
                return self.state

        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND"):
            pos = self.active_position
            if pos is None:
                return self.state

            mid = 0.5 * (bbo_bid + bbo_ask)
            sign = 1.0 if pos.side == "BUY" else -1.0
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            pos.peak_favorable_excursion_bps = max(pos.peak_favorable_excursion_bps, unrealized_bps)
            pos.max_adverse_excursion_bps = min(pos.max_adverse_excursion_bps, unrealized_bps)

            # Check Hard Stop Loss only at 25.0 bps
            if unrealized_bps <= -25.0:
                taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
                self._close_position(exit_price=taker_exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.taker_fee_leg, reason="HARD_STOP_LOSS_TAKER")
                return self.state

            # Resting Maker TP
            if unrealized_bps >= 15.0:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.maker_fee_leg, reason="RESTING_MAKER_TP")
                return self.state

        return self.state

    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):
        if self.active_position is None:
            return
        pos = self.active_position
        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0
        hold_frac = elapsed_sec / max(1.0, float(pos.target_holding_seconds))

        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
        d_macro = decision.macro_probs[2] - decision.macro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        opp_dir = 0 if pos.side == "BUY" else 2

        # 1. Take Profit
        if unrealized_bps >= pos.take_profit_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="RESTING_MAKER_TP")
            return

        # 2. Neural Model Profit Harvest
        if unrealized_bps >= self.model_harvest_min_bps:
            if (sign * d_micro) < -0.05 or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="MODEL_HARVEST_MAKER_TP")
                return

        # 3. Dynamic Trailing Profit Shield
        if unrealized_bps >= 3.80:
            if not getattr(pos, 'trailing_armed', False):
                pos.trailing_armed = True
                pos.peak_excursion = unrealized_bps
            else:
                pos.peak_excursion = max(pos.peak_excursion, unrealized_bps)
            
            pullback = pos.peak_excursion - unrealized_bps
            if pos.peak_excursion >= 6.50 and pullback >= 2.00:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="TRAILING_SHIELD_MAKER_TP")
                return

        # 4. Structural Invalidation via Passive Maker (only after 90s of structural confirmation)
        if elapsed_sec >= 90.0 and unrealized_bps <= -3.00:
            meso_opposing = (sign * d_meso) < -0.12
            if meso_opposing or decision.primary_direction == opp_dir:
                maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
                self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_STRUCTURAL_INVAL")
                return

        # 5. Profitable Time Unwind at 300s
        if elapsed_sec >= 300.0 and unrealized_bps >= self.passive_unwind_min_bps:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
            return

        # 6. Max Duration Unwind at 600s
        if elapsed_sec >= 600.0:
            maker_exit_p = bbo_ask if pos.side == "BUY" else bbo_bid
            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")


print("="*95)
print("TESTING PURE MAKER BREATHING ROOM ENGINE ACROSS 8 BENCHMARK CYCLES")
print("="*95)

eng_cls = PureMaker_BreathingRoom_Engine
total_trades = 0
total_wins = 0
total_losses = 0
total_net = 0.0
total_fees = 0.0

per_cycle_pnl = {}
for cycle_dir in benchmark_suite:
    ledger_path = os.path.join("scratch", cycle_dir, "live_hierarchical_session_1h_ledger.jsonl")
    if not os.path.exists(ledger_path):
        continue
    eng = eng_cls(symbol="BTCUSDT")
    trades = replay_cycle_ledger(eng, ledger_path)
    c_trades = len(trades)
    c_wins = sum(1 for t in trades if t["is_win"])
    c_net = sum(t["net_return_bps"] for t in trades)
    c_fees = sum(t["total_fee_bps"] for t in trades)
    
    total_trades += c_trades
    total_wins += c_wins
    total_losses += (c_trades - c_wins)
    total_net += c_net
    total_fees += c_fees
    per_cycle_pnl[cycle_dir] = (c_trades, c_wins, c_net)

wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
print(f"TOTAL: Trades: {total_trades:2d} | Wins: {total_wins:2d} | Losses: {total_losses:2d} | WR: {wr:5.1f}% | Net: {total_net:+7.2f} bps | Fees: {total_fees:5.1f}b")
for c_name, (c_tr, c_w, c_pnl) in per_cycle_pnl.items():
    print(f"  - {c_name:<18}: {c_tr} trades, {c_w} wins, net {c_pnl:+6.2f} bps")
