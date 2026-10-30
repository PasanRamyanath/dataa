import json
import os
import sys
import copy
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath("feature engineering scripts"))
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine, ActiveSwingPosition, ClosedSwingTradeRecord
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

class SteppedProfitRatchetScalper(HierarchicalSniperExecutionEngine):
    """
    Challenger: Stepped Profit Ratchet Scalper
    Inherits Peak baseline + Multi-tiered profit ratchet floor:
    - MFE >= 3.80 bps -> floor locked at +4.10 bps gross (+0.10 net)
    - MFE >= 5.50 bps -> floor ratcheted to +5.20 bps gross (+1.20 net)
    - MFE >= 7.50 bps -> floor ratcheted to +7.00 bps gross (+3.00 net)
    - MFE >= 10.0 bps -> floor ratcheted to +9.00 bps gross (+5.00 net)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Stepped Profit Ratchet Scalper"
        self.last_trade_exit_price: float = 0.0
        self.last_trade_side: str = ""

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None:
            self.last_trade_exit_price = exit_price
            self.last_trade_side = self.active_position.side
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

    def on_market_micro_tick(self, ts_ms, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, *args, **kwargs):
        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and self.active_position is not None:
            pos = self.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            mid = 0.5 * (bbo_bid + bbo_ask)
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            if unrealized_bps > pos.peak_favorable_excursion_bps:
                pos.peak_favorable_excursion_bps = unrealized_bps

            # Determine Stepped Ratchet Floor
            mfe = pos.peak_favorable_excursion_bps
            floor_gross = 0.0
            if mfe >= 10.0:
                floor_gross = 9.00
            elif mfe >= 7.50:
                floor_gross = 7.00
            elif mfe >= 5.50:
                floor_gross = 5.20
            elif mfe >= 3.80:
                floor_gross = 4.10

            if floor_gross > 0.0 and not pos.trailing_maker_armed:
                if unrealized_bps <= floor_gross:
                    exit_p = pos.entry_price * (1.0 + sign * floor_gross * 1e-4)
                    self._close_position(
                        exit_price=exit_p,
                        exit_ts_ms=ts_ms,
                        exit_fee_bps=self.maker_fee_leg,
                        reason=f"STEPPED_PROFIT_RATCHET_{floor_gross:.1f}BPS",
                    )
                    return self.state

        return super().on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bbo_bid, bbo_ask=bbo_ask, bbo_bid_qty=bbo_bid_qty, bbo_ask_qty=bbo_ask_qty, *args, **kwargs)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            cand_dir = None
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= 0.00) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= 0.00)
            )

            if tact_confirmed:
                cand_dir = tact_dir
            elif decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.00:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= 0.00:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 90.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 20.0:
                        return self.state

                    if elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 15.0):
                            return self.state
                        elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 15.0):
                            return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)


def run_session_replay(ledger_path, engine_factory):
    with open(ledger_path, "r") as f:
        bars = [json.loads(line) for line in f if line.strip()]

    engine = engine_factory()
    for bar in bars:
        bid = bar["bbo_bid"]
        ask = bar["bbo_ask"]
        ts = bar["ts_ms"]

        dec = HierarchicalDecisionOutput(
            ts_ms=ts,
            macro_probs=tuple(bar["macro_probs"]),
            meso_probs=tuple(bar["meso_probs"]),
            tactical_probs=tuple(bar["tactical_probs"]),
            micro_probs=tuple(bar["micro_probs"]),
            fused_swing_probs=(0.33, 0.34, 0.33),
            primary_direction=bar["primary_direction"],
            cross_horizon_alignment=1.0,
            expected_gross_move_bps=bar.get("expected_gross_move_bps", 10.0),
            expected_net_edge_bps=bar["expected_net_edge_bps"],
            meta_act_probability=0.6,
            micro_sniper_trigger_prob=0.6,
            target_holding_seconds=bar.get("suggested_holding_seconds", 300),
            take_profit_bps=15.0,
            stop_loss_bps=12.0,
            kelly_position_fraction=0.1,
            execution_mode="PASSIVE_MAKER_SNIPER" if bar["primary_direction"] != 1 else "NONE"
        )

        engine.on_market_micro_tick(
            ts_ms=ts,
            bbo_bid=bid,
            bbo_ask=ask,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            bar_high=bar.get("bar_high", bid),
            bar_low=bar.get("bar_low", ask),
            current_decision=dec,
        )

        engine.on_hierarchical_decision(
            decision=dec,
            bbo_bid=bid,
            bbo_ask=ask,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            equity_usdt=1000.0,
            intra_bar_open=bar["bbo_mid"]
        )

    trades = engine.closed_trades
    n = len(trades)
    w = sum(1 for t in trades if t.net_return_bps > 0)
    net = sum(t.net_return_bps for t in trades)
    wr = (w / n * 100.0) if n > 0 else 0.0
    return {
        "trades": n,
        "wins": w,
        "win_rate": wr,
        "net_pnl": net,
        "trade_details": [
            {
                "side": t.side,
                "net": round(t.net_return_bps, 2),
                "mfe": round(t.peak_mfe_bps, 2),
                "dur": t.holding_duration_sec,
                "exit": t.exit_reason
            }
            for t in trades
        ]
    }

print("=== REPLAY ON CYCLE 31 ===")
from execution.champion_challenger_framework import PeakSameSideRetraceScalper
c31_champ = run_session_replay("scratch/archive_cycle31/live_hierarchical_session_1h_ledger.jsonl", lambda: PeakSameSideRetraceScalper(symbol="BTCUSDT"))
c31_stepped = run_session_replay("scratch/archive_cycle31/live_hierarchical_session_1h_ledger.jsonl", lambda: SteppedProfitRatchetScalper(symbol="BTCUSDT"))
print("Cycle 31 Champion:", c31_champ)
print("Cycle 31 Stepped Ratchet:", c31_stepped)

print("\n=== REPLAY ON CYCLE 26 ===")
c26_champ = run_session_replay("scratch/archive_cycle26/live_hierarchical_session_1h_ledger.jsonl", lambda: PeakSameSideRetraceScalper(symbol="BTCUSDT"))
c26_stepped = run_session_replay("scratch/archive_cycle26/live_hierarchical_session_1h_ledger.jsonl", lambda: SteppedProfitRatchetScalper(symbol="BTCUSDT"))
print("Cycle 26 Champion:", c26_champ)
print("Cycle 26 Stepped Ratchet:", c26_stepped)

print("\n=== REPLAY ON CYCLE 25 ===")
c25_champ = run_session_replay("scratch/archive_cycle25/live_hierarchical_session_1h_ledger.jsonl", lambda: PeakSameSideRetraceScalper(symbol="BTCUSDT"))
c25_stepped = run_session_replay("scratch/archive_cycle25/live_hierarchical_session_1h_ledger.jsonl", lambda: SteppedProfitRatchetScalper(symbol="BTCUSDT"))
print("Cycle 25 Champion:", c25_champ)
print("Cycle 25 Stepped Ratchet:", c25_stepped)
