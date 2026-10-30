"""
scratch/replay_cycle26_exact.py
Replay Cycle 26 exactly using SameSideRetracePriceBarrierScalper
"""
import json
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path("feature engineering scripts").resolve()))
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine

class SameSideRetracePriceBarrierScalper(HierarchicalSniperExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Reigning Champion: Same-Side Retrace Barrier Scalper (+30.68 bps net in Cycle 25)"
        self.min_net_edge_bps = 4.0
        self.last_trade_exit_price: float = 0.0
        self.last_trade_side: str = ""

    def _close_position(self, exit_price, exit_ts_ms, exit_fee_bps, reason="TAKE_PROFIT"):
        if self.active_position is not None:
            self.last_trade_exit_price = exit_price
            self.last_trade_side = self.active_position.side
        return super()._close_position(exit_price, exit_ts_ms, exit_fee_bps, reason)

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= -0.05) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= 0.05)
            )
            
            if tact_confirmed and not self._is_parabolic_surge(tact_dir, now_ms, mid):
                can_reenter = True
                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 90.0:
                        can_reenter = False
                    elif self.last_exit_was_win and elapsed < 20.0:
                        can_reenter = False
                    
                    if can_reenter and elapsed < 300.0 and self.last_trade_exit_price > 0:
                        if tact_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 15.0):
                            can_reenter = False
                        elif tact_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 15.0):
                            can_reenter = False

                if can_reenter:
                    tact_dec = copy.copy(decision)
                    object.__setattr__(tact_dec, "primary_direction", tact_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", tact_dir)
                    self.stalking_decision = tact_dec
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

# Replay Cycle 26
with open("scratch/archive_cycle26/live_hierarchical_session_1h_ledger.jsonl") as f:
    bars = [json.loads(line) for line in f]

from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
import numpy as np

engine = SameSideRetracePriceBarrierScalper(symbol="BTCUSDT")

for b in bars:
    dec = HierarchicalDecisionOutput(
        ts_ms=b["ts_ms"],
        primary_direction=b["primary_direction"],
        macro_probs=np.array(b["macro_probs"]),
        meso_probs=np.array(b["meso_probs"]),
        tactical_probs=np.array(b["tactical_probs"]),
        micro_probs=np.array(b["micro_probs"]),
        fused_swing_probs=np.array(b["tactical_probs"]),
        cross_horizon_alignment=0.8,
        expected_gross_move_bps=15.0,
        expected_net_edge_bps=b["expected_net_edge_bps"],
        meta_act_probability=0.75,
        micro_sniper_trigger_prob=0.6,
        target_holding_seconds=300,
        take_profit_bps=25.0,
        stop_loss_bps=15.0,
        kelly_position_fraction=0.1,
        execution_mode=b.get("execution_mode", "NONE"),
    )
    bbo_bid = b["bbo_bid"]
    bbo_ask = b["bbo_ask"]
    mid = b["bbo_mid"]
    
    engine.on_market_micro_tick(
        ts_ms=b["ts_ms"],
        bbo_bid=bbo_bid,
        bbo_ask=bbo_ask,
        bbo_bid_qty=1.0,
        bbo_ask_qty=1.0,
        bar_high=mid + 1.0,
        bar_low=mid - 1.0,
        current_decision=dec,
    )
    engine.on_hierarchical_decision(
        decision=dec,
        bbo_bid=bbo_bid,
        bbo_ask=bbo_ask,
        bbo_bid_qty=1.0,
        bbo_ask_qty=1.0,
        equity_usdt=10000.0,
        intra_bar_open=mid,
    )

print(f"Closed trades: {len(engine.closed_trades)}")
for i, t in enumerate(engine.closed_trades, 1):
    print(f"Trade {i}: {t.side} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:+.2f} bps | MFE: {t.peak_mfe_bps:+.2f}b | {t.exit_reason}")
