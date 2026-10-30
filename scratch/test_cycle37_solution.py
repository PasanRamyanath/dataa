import sys
sys.path.insert(0, "feature engineering scripts")
import json
from pathlib import Path
from execution.champion_challenger_framework import ConformalConfidenceFilterScalper
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

# Test engine with OFI depth + $15 range boundary
class Cycle38_PeakConformalScalper(ConformalConfidenceFilterScalper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_boundary_buffer_dollars = 15.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        imbalance_ratio = bbo_bid_qty / max(bbo_ask_qty, 1e-4)
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        conf_tact = max(decision.tactical_probs)
        conf_meso = max(decision.meso_probs)
        if conf_tact < 0.45 or conf_meso < 0.45:
            return self.state

        if self.state == "IDLE" and self.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.03 and imbalance_ratio >= 1.05) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= -0.03 and imbalance_ratio <= 0.95)
            )

            cand_dir = tact_dir if tact_confirmed else None
            if cand_dir in (0, 2):
                if self._is_range_boundary_violation(cand_dir, mid):
                    return self.state
                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                tact_dec = copy.copy(decision)
                setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)

def test_on_cycle(cycle_num: int):
    ledger_path = Path(f"scratch/archive_cycle{cycle_num}/live_hierarchical_session_1h_ledger.jsonl")
    bars = [json.loads(line) for line in open(ledger_path) if line.strip()]

    eng_c35 = ConformalConfidenceFilterScalper(symbol="BTCUSDT")
    eng_c38 = Cycle38_PeakConformalScalper(symbol="BTCUSDT")

    for b in bars:
        dec = HierarchicalDecisionOutput(
            macro_probs=b["macro_probs"],
            meso_probs=b["meso_probs"],
            tactical_probs=b["tactical_probs"],
            micro_probs=b["micro_probs"],
            fused_swing_probs=b.get("fused_swing_probs", b["tactical_probs"]),
            primary_direction=b["primary_direction"],
            cross_horizon_alignment=b.get("cross_horizon_alignment", 1.0),
            expected_gross_move_bps=b.get("expected_gross_move_bps", 12.0),
            expected_net_edge_bps=b.get("expected_net_edge_bps", 6.0),
            meta_act_probability=b.get("meta_act_probability", 0.8),
            micro_sniper_trigger_prob=b.get("micro_sniper_trigger_prob", 0.7),
            target_holding_seconds=b.get("target_holding_seconds", 300),
            take_profit_bps=b.get("take_profit_bps", 30.0),
            stop_loss_bps=b.get("stop_loss_bps", 14.5),
            kelly_position_fraction=b.get("kelly_position_fraction", 0.5),
            execution_mode="PASSIVE_MAKER_SNIPER",
            ts_ms=b["ts_ms"],
        )
        for eng in [eng_c35, eng_c38]:
            eng.on_market_micro_tick(
                ts_ms=b["ts_ms"],
                bbo_bid=b["bbo_bid"],
                bbo_ask=b["bbo_ask"],
                bbo_bid_qty=b.get("bbo_bid_qty", 1.0),
                bbo_ask_qty=b.get("bbo_ask_qty", 1.0),
                bar_high=b.get("bar_high", b["bbo_ask"]),
                bar_low=b.get("bar_low", b["bbo_bid"]),
                current_decision=dec,
            )
            eng.on_hierarchical_decision(
                decision=dec,
                bbo_bid=b["bbo_bid"],
                bbo_ask=b["bbo_ask"],
                bbo_bid_qty=b.get("bbo_bid_qty", 1.0),
                bbo_ask_qty=b.get("bbo_ask_qty", 1.0),
                equity_usdt=10000.0,
                intra_bar_open=b.get("intra_bar_open", b["bbo_bid"]),
            )

    print(f"Cycle {cycle_num}:")
    print(f"  Cycle 35 Champion: Trades={len(eng_c35.closed_trades)}, Net={sum(t.net_return_bps for t in eng_c35.closed_trades):.2f} bps")
    print(f"  Cycle 38 OFI+Range: Trades={len(eng_c38.closed_trades)}, Net={sum(t.net_return_bps for t in eng_c38.closed_trades):.2f} bps")

if __name__ == "__main__":
    import copy
    test_on_cycle(35)
    test_on_cycle(36)
    test_on_cycle(37)
