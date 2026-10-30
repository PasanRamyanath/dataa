"""
Test Cycle 33 replay with:
1. Baseline (as executed)
2. Fix 1: Unchoke Break-Even Shield (arm at MFE >= 5.50 bps, floor at <= 4.20 bps)
3. Fix 2: Meso conviction filter d_meso >= 0.05 for BUY / <= -0.05 for SELL
"""
import sys
import json
from pathlib import Path
import copy

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEAT_ENGINE_DIR = (REPO_ROOT / "feature engineering scripts").resolve()
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine, HierarchicalDecisionOutput
from execution.champion_challenger_framework import PeakSameSideRetraceScalper, Challenger5_DynamicFrequencyHighVelocityScalper

LEDGER_PATH = REPO_ROOT / "scratch" / "archive_cycle33" / "live_hierarchical_session_1h_ledger.jsonl"

def replay(engine_cls, engine_kwargs=None, d_meso_thresh=0.00, be_hurdle=3.80, be_floor=4.10):
    if engine_kwargs is None:
        engine_kwargs = {}
    engine = engine_cls(**engine_kwargs)
    
    # Override BE parameters for testing
    original_on_market_micro_tick = engine.on_market_micro_tick
    def patched_micro_tick(ts_ms, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty):
        if engine.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and engine.active_position is not None:
            pos = engine.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            mid = 0.5 * (bbo_bid + bbo_ask)
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            if unrealized_bps > pos.peak_favorable_excursion_bps:
                pos.peak_favorable_excursion_bps = unrealized_bps
            
            # Check unchoked BE shield
            if pos.peak_favorable_excursion_bps >= be_hurdle and not pos.trailing_maker_armed:
                if unrealized_bps <= be_floor:
                    exit_be_p = pos.entry_price * (1.0 + sign * be_floor * 1e-4)
                    engine._close_position(
                        exit_price=exit_be_p,
                        exit_ts_ms=ts_ms,
                        exit_fee_bps=engine.maker_fee_leg,
                        reason="BREAKEVEN_MAKER_PROFIT_SHIELD",
                    )
                    return engine.state
        return original_on_market_micro_tick(ts_ms, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty)
    
    # Also patch Meso concordance if required
    original_on_decision = engine.on_hierarchical_decision
    def patched_on_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        engine.recent_mid_prices.append((now_ms, mid))
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if engine.state == "IDLE" and engine.ticks_seen >= 10:
            cand_dir = None
            tact_dir = 2 if d_tact >= 0.14 else (0 if d_tact <= -0.14 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.02 and d_meso >= d_meso_thresh) or
                (tact_dir == 0 and d_micro < -0.02 and d_meso <= -d_meso_thresh)
            )

            if tact_confirmed:
                cand_dir = tact_dir
            elif decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= engine.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= d_meso_thresh:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= -d_meso_thresh:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if engine._is_parabolic_surge(cand_dir, now_ms, mid):
                    return engine.state

                if engine.last_exit_ts_ms > 0:
                    elapsed = (now_ms - engine.last_exit_ts_ms) / 1000.0
                    if not engine.last_exit_was_win and elapsed < 90.0:
                        return engine.state
                    elif engine.last_exit_was_win and elapsed < 20.0:
                        return engine.state

                    if elapsed < 300.0 and engine.last_trade_exit_price > 0:
                        if cand_dir == 2 and engine.last_trade_side == "BUY" and (mid > engine.last_trade_exit_price - 15.0):
                            return engine.state
                        elif cand_dir == 0 and engine.last_trade_side == "SELL" and (mid < engine.last_trade_exit_price + 15.0):
                            return engine.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir)
                engine.stalking_decision = tact_dec
                engine._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return engine.state

        # Manage active position
        engine.ticks_seen += 1
        if engine.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND"):
            engine._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)
        elif engine.state == "RESTING_ENTRY_QUEUE":
            pass
        return engine.state

    engine.on_market_micro_tick = patched_micro_tick
    engine.on_hierarchical_decision = patched_on_decision

    with open(LEDGER_PATH, 'r') as f:
        bars = [json.loads(line) for line in f]

    for bar in bars:
        ts = bar["ts_ms"]
        bid = bar["bbo_bid"]
        ask = bar["bbo_ask"]
        mid = bar["bbo_mid"]

        # Run tick
        engine.on_market_micro_tick(ts_ms=ts, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.0, bbo_ask_qty=1.0)

        # Build decision
        dec = HierarchicalDecisionOutput(
            ts_ms=ts,
            macro_probs=tuple(bar["macro_probs"]),
            meso_probs=tuple(bar["meso_probs"]),
            tactical_probs=tuple(bar["tactical_probs"]),
            micro_probs=tuple(bar["micro_probs"]),
            fused_swing_probs=(0.33, 0.34, 0.33),
            primary_direction=bar["primary_direction"],
            cross_horizon_alignment=0.8,
            expected_gross_move_bps=34.0,
            expected_net_edge_bps=bar["expected_net_edge_bps"],
            meta_act_probability=0.8,
            micro_sniper_trigger_prob=0.8,
            target_holding_seconds=1800,
            take_profit_bps=30.0,
            stop_loss_bps=14.5,
            kelly_position_fraction=0.012,
            execution_mode="PASSIVE_MAKER_SNIPER"
        )
        engine.on_hierarchical_decision(dec, bid, ask, 1.0, 1.0)

    trades = engine.closed_trades
    wins = [t for t in trades if t.net_return_bps > 0]
    total_net = sum(t.net_return_bps for t in trades)
    wr = len(wins) / len(trades) * 100.0 if trades else 0.0
    return len(trades), len(wins), wr, total_net, trades

print("--- REPLAY RESULTS ---")
n, w, wr, net, tr = replay(PeakSameSideRetraceScalper, d_meso_thresh=0.00, be_hurdle=3.80, be_floor=4.10)
print(f"Baseline: Trades={n}, Wins={w}, WR={wr:.1f}%, Net={net:.2f} bps")
for t in tr:
    print(f"  {t.side} Entry={t.entry_price} Exit={t.exit_price} Net={t.net_return_bps:.2f}b MFE={t.peak_mfe_bps:.2f}b Reason={t.exit_reason}")

n, w, wr, net, tr = replay(PeakSameSideRetraceScalper, d_meso_thresh=0.05, be_hurdle=5.50, be_floor=4.20)
print(f"\nUnchoked BE (5.5b hurdle) + Meso Conviction (0.05): Trades={n}, Wins={w}, WR={wr:.1f}%, Net={net:.2f} bps")
for t in tr:
    print(f"  {t.side} Entry={t.entry_price} Exit={t.exit_price} Net={t.net_return_bps:.2f}b MFE={t.peak_mfe_bps:.2f}b Reason={t.exit_reason}")

def replay_c5(d_meso_thresh=0.00, be_hurdle=3.80, be_floor=4.10):
    engine = Challenger5_DynamicFrequencyHighVelocityScalper()
    
    # Patched tick
    original_on_market_micro_tick = engine.on_market_micro_tick
    def patched_micro_tick(ts_ms, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty):
        if engine.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and engine.active_position is not None:
            pos = engine.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            mid = 0.5 * (bbo_bid + bbo_ask)
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            if unrealized_bps > pos.peak_favorable_excursion_bps:
                pos.peak_favorable_excursion_bps = unrealized_bps
            
            if pos.peak_favorable_excursion_bps >= be_hurdle and not pos.trailing_maker_armed:
                if unrealized_bps <= be_floor:
                    exit_be_p = pos.entry_price * (1.0 + sign * be_floor * 1e-4)
                    engine._close_position(
                        exit_price=exit_be_p,
                        exit_ts_ms=ts_ms,
                        exit_fee_bps=engine.maker_fee_leg,
                        reason="BREAKEVEN_MAKER_PROFIT_SHIELD",
                    )
                    return engine.state
        return original_on_market_micro_tick(ts_ms, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty)
    
    original_on_decision = engine.on_hierarchical_decision
    def patched_on_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        engine.recent_mid_prices.append((now_ms, mid))
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        if engine.state == "IDLE" and engine.ticks_seen >= 10:
            tact_dir = 2 if d_tact >= 0.11 else (0 if d_tact <= -0.11 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= d_meso_thresh) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= -d_meso_thresh)
            )

            cand_dir = tact_dir if tact_confirmed else None
            if cand_dir is None and decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= engine.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= d_meso_thresh:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= -d_meso_thresh:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if engine._is_parabolic_surge(cand_dir, now_ms, mid):
                    return engine.state

                if engine.last_exit_ts_ms > 0:
                    elapsed = (now_ms - engine.last_exit_ts_ms) / 1000.0
                    if not engine.last_exit_was_win and elapsed < 60.0:
                        return engine.state
                    elif engine.last_exit_was_win and elapsed < 12.0:
                        return engine.state

                    if elapsed < 240.0 and engine.last_trade_exit_price > 0:
                        if cand_dir == 2 and engine.last_trade_side == "BUY" and (mid > engine.last_trade_exit_price - 12.0):
                            return engine.state
                        elif cand_dir == 0 and engine.last_trade_side == "SELL" and (mid < engine.last_trade_exit_price + 12.0):
                            return engine.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir)
                engine.stalking_decision = tact_dec
                engine._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return engine.state

        engine.ticks_seen += 1
        if engine.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND"):
            engine._manage_active_swing_position(decision, bbo_bid, bbo_ask, now_ms)
        return engine.state

    engine.on_market_micro_tick = patched_micro_tick
    engine.on_hierarchical_decision = patched_on_decision

    with open(LEDGER_PATH, 'r') as f:
        bars = [json.loads(line) for line in f]

    for bar in bars:
        ts = bar["ts_ms"]
        bid = bar["bbo_bid"]
        ask = bar["bbo_ask"]
        engine.on_market_micro_tick(ts_ms=ts, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=1.0, bbo_ask_qty=1.0)
        dec = HierarchicalDecisionOutput(
            ts_ms=ts,
            macro_probs=tuple(bar["macro_probs"]),
            meso_probs=tuple(bar["meso_probs"]),
            tactical_probs=tuple(bar["tactical_probs"]),
            micro_probs=tuple(bar["micro_probs"]),
            fused_swing_probs=(0.33, 0.34, 0.33),
            primary_direction=bar["primary_direction"],
            cross_horizon_alignment=0.8,
            expected_gross_move_bps=34.0,
            expected_net_edge_bps=bar["expected_net_edge_bps"],
            meta_act_probability=0.8,
            micro_sniper_trigger_prob=0.8,
            target_holding_seconds=1800,
            take_profit_bps=30.0,
            stop_loss_bps=14.5,
            kelly_position_fraction=0.012,
            execution_mode="PASSIVE_MAKER_SNIPER"
        )
        engine.on_hierarchical_decision(dec, bid, ask, 1.0, 1.0)

    trades = engine.closed_trades
    wins = [t for t in trades if t.net_return_bps > 0]
    total_net = sum(t.net_return_bps for t in trades)
    wr = len(wins) / len(trades) * 100.0 if trades else 0.0
    return len(trades), len(wins), wr, total_net, trades

print("\n--- CHALLENGER 5 REPLAY ---")
n, w, wr, net, tr = replay_c5(d_meso_thresh=0.00, be_hurdle=3.80, be_floor=4.10)
print(f"C5 Baseline: Trades={n}, Wins={w}, WR={wr:.1f}%, Net={net:.2f} bps")
for t in tr:
    print(f"  {t.side} Entry={t.entry_price} Exit={t.exit_price} Net={t.net_return_bps:.2f}b MFE={t.peak_mfe_bps:.2f}b Reason={t.exit_reason}")

n, w, wr, net, tr = replay_c5(d_meso_thresh=0.05, be_hurdle=5.50, be_floor=4.20)
print(f"\nC5 with Unchoked BE + Meso Conviction (0.05): Trades={n}, Wins={w}, WR={wr:.1f}%, Net={net:.2f} bps")
for t in tr:
    print(f"  {t.side} Entry={t.entry_price} Exit={t.exit_price} Net={t.net_return_bps:.2f}b MFE={t.peak_mfe_bps:.2f}b Reason={t.exit_reason}")
