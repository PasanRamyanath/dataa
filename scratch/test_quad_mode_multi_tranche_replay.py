import os
import glob
import json
import sys
import copy
import types
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('feature engineering scripts')

from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine, ActiveSwingPosition

class Cycle66_QuadModeMultiTranchePeakEngine(HierarchicalSniperExecutionEngine):
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Cycle66_QuadModeMultiTranchePeakEngine"
        self.maker_fee_leg = 2.0
        self.taker_fee_leg = 5.0
        self.min_warmup_tick = 10
        self.max_entry_tick = 270
        self.max_net_edge_bps = 48.0
        self.min_rng_dollars = 10.0
        self.min_rbps_short = 12.0
        self.max_queue_wait_sec = 15.0
        self.tranche_levels = [(-6.50, 2.0), (-13.50, 4.0), (-22.00, 8.0)]
        self.base_tranche_fraction = 1.0 / 15.0
        self.mid_history = []
        self.resting_entry_ts_ms = 0
        self.resting_entry_sign = 0.0
        self.pos_orig_entry_price = 0.0
        self.pos_avg_entry_price = 0.0
        self.pos_weight = 0.0
        self.pos_tranches = 0
        self.pos_mfe_bps = 0.0
        self.pos_mae_bps = 0.0
        self.ticks_seen = 0
        self._last_rolling_ts = 0

    def on_market_micro_tick(self, ts_ms: int, bbo_bid: float, bbo_ask: float, bbo_bid_qty: float, bbo_ask_qty: float, *args, **kwargs):
        ts_ms = ts_ms * 1000 if ts_ms < 10**12 else ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        if mid <= 0:
            return self.state

        if self.state == "RESTING_ENTRY_QUEUE" and self.stalking_decision is not None:
            if (ts_ms - self.resting_entry_ts_ms) / 1000.0 > self.max_queue_wait_sec:
                self.state = "IDLE"
                self.stalking_decision = None
                self.resting_entry_side = None
                self.resting_entry_price = 0.0
                return self.state

            filled = (bbo_bid <= self.resting_entry_price) if self.resting_entry_side == "BUY" else (bbo_ask >= self.resting_entry_price)
            if filled:
                ep = self.resting_entry_price
                side = self.resting_entry_side
                sign = 1.0 if side == "BUY" else -1.0
                self.active_position = ActiveSwingPosition(
                    symbol=self.symbol,
                    side=side,
                    entry_price=ep,
                    quantity=self.resting_entry_qty,
                    entry_ts_ms=ts_ms,
                    target_holding_seconds=900,
                    take_profit_bps=35.0,
                    stop_loss_bps=45.0,
                    tp_price=ep * (1.0 + sign * 0.0035),
                    sl_price=ep * (1.0 - sign * 0.0045),
                    maker_entry_fee_bps=self.maker_fee_leg,
                )
                self.pos_orig_entry_price = ep
                self.pos_avg_entry_price = ep
                self.pos_weight = 1.0
                self.pos_tranches = 1
                self.pos_mfe_bps = 0.0
                self.pos_mae_bps = 0.0
                self.state = "FILLED_SWING_HOLDING"
                self.stalking_decision = None
                return self.state

        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and self.active_position is not None:
            self._evaluate_tranche_and_ratchet_exit(ts_ms, bbo_bid, bbo_ask)

        return self.state

    def _evaluate_tranche_and_ratchet_exit(self, ts_ms: int, bbo_bid: float, bbo_ask: float):
        if self.active_position is None:
            return
        pos = self.active_position
        sign = 1.0 if pos.side == "BUY" else -1.0
        mid = 0.5 * (bbo_bid + bbo_ask)
        rem_bars = max(0, 360 - self.ticks_seen)
        orig_u_bps = sign * ((mid - self.pos_orig_entry_price) / self.pos_orig_entry_price) * 1e4

        # Scale-in loop
        while self.pos_tranches - 1 < len(self.tranche_levels):
            lvl_bps, add_w = self.tranche_levels[self.pos_tranches - 1]
            if orig_u_bps <= lvl_bps and rem_bars >= 6:
                fill_px = bbo_bid if sign == 1.0 else bbo_ask
                new_w = self.pos_weight + add_w
                self.pos_avg_entry_price = (self.pos_avg_entry_price * self.pos_weight + fill_px * add_w) / new_w
                pos.entry_price = self.pos_avg_entry_price
                pos.quantity = round(pos.quantity * (new_w / self.pos_weight), 3)
                self.pos_weight = new_w
                self.pos_tranches += 1
                self.pos_mfe_bps = 0.0
            else:
                break

        u_bps = sign * ((mid - self.pos_avg_entry_price) / self.pos_avg_entry_price) * 1e4
        self.pos_mfe_bps = max(self.pos_mfe_bps, u_bps)
        self.pos_mae_bps = min(self.pos_mae_bps, u_bps)
        pos.peak_favorable_excursion_bps = self.pos_mfe_bps
        pos.max_adverse_excursion_bps = self.pos_mae_bps
        elapsed_s = (ts_ms - pos.entry_ts_ms) / 1000.0

        active_floor, reason = (None, "")
        if self.pos_tranches > 1:
            if self.pos_mfe_bps >= 12.00:
                active_floor, reason = (9.50, "SCALE_R2_MAKER_TP")
            elif self.pos_mfe_bps >= 8.00:
                active_floor, reason = (6.50, "SCALE_R1_MAKER_TP")
            elif self.pos_mfe_bps >= 5.50:
                active_floor, reason = (4.60, "SCALE_BE_SHIELD")
            elif self.pos_mfe_bps >= 4.15 and elapsed_s >= 120.0:
                active_floor, reason = (4.08, "SCALE_LATE_BE_SHIELD")
        else:
            if self.pos_mfe_bps >= 22.00:
                active_floor, reason = (18.50, "TIER3_RUNNER_EXPANSION_TP")
            elif self.pos_mfe_bps >= 16.00:
                active_floor, reason = (13.00, "TIER2_RUNNER_RATCHET_TP")
            elif self.pos_mfe_bps >= 11.00:
                active_floor, reason = (8.50, "TIER1_STEPPED_RATCHET_TP")
            elif self.pos_mfe_bps >= 6.50:
                active_floor, reason = (5.25, "TIER0_5_CYCLE48_CROWN_RATCHET_TP")
            elif self.pos_mfe_bps >= 4.15 and elapsed_s >= 180.0:
                active_floor, reason = (4.08, "TIER0_LATE_BREAKEVEN_SHIELD")

        # Exits
        if u_bps >= 35.00:
            exit_p = self.pos_avg_entry_price * (1.0 + sign * 0.0035)
            self._close_position(exit_price=exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.maker_fee_leg, reason="TARGET_MAKER_TP")
            return
        elif active_floor is not None and u_bps <= active_floor:
            exit_p = self.pos_avg_entry_price * (1.0 + sign * active_floor * 1e-4)
            self._close_position(exit_price=exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.maker_fee_leg, reason=reason)
            return
        elif self.pos_tranches > 1 and u_bps >= 6.50:
            exit_p = self.pos_avg_entry_price * (1.0 + sign * u_bps * 1e-4)
            self._close_position(exit_price=exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.maker_fee_leg, reason="SCALE_TARGET_MAKER_TP")
            return
        elif (elapsed_s >= 900.0 and u_bps >= 4.15) or (rem_bars <= 3 and u_bps >= 4.02):
            exit_p = self.pos_avg_entry_price * (1.0 + sign * u_bps * 1e-4)
            self._close_position(exit_price=exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.maker_fee_leg, reason="PASSIVE_MAKER_TIME_UNWIND")
            return
        elif orig_u_bps <= -45.00:
            taker_exit_p = bbo_bid if sign == 1.0 else bbo_ask
            self._close_position(exit_price=taker_exit_p, exit_ts_ms=ts_ms, exit_fee_bps=self.taker_fee_leg, reason="HARD_STOP_LOSS_TAKER")
            return

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms * 1000 if decision.ts_ms < 10**12 else decision.ts_ms
        self.ticks_seen += 1
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.mid_history.append(mid)

        if self._last_rolling_ts != now_ms:
            self._last_rolling_ts = now_ms
            self.rolling_15m_prices.append(mid)

        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND"):
            self._evaluate_tranche_and_ratchet_exit(now_ms, bbo_bid, bbo_ask)

        if self.state == "IDLE":
            if len(self.mid_history) < 11:
                return self.state
            if self.ticks_seen > self.max_entry_tick:
                return self.state
            if decision.expected_net_edge_bps > self.max_net_edge_bps:
                return self.state

            window = self.mid_history[-91:]
            sh = max(window)
            sl = min(window)
            rng = sh - sl
            if rng < self.min_rng_dollars:
                return self.state

            rbps = (rng / mid) * 1e4

            for sign in (1.0, -1.0):
                if sign == -1.0 and rbps < self.min_rbps_short:
                    continue

                dt = sign * (decision.tactical_probs[2] - decision.tactical_probs[0])
                dm = sign * (decision.meso_probs[2] - decision.meso_probs[0])
                dmac = sign * (decision.macro_probs[2] - decision.macro_probs[0])
                dmic = sign * (decision.micro_probs[2] - decision.micro_probs[0])
                pct = (mid - sl) / rng if sign == 1.0 else (sh - mid) / rng

                r60 = sign * ((mid - self.mid_history[-7]) / self.mid_history[-7]) * 1e4 if len(self.mid_history) >= 7 else 0.0
                r180 = sign * ((mid - self.mid_history[-19]) / self.mid_history[-19]) * 1e4 if len(self.mid_history) >= 19 else 0.0
                r300 = sign * ((mid - self.mid_history[-31]) / self.mid_history[-31]) * 1e4 if len(self.mid_history) >= 31 else 0.0

                p_ok = (decision.primary_direction == 2 and sign == 1.0) or (decision.primary_direction == 0 and sign == -1.0)

                m1 = (pct <= 0.40 and rbps >= 8.0 and dm >= 0.08 and dmac >= 0.02 and dmic >= -0.02)
                m2 = (p_ok and decision.expected_net_edge_bps >= 2.0 and pct <= 0.74 and dt >= 0.05 and dm >= 0.02 and dmac >= -0.02 and dmic >= -0.02 and r60 <= 8.0 and r180 <= (99.0 if sign == 1.0 else 8.0) and r300 <= 12.0)
                m3 = (pct <= 0.72 and rbps >= 6.0 and dt >= 0.10 and dm >= 0.03 and dmac >= 0.00 and dmic >= 0.00 and r60 <= 5.5 and r180 <= 8.0 and r300 <= 8.0)

                m4a = (sign == 1.0 and p_ok and self.ticks_seen <= 240 and pct <= 0.96 and rbps >= 14.0 and dt >= 0.16 and dm >= 0.195 and dmac >= 0.05 and dmic >= 0.00 and r60 <= 5.5 and r300 <= 12.0)
                m4b = (sign == 1.0 and p_ok and self.ticks_seen <= 240 and pct <= 0.96 and rbps >= 7.5 and dt >= 0.16 and dm >= 0.08 and dmac >= 0.30 and dmic >= 0.04 and r60 <= 5.5 and r300 <= 5.0)
                m4 = m4a or m4b

                if m1 or m2 or m3 or m4:
                    cand_dir = 2 if sign == 1.0 else 0
                    tact_dec = copy.copy(decision)
                    if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False):
                        object.__setattr__(tact_dec, "primary_direction", cand_dir)
                        object.__setattr__(tact_dec, "execution_mode", "PASSIVE_MAKER_SNIPER")
                        object.__setattr__(tact_dec, "kelly_position_fraction", max(0.10, decision.kelly_position_fraction) * self.base_tranche_fraction)
                    else:
                        setattr(tact_dec, "primary_direction", cand_dir)
                        setattr(tact_dec, "execution_mode", "PASSIVE_MAKER_SNIPER")
                        setattr(tact_dec, "kelly_position_fraction", max(0.10, getattr(decision, "kelly_position_fraction", 0.15)) * self.base_tranche_fraction)

                    self.stalking_decision = tact_dec
                    self.resting_entry_ts_ms = now_ms
                    self.resting_entry_sign = sign
                    self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                    break

        return self.state


if __name__ == "__main__":
    base_dir = "scratch"
    archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])
    all_closed_trades = []
    unclosed_trades = []
    active_sessions = 0
    tranche_counts = Counter()

    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers:
            continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(line) for line in f if line.strip().startswith("{")]

        engine = Cycle66_QuadModeMultiTranchePeakEngine(symbol="BTCUSDT")

        for idx, bar in enumerate(bars):
            mid = bar.get("bbo_mid", bar.get("mid", 0.0))
            bid = bar.get("bbo_bid", mid - 0.05)
            ask = bar.get("bbo_ask", mid + 0.05)
            bid_qty = bar.get("bbo_bid_qty", 10.0)
            ask_qty = bar.get("bbo_ask_qty", 10.0)
            ts_ms = bar.get("ts_ms", idx * 10000)

            tact = bar.get("tactical_probs", [0.33, 0.33, 0.33])
            meso = bar.get("meso_probs", [0.33, 0.33, 0.33])
            macro = bar.get("macro_probs", [0.33, 0.33, 0.33])
            micro = bar.get("micro_probs", [0.33, 0.33, 0.33])
            p_dir = bar.get("primary_direction", bar.get("direction", 1))
            edge = bar.get("expected_net_edge_bps", bar.get("edge", 0.0))

            decision = types.SimpleNamespace(
                ts_ms=ts_ms,
                primary_direction=p_dir,
                expected_net_edge_bps=edge,
                tactical_probs=tact,
                meso_probs=meso,
                macro_probs=macro,
                micro_probs=micro,
                kelly_position_fraction=0.15,
                execution_mode="PASSIVE_MAKER_SNIPER",
                target_holding_seconds=900,
                take_profit_bps=35.0,
                stop_loss_bps=45.0,
                micro_sniper_trigger_prob=0.85
            )

            # 1. Market micro tick
            engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

            # 2. Hierarchical decision
            engine.on_hierarchical_decision(decision=decision, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, equity_usdt=10000.0, intra_bar_open=mid)

            # 3. Post-decision fill evaluation if resting
            if engine.state == "RESTING_ENTRY_QUEUE":
                engine.on_market_micro_tick(ts_ms=ts_ms+1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask+0.05, bar_low=bid-0.05, current_decision=decision)

        # Honest Session-End Mark-to-Market check
        if engine.active_position is not None:
            last_mid = bars[-1].get("bbo_mid", bars[-1].get("mid", 0.0))
            pos = engine.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            u_bps = sign * ((last_mid - engine.pos_avg_entry_price) / engine.pos_avg_entry_price) * 1e4
            net_bps = u_bps - 4.0  # 4.0 bps round trip maker
            unclosed_trades.append({
                "cycle": arch, "side": pos.side, "tranches": engine.pos_tranches, "weight": engine.pos_weight,
                "net": net_bps, "gross": u_bps
            })
            engine._close_position(exit_price=last_mid, exit_ts_ms=bars[-1].get("ts_ms", 3600000), exit_fee_bps=2.0, reason="SESSION_END_MARK_TO_MARKET")

        session_trades = engine.closed_trades
        if session_trades:
            active_sessions += 1
            for t in session_trades:
                all_closed_trades.append({
                    "cycle": arch, "side": t.side, "net": t.net_return_bps, "reason": t.exit_reason,
                    "gross": t.gross_return_bps, "dur": t.holding_duration_sec,
                    "weight": getattr(t, "pos_weight", 1.0)
                })

    n = len(all_closed_trades)
    w = sum(1 for t in all_closed_trades if t["net"] > 0)
    l = n - w
    wr = (w / n * 100.0) if n > 0 else 0.0
    tot_pnl = sum(t["net"] for t in all_closed_trades)
    avg_pnl = (tot_pnl / n) if n > 0 else 0.0

    print("=" * 90)
    print("CYCLE 66 QUAD-MODE MULTI-TRANCHE REPLAY ACROSS ALL 64 HISTORICAL CYCLES")
    print("=" * 90)
    print(f"Total Completed Positions: {n} across {active_sessions} active sessions ({n/max(1, active_sessions):.2f}/sess)")
    print(f"Wins: {w} | Losses: {l} | Win Rate: {wr:.2f}%")
    print(f"Net Realized PnL: {tot_pnl:+.2f} bps | Avg Profit: {avg_pnl:+.2f} bps/trade")
    print(f"Unclosed Stranded Positions at Session End: {len(unclosed_trades)}")
    for uc in unclosed_trades:
        print(f"  Stranded: {uc}")
    print("=" * 90)
