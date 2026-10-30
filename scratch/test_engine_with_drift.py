import os
import glob
import json
import io
import sys
import types

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('feature engineering scripts')

from execution.champion_challenger_framework import Cycle66_QuadModeMultiTranchePeakEngine

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

class RefinedCycle66WithDrift(Cycle66_QuadModeMultiTranchePeakEngine):
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(symbol=symbol)
        self.session_open_mid = None

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms * 1000 if decision.ts_ms < 10**12 else decision.ts_ms
        self.ticks_seen += 1
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.mid_history.append(mid)
        if self.session_open_mid is None:
            self.session_open_mid = mid

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
            session_drift_bps = ((mid - self.session_open_mid) / self.session_open_mid) * 1e4

            for sign in (1.0, -1.0):
                # Drift Filter: don't buy if market is falling hard, don't short if market is rallying hard
                if sign == 1.0 and session_drift_bps < -12.0:
                    continue
                if sign == -1.0 and session_drift_bps > 12.0:
                    continue

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
                m3 = (pct <= 0.72 and rbps >= 6.0 and dt >= 0.10 and (dm >= 0.08 if not p_ok else dm >= 0.03) and dmac >= 0.00 and dmic >= 0.00 and r60 <= 5.5 and r180 <= 8.0 and r300 <= 8.0)

                m4a = (sign == 1.0 and p_ok and self.ticks_seen <= 240 and pct <= 0.96 and rbps >= 14.0 and dt >= 0.16 and dm >= 0.195 and dmac >= 0.05 and dmic >= 0.00 and r60 <= 5.5 and r300 <= 12.0)
                m4b = (sign == 1.0 and p_ok and self.ticks_seen <= 240 and pct <= 0.96 and rbps >= 7.5 and dt >= 0.16 and dm >= 0.08 and dmac >= 0.30 and dmic >= 0.04 and r60 <= 5.5 and r300 <= 5.0)
                m4 = m4a or m4b

                if m1 or m2 or m3 or m4:
                    cand_dir = 2 if sign == 1.0 else 0
                    import copy
                    tact_dec = copy.copy(decision)
                    if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False):
                        object.__setattr__(tact_dec, "primary_direction", cand_dir)
                        object.__setattr__(tact_dec, "execution_mode", "PASSIVE_MAKER_SNIPER")
                        object.__setattr__(tact_dec, "kelly_position_fraction", max(0.10, getattr(decision, "kelly_position_fraction", 0.15)) * self.base_tranche_fraction)
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

all_closed_trades = []
loss_trades = []

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
    if not ledgers:
        continue
    with open(ledgers[0], "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f if line.strip().startswith("{")]

    engine = RefinedCycle66WithDrift(symbol="BTCUSDT")

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

        engine.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask + 0.05, bar_low=bid - 0.05, current_decision=decision)
        engine.on_hierarchical_decision(decision=decision, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, equity_usdt=10000.0, intra_bar_open=mid)
        if engine.state == "RESTING_ENTRY_QUEUE":
            engine.on_market_micro_tick(ts_ms=ts_ms + 1000, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty, bar_high=ask + 0.05, bar_low=bid - 0.05, current_decision=decision)

    if engine.active_position is not None:
        last_mid = bars[-1].get("bbo_mid", bars[-1].get("mid", 0.0))
        pos = engine.active_position
        sign = 1.0 if pos.side == "BUY" else -1.0
        avg_ep = getattr(engine, "pos_avg_entry_price", pos.entry_price) or pos.entry_price
        u_bps = sign * ((last_mid - avg_ep) / avg_ep) * 1e4
        net_m2m = u_bps - 4.0
        engine._close_position(exit_price=last_mid, exit_ts_ms=bars[-1].get("ts_ms", 3600000), exit_fee_bps=2.0, reason="SESSION_END_MARK_TO_MARKET")

    for t in engine.closed_trades:
        rec = {
            "cycle": arch,
            "side": t.side,
            "gross": t.gross_return_bps,
            "net": t.net_return_bps,
            "reason": t.exit_reason
        }
        all_closed_trades.append(rec)
        if t.net_return_bps <= 0:
            loss_trades.append(rec)

n = len(all_closed_trades)
w = sum(1 for t in all_closed_trades if t["net"] > 0)
l = n - w
wr = (w / n * 100.0) if n > 0 else 0.0
tot_pnl = sum(t["net"] for t in all_closed_trades)
print(f"Results with Session Drift Filter:")
print(f"Total={n} | Wins={w} | Losses={l} | WR={wr:.2f}% | Net PnL={tot_pnl:+.2f} bps")
print(f"Losses: {[t['cycle'] + '_' + t['side'] + '(' + str(round(t['net'], 1)) + 'b, ' + t['reason'] + ')' for t in loss_trades]}")
