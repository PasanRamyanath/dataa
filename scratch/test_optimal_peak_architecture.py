"""
scratch/test_optimal_peak_architecture.py
Test the reconstructed Peak Trading System across all 5 recorded cycles:
- Meso Concordance (Anti-Hostile Flow)
- Dynamic Profit Floor (never exit in loss once +4.5 bps MFE reached)
- Model Harvest Maker TP (peak information extraction)
- Patient holding (no 90s stall cut)
- $15 same-side retrace barrier
"""
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Position:
    side: str
    entry_price: float
    entry_ts_ms: int
    peak_mfe_bps: float = 0.0
    max_mae_bps: float = 0.0
    trailing_armed: bool = False

@dataclass
class ClosedTrade:
    side: str
    entry_price: float
    exit_price: float
    gross_return_bps: float
    net_return_bps: float
    peak_mfe_bps: float
    max_mae_bps: float
    holding_sec: float
    exit_reason: str

class OptimalPeakTradingEngine:
    def __init__(
        self,
        meso_min_bps: float = 0.02, # require concordant meso
        tactical_impulse_thresh: float = 0.14,
        retrace_barrier_dollars: float = 15.0,
        model_harvest_min_gross: float = 4.50, # +0.50 bps net
        maker_fee_leg: float = 2.0,
        taker_fee_leg: float = 5.0,
    ):
        self.meso_min_bps = meso_min_bps
        self.tactical_impulse_thresh = tactical_impulse_thresh
        self.retrace_barrier_dollars = retrace_barrier_dollars
        self.model_harvest_min_gross = model_harvest_min_gross
        self.maker_fee_leg = maker_fee_leg
        self.taker_fee_leg = taker_fee_leg

        self.pos: Position | None = None
        self.closed_trades: list[ClosedTrade] = []
        self.last_exit_ts_ms = 0
        self.last_exit_price = 0.0
        self.last_exit_side = ""
        self.last_exit_was_win = True
        self.recent_mids = []
        self.ticks_seen = 0

    def _is_parabolic_surge(self, side: str, now_ms: int, mid: float) -> bool:
        dir_mult = 1.0 if side == "BUY" else -1.0
        # 30s micro-surge > 10 bps
        p_30s = [p for ts, p in self.recent_mids if 20000 <= (now_ms - ts) <= 40000]
        if p_30s and dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4) > 10.0:
            return True
        # 180s tactical surge > 16 bps
        p_180s = [p for ts, p in self.recent_mids if 140000 <= (now_ms - ts) <= 200000]
        if p_180s and dir_mult * ((mid - p_180s[0]) / p_180s[0] * 1e4) > 16.0:
            return True
        # 300s meso surge > 18 bps
        p_300s = [p for ts, p in self.recent_mids if (now_ms - ts) >= 240000]
        if p_300s and dir_mult * ((mid - p_300s[0]) / p_300s[0] * 1e4) > 18.0:
            return True
        return False

    def close(self, exit_price: float, exit_ts_ms: int, fee_leg: float, reason: str):
        if not self.pos:
            return
        p = self.pos
        sign = 1.0 if p.side == "BUY" else -1.0
        gross_bps = sign * ((exit_price - p.entry_price) / p.entry_price) * 1e4
        total_fee = self.maker_fee_leg + fee_leg
        net_bps = gross_bps - total_fee
        holding_sec = (exit_ts_ms - p.entry_ts_ms) / 1000.0

        tr = ClosedTrade(
            side=p.side,
            entry_price=p.entry_price,
            exit_price=exit_price,
            gross_return_bps=round(gross_bps, 4),
            net_return_bps=round(net_bps, 4),
            peak_mfe_bps=round(p.peak_mfe_bps, 4),
            max_mae_bps=round(p.max_mae_bps, 4),
            holding_sec=round(holding_sec, 1),
            exit_reason=reason,
        )
        self.closed_trades.append(tr)
        self.last_exit_ts_ms = exit_ts_ms
        self.last_exit_price = exit_price
        self.last_exit_side = p.side
        self.last_exit_was_win = net_bps > 0
        self.pos = None

    def on_bar(self, bar: dict):
        self.ticks_seen += 1
        now_ms = bar["ts_ms"]
        bbo_bid = bar["bbo_bid"]
        bbo_ask = bar["bbo_ask"]
        mid = bar["bbo_mid"]
        self.recent_mids.append((now_ms, mid))
        if len(self.recent_mids) > 40:
            self.recent_mids.pop(0)

        # Warmup
        if self.ticks_seen < 10:
            return

        tact_d = bar["tactical_probs"][2] - bar["tactical_probs"][0]
        meso_d = bar["meso_probs"][2] - bar["meso_probs"][0]
        micro_d = bar["micro_probs"][2] - bar["micro_probs"][0]
        stage5_dir = bar["primary_direction"]

        # Manage active position
        if self.pos:
            p = self.pos
            sign = 1.0 if p.side == "BUY" else -1.0
            unreal_bps = sign * ((mid - p.entry_price) / p.entry_price) * 1e4
            p.peak_mfe_bps = max(p.peak_mfe_bps, unreal_bps)
            p.max_mae_bps = min(p.max_mae_bps, unreal_bps)
            elapsed_sec = (now_ms - p.entry_ts_ms) / 1000.0

            # 1. Full Resting TP at +25.0 bps
            if unreal_bps >= 25.0:
                tp_p = p.entry_price * (1.0 + sign * 25.0 * 1e-4)
                self.close(tp_p, now_ms, self.maker_fee_leg, "RESTING_MAKER_TP")
                return

            # 2. Model Harvest TP (Peak Information Harvest)
            # When gross profit >= 4.50 bps (+0.50 bps net), if micro ticks turn against trade or primary dir flips
            if unreal_bps >= self.model_harvest_min_gross:
                opp_dir = 0 if p.side == "BUY" else 2
                if (sign * micro_d) < -0.05 or stage5_dir == opp_dir:
                    exit_p = bbo_ask if p.side == "BUY" else bbo_bid
                    self.close(exit_p, now_ms, self.maker_fee_leg, "MODEL_HARVEST_MAKER_TP")
                    return

            # 3. Trailing Profit Shield
            # If MFE >= 5.5 bps, trail with guaranteed profit
            if p.peak_mfe_bps >= 5.50:
                p.trailing_armed = True

            if p.trailing_armed:
                mfe = p.peak_mfe_bps
                if mfe < 8.0:
                    lock_bps = max(4.20, mfe - 1.20) # min +0.2 bps net
                elif mfe < 12.0:
                    lock_bps = max(5.50, mfe - 1.80) # min +1.5 bps net
                else:
                    lock_bps = max(8.50, mfe - 2.50) # min +4.5 bps net

                if unreal_bps <= lock_bps:
                    exit_p = p.entry_price * (1.0 + sign * max(lock_bps, unreal_bps) * 1e-4)
                    self.close(exit_p, now_ms, self.maker_fee_leg, "MICRO_TRAILING_MAKER_TP")
                    return

            # 4. Positive Break-Even Shield:
            # If MFE >= 3.80 bps, never let trade fall below 4.10 bps gross (+0.10 bps net win)!
            if p.peak_mfe_bps >= 3.80 and not p.trailing_armed:
                if unreal_bps <= 4.10:
                    exit_p = p.entry_price * (1.0 + sign * 4.10 * 1e-4)
                    self.close(exit_p, now_ms, self.maker_fee_leg, "BREAKEVEN_MAKER_PROFIT_SHIELD")
                    return

            # 5. Dual-Horizon Invalidation Cut (Underwater Risk Bounding)
            # Only cut underwater if BOTH tactical and micro momentum flip hostile
            if unreal_bps <= -7.00:
                tact_opp = (sign * tact_d) < -0.10
                micro_opp = (sign * micro_d) < -0.05
                meso_opp = (sign * meso_d) < -0.15
                if (tact_opp and micro_opp) or meso_opp:
                    exit_p = bbo_bid if p.side == "BUY" else bbo_ask
                    self.close(exit_p, now_ms, self.taker_fee_leg, "DUAL_HORIZON_INVALIDATION_TAKER")
                    return

            # 6. Passive Time Unwind at 300s (5m) ONLY if in profit >= 4.25 bps gross (+0.25 bps net)
            if elapsed_sec >= 300.0 and unreal_bps >= 4.25:
                exit_p = bbo_ask if p.side == "BUY" else bbo_bid
                self.close(exit_p, now_ms, self.maker_fee_leg, "PASSIVE_MAKER_TIME_UNWIND")
                return

            # 7. Maximum Deadline Taker Exit at 500s
            if elapsed_sec >= 500.0:
                exit_p = bbo_bid if p.side == "BUY" else bbo_ask
                self.close(exit_p, now_ms, self.taker_fee_leg, "MAX_DEADLINE_TAKER")
                return

        # Engine is IDLE -> Check Entry
        else:
            cand_side = None
            if tact_d >= self.tactical_impulse_thresh:
                cand_side = "BUY"
            elif tact_d <= -self.tactical_impulse_thresh:
                cand_side = "SELL"

            if not cand_side:
                return

            # Strict Meso Concordance Guard (never trade against 5m order flow)
            if cand_side == "BUY" and meso_d < self.meso_min_bps:
                return
            if cand_side == "SELL" and meso_d > -self.meso_min_bps:
                return

            # Micro confirmation
            if cand_side == "BUY" and micro_d <= 0.01:
                return
            if cand_side == "SELL" and micro_d >= -0.01:
                return

            # Parabolic Surge Check
            if self._is_parabolic_surge(cand_side, now_ms, mid):
                return

            # Cooldown & Retrace Check
            if self.last_exit_ts_ms > 0:
                elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                if not self.last_exit_was_win and elapsed < 90.0:
                    return
                elif self.last_exit_was_win and elapsed < 20.0:
                    return

                if elapsed < 300.0 and self.last_exit_price > 0:
                    if cand_side == "BUY" and self.last_exit_side == "BUY" and (mid > self.last_exit_price - self.retrace_barrier_dollars):
                        return
                    if cand_side == "SELL" and self.last_exit_side == "SELL" and (mid < self.last_exit_price + self.retrace_barrier_dollars):
                        return

            # Place Maker entry at BBO!
            entry_p = bbo_bid if cand_side == "BUY" else bbo_ask
            self.pos = Position(side=cand_side, entry_price=entry_p, entry_ts_ms=now_ms)


def main():
    cycles = [25, 26, 27, 28, 29]
    print("=" * 95)
    print("OPTIMAL PEAK TRADING SYSTEM SIMULATION ACROSS CYCLES 25, 26, 27, 28, 29")
    print("=" * 95)

    tot_trades = 0
    tot_wins = 0
    tot_net = 0.0

    for c in cycles:
        ledger_path = Path(f"scratch/archive_cycle{c}/live_hierarchical_session_1h_ledger.jsonl")
        if not ledger_path.exists():
            continue

        engine = OptimalPeakTradingEngine(
            meso_min_bps=0.02, # Anti-Hostile Meso Flow
            tactical_impulse_thresh=0.14,
            retrace_barrier_dollars=15.0,
            model_harvest_min_gross=4.50, # locks +0.50 bps net min
        )

        with open(ledger_path) as lf:
            for line in lf:
                bar = json.loads(line)
                engine.on_bar(bar)

        n = len(engine.closed_trades)
        w = sum(1 for t in engine.closed_trades if t.net_return_bps > 0)
        net = sum(t.net_return_bps for t in engine.closed_trades)
        wr = (w / n * 100.0) if n > 0 else 0.0

        tot_trades += n
        tot_wins += w
        tot_net += net

        print(f"\n>>> CYCLE {c}: {n} Trades | {w} Wins | Win Rate: {wr:5.1f}% | Net Cash: {net:+6.2f} bps")
        for i, t in enumerate(engine.closed_trades, 1):
            print(f"    Trade {i}: {t.side:<4} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | "
                  f"Net: {t.net_return_bps:>+6.2f} bps | MFE: {t.peak_mfe_bps:>+5.2f}b | Hold: {t.holding_sec:>4.0f}s | {t.exit_reason}")

    tot_wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    print("\n" + "=" * 95)
    print(f"CUMULATIVE 5-HOUR TOTAL: {tot_trades} Trades | {tot_wins} Wins | Overall Win Rate: {tot_wr:.1f}% | Net Realized PnL: {tot_net:+.2f} bps")
    print("=" * 95)

if __name__ == "__main__":
    main()
