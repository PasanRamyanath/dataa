"""
scratch/test_range_boundary_filter.py
Tests the Range Boundary Filter + Peak Retrace Engine across Cycles 25-29
"""
import json
from pathlib import Path
from collections import deque
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

class RangeBoundaryPeakEngine:
    def __init__(
        self,
        meso_min_bps: float = 0.02,
        tactical_impulse_thresh: float = 0.14,
        retrace_barrier_dollars: float = 15.0,
        range_buffer_dollars: float = 15.0,
        model_harvest_min_gross: float = 4.50,
        maker_fee_leg: float = 2.0,
        taker_fee_leg: float = 5.0,
    ):
        self.meso_min_bps = meso_min_bps
        self.tactical_impulse_thresh = tactical_impulse_thresh
        self.retrace_barrier_dollars = retrace_barrier_dollars
        self.range_buffer_dollars = range_buffer_dollars
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
        self.rolling_15m_prices = deque(maxlen=90)
        self.ticks_seen = 0

    def _is_parabolic_surge(self, side: str, now_ms: int, mid: float) -> bool:
        dir_mult = 1.0 if side == "BUY" else -1.0
        p_30s = [p for ts, p in self.recent_mids if 20000 <= (now_ms - ts) <= 40000]
        if p_30s and dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4) > 10.0:
            return True
        p_180s = [p for ts, p in self.recent_mids if 140000 <= (now_ms - ts) <= 200000]
        if p_180s and dir_mult * ((mid - p_180s[0]) / p_180s[0] * 1e4) > 16.0:
            return True
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

        self.rolling_15m_prices.append(mid)

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

            # 1. Full Resting TP at +30.0 bps
            if unreal_bps >= 30.0:
                tp_p = p.entry_price * (1.0 + sign * 30.0 * 1e-4)
                self.close(tp_p, now_ms, self.maker_fee_leg, "RESTING_MAKER_TP")
                return

            # 2. Model Harvest TP (Peak Information Harvest)
            if unreal_bps >= self.model_harvest_min_gross:
                opp_dir = 0 if p.side == "BUY" else 2
                if (sign * micro_d) < -0.05 or stage5_dir == opp_dir:
                    exit_p = bbo_ask if p.side == "BUY" else bbo_bid
                    self.close(exit_p, now_ms, self.maker_fee_leg, "MODEL_HARVEST_MAKER_TP")
                    return

            # 3. Dynamic Trailing Profit Shield
            if p.peak_mfe_bps >= 5.50:
                p.trailing_armed = True

            if p.trailing_armed:
                mfe = p.peak_mfe_bps
                if mfe < 8.0:
                    lock_bps = max(4.10, mfe - 1.20)
                elif mfe < 12.0:
                    lock_bps = max(5.50, mfe - 1.80)
                else:
                    lock_bps = max(8.50, mfe - 2.50)

                if unreal_bps <= lock_bps:
                    exit_p = p.entry_price * (1.0 + sign * max(lock_bps, unreal_bps) * 1e-4)
                    self.close(exit_p, now_ms, self.maker_fee_leg, "MICRO_TRAILING_MAKER_TP")
                    return

            # 4. Positive Break-Even Shield:
            if p.peak_mfe_bps >= 3.80 and not p.trailing_armed:
                if unreal_bps <= 4.10:
                    exit_p = p.entry_price * (1.0 + sign * 4.10 * 1e-4)
                    self.close(exit_p, now_ms, self.maker_fee_leg, "BREAKEVEN_MAKER_PROFIT_SHIELD")
                    return

            # 5. Dual-Horizon Invalidation Cut (Underwater Risk Bounding)
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

            # 7. Maximum Deadline Taker Exit at 750s
            if elapsed_sec >= 750.0:
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

            # Strict Meso Concordance Guard
            if cand_side == "BUY" and meso_d < self.meso_min_bps:
                return
            if cand_side == "SELL" and meso_d > -self.meso_min_bps:
                return

            # Micro confirmation: require strictly positive micro-momentum
            if cand_side == "BUY" and micro_d < 0.02:
                return
            if cand_side == "SELL" and micro_d > -0.02:
                return

            # Range Boundary Filter:
            # If at least 18 bars (3 minutes) of data:
            # Block BUY within range_buffer_dollars of 15m session high
            # Block SELL within range_buffer_dollars of 15m session low
            if len(self.rolling_15m_prices) >= 18:
                sess_high = max(self.rolling_15m_prices)
                sess_low = min(self.rolling_15m_prices)
                if cand_side == "BUY" and (mid >= sess_high - self.range_buffer_dollars):
                    return
                if cand_side == "SELL" and (mid <= sess_low + self.range_buffer_dollars):
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
    print("RANGE BOUNDARY PEAK ENGINE ACROSS ALL 5 CYCLES (25-29)")
    print("=" * 95)

    tot_trades = 0
    tot_wins = 0
    tot_net = 0.0

    for c in cycles:
        ledger_path = Path(f"scratch/archive_cycle{c}/live_hierarchical_session_1h_ledger.jsonl")
        if not ledger_path.exists():
            continue

        engine = RangeBoundaryPeakEngine(
            meso_min_bps=0.02,
            tactical_impulse_thresh=0.14,
            retrace_barrier_dollars=15.0,
            range_buffer_dollars=15.0,
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
