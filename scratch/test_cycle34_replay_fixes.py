"""
scratch/test_cycle34_replay_fixes.py
Simulates Cycle 34 with:
1. Stale Order Cancellation (cancel if dir != resting_dir or age > 15s)
2. Strict Trade-Through Limit Fill (eff_low <= limit for BUY, eff_high >= limit for SELL)
3. 15m Range Boundary Filter ($12 from 15m high/low)
4. Unchoked Break-Even Shield (MFE >= 5.50 bps, lock at 4.25 bps gross)
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

class SniperEngineSimulator:
    def __init__(
        self,
        cancel_stale_on_flat: bool = True,
        max_resting_sec: float = 15.0,
        strict_fill: bool = True,
        enable_range_boundary: bool = True,
        range_buffer_dollars: float = 12.0,
        tactical_impulse_thresh: float = 0.14,
        meso_min_bps: float = 0.04,
        retrace_barrier_dollars: float = 12.0,
    ):
        self.cancel_stale_on_flat = cancel_stale_on_flat
        self.max_resting_sec = max_resting_sec
        self.strict_fill = strict_fill
        self.enable_range_boundary = enable_range_boundary
        self.range_buffer_dollars = range_buffer_dollars
        self.tactical_impulse_thresh = tactical_impulse_thresh
        self.meso_min_bps = meso_min_bps
        self.retrace_barrier_dollars = retrace_barrier_dollars

        self.maker_fee = 2.0
        self.taker_fee = 5.0

        self.state = "IDLE"
        self.pos: Position | None = None
        self.closed_trades: list[ClosedTrade] = []

        # Resting order state
        self.resting_side = None
        self.resting_price = 0.0
        self.resting_ts_ms = 0
        self.resting_expected_dir = 1

        self.last_exit_ts_ms = 0
        self.last_exit_price = 0.0
        self.last_exit_side = ""
        self.last_exit_was_win = True

        self.recent_mids = []
        self.rolling_15m_prices = deque(maxlen=90)
        self.ticks_seen = 0

    def close(self, exit_price: float, exit_ts_ms: int, fee_leg: float, reason: str):
        if not self.pos:
            return
        p = self.pos
        sign = 1.0 if p.side == "BUY" else -1.0
        gross_bps = sign * ((exit_price - p.entry_price) / p.entry_price) * 1e4
        total_fee = self.maker_fee + fee_leg
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
        self.state = "IDLE"

    def on_bar(self, bar: dict):
        self.ticks_seen += 1
        now_ms = bar["ts_ms"]
        bbo_bid = bar["bbo_bid"]
        bbo_ask = bar["bbo_ask"]
        mid = bar["bbo_mid"]
        eff_high = bar.get("bar_high", bbo_ask)
        eff_low = bar.get("bar_low", bbo_bid)

        self.recent_mids.append((now_ms, mid))
        if len(self.recent_mids) > 40:
            self.recent_mids.pop(0)

        self.rolling_15m_prices.append(mid)

        tact_d = bar["tactical_probs"][2] - bar["tactical_probs"][0]
        meso_d = bar["meso_probs"][2] - bar["meso_probs"][0]
        micro_d = bar["micro_probs"][2] - bar["micro_probs"][0]
        primary_dir = bar["primary_direction"]

        # Case 1: Resting Entry Queue
        if self.state == "RESTING_ENTRY_QUEUE":
            # Cancellation check
            opp_dir = 0 if self.resting_side == "BUY" else 2
            cancel = False

            if self.cancel_stale_on_flat:
                if primary_dir != self.resting_expected_dir:
                    cancel = True
            else:
                if primary_dir == opp_dir:
                    cancel = True

            resting_sec = (now_ms - self.resting_ts_ms) / 1000.0
            if resting_sec > self.max_resting_sec:
                cancel = True

            if cancel:
                self.state = "IDLE"
                self.resting_side = None
                return

            # Fill check
            filled = False
            if self.resting_side == "BUY":
                if self.strict_fill:
                    if eff_low <= self.resting_price:
                        filled = True
                else:
                    if eff_low <= self.resting_price * 1.0012:
                        filled = True
            else:
                if self.strict_fill:
                    if eff_high >= self.resting_price:
                        filled = True
                else:
                    if eff_high >= self.resting_price * 0.9988:
                        filled = True

            if filled:
                self.pos = Position(side=self.resting_side, entry_price=self.resting_price, entry_ts_ms=now_ms)
                self.state = "FILLED_SWING_HOLDING"
                self.resting_side = None
                return
            return

        # Case 2: Active Swing Holding
        if self.state == "FILLED_SWING_HOLDING":
            p = self.pos
            sign = 1.0 if p.side == "BUY" else -1.0
            unreal_bps = sign * ((mid - p.entry_price) / p.entry_price) * 1e4
            p.peak_mfe_bps = max(p.peak_mfe_bps, unreal_bps)
            p.max_mae_bps = min(p.max_mae_bps, unreal_bps)
            elapsed_sec = (now_ms - p.entry_ts_ms) / 1000.0

            # 1. Full TP at +30 bps
            if unreal_bps >= 30.0:
                tp_p = p.entry_price * (1.0 + sign * 30.0 * 1e-4)
                self.close(tp_p, now_ms, self.maker_fee, "RESTING_MAKER_TP")
                return

            # 2. Model Harvest TP (when in profit >= 4.5 bps and micro momentum stalls)
            if unreal_bps >= 4.50:
                opp_dir = 0 if p.side == "BUY" else 2
                if (sign * micro_d) < -0.05 or primary_dir == opp_dir:
                    exit_p = bbo_ask if p.side == "BUY" else bbo_bid
                    self.close(exit_p, now_ms, self.maker_fee, "MODEL_HARVEST_MAKER_TP")
                    return

            # 3. Dynamic Trailing Profit Shield
            if p.peak_mfe_bps >= 5.50:
                p.trailing_armed = True

            if p.trailing_armed:
                mfe = p.peak_mfe_bps
                if mfe < 8.0:
                    lock_bps = max(4.25, mfe - 1.20)
                elif mfe < 12.0:
                    lock_bps = max(5.50, mfe - 1.80)
                else:
                    lock_bps = max(8.50, mfe - 2.50)

                if unreal_bps <= lock_bps:
                    exit_p = p.entry_price * (1.0 + sign * max(lock_bps, unreal_bps) * 1e-4)
                    self.close(exit_p, now_ms, self.maker_fee, "MICRO_TRAILING_MAKER_TP")
                    return

            # 4. Unchoked Break-Even Shield: only arm when MFE >= 5.50 bps; exit if drops to <= 4.25 bps
            if p.peak_mfe_bps >= 5.50 and not p.trailing_armed:
                if unreal_bps <= 4.25:
                    exit_p = p.entry_price * (1.0 + sign * 4.25 * 1e-4)
                    self.close(exit_p, now_ms, self.maker_fee, "BREAKEVEN_MAKER_PROFIT_SHIELD")
                    return

            # 5. Dual-Horizon Invalidation (Responsive Model Exit)
            opp_dir = 0 if p.side == "BUY" else 2
            if primary_dir == opp_dir or (sign * meso_d) < -0.10:
                exit_p = bbo_ask if p.side == "BUY" else bbo_bid
                self.close(exit_p, now_ms, self.maker_fee, "RESPONSIVE_MODEL_MAKER_EXIT")
                return

            # 6. Hard stop loss
            if unreal_bps <= -14.50:
                exit_p = bbo_bid if p.side == "BUY" else bbo_ask
                self.close(exit_p, now_ms, self.taker_fee, "HARD_STOP_LOSS")
                return
            return

        # Case 3: IDLE -> Check Entry
        if self.ticks_seen < 10:
            return

        cand_side = None
        if tact_d >= self.tactical_impulse_thresh and meso_d >= self.meso_min_bps:
            cand_side = "BUY"
        elif tact_d <= -self.tactical_impulse_thresh and meso_d <= -self.meso_min_bps:
            cand_side = "SELL"

        if not cand_side:
            return

        # Micro confirmation
        if cand_side == "BUY" and micro_d < 0.02:
            return
        if cand_side == "SELL" and micro_d > -0.02:
            return

        # 15m Range Boundary Filter
        if self.enable_range_boundary and len(self.rolling_15m_prices) >= 18:
            sess_high = max(self.rolling_15m_prices)
            sess_low = min(self.rolling_15m_prices)
            if cand_side == "BUY" and (mid >= sess_high - self.range_buffer_dollars):
                return
            if cand_side == "SELL" and (mid <= sess_low + self.range_buffer_dollars):
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

        # Queue resting maker order at BBO
        self.state = "RESTING_ENTRY_QUEUE"
        self.resting_side = cand_side
        self.resting_price = bbo_bid if cand_side == "BUY" else bbo_ask
        self.resting_ts_ms = now_ms
        self.resting_expected_dir = 2 if cand_side == "BUY" else 0


def main():
    ledger_path = Path("scratch/archive_cycle34/live_hierarchical_session_1h_ledger.jsonl")
    if not ledger_path.exists():
        print(f"Error: {ledger_path} not found")
        return

    bars = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line))

    print(f"Loaded {len(bars)} bars from Cycle 34 ledger.")

    configs = [
        ("Cycle 34 Baseline (Flawed Stale Queue, 1.0012 mult, No Range Filter)", False, 999.0, False, False, 12.0),
        ("+ Fix Stale Queue Eviction (Cancel on FLAT or age > 15s)", True, 15.0, False, False, 12.0),
        ("+ Fix Strict Limit Trade-Through (eff_low <= limit)", True, 15.0, True, False, 12.0),
        ("+ Deploy 15m Range Boundary Filter ($12 buffer)", True, 15.0, True, True, 12.0),
        ("+ Deploy 15m Range Boundary Filter ($15 buffer)", True, 15.0, True, True, 15.0),
    ]

    for label, cancel_flat, max_rest, strict_f, range_b, range_buf in configs:
        sim = SniperEngineSimulator(
            cancel_stale_on_flat=cancel_flat,
            max_resting_sec=max_rest,
            strict_fill=strict_f,
            enable_range_boundary=range_b,
            range_buffer_dollars=range_buf,
        )
        for b in bars:
            sim.on_bar(b)

        n_trades = len(sim.closed_trades)
        wins = sum(1 for t in sim.closed_trades if t.net_return_bps > 0)
        wr = (wins / n_trades * 100) if n_trades > 0 else 0.0
        tot_net = sum(t.net_return_bps for t in sim.closed_trades)

        print("-" * 80)
        print(f"CONFIG: {label}")
        print(f"  Trades: {n_trades} | Wins: {wins} | Win Rate: {wr:.1f}% | Net PnL: {tot_net:+.2f} bps")
        for i, t in enumerate(sim.closed_trades, 1):
            print(f"    T{i}: {t.side} @ {t.entry_price} -> {t.exit_price} | Net: {t.net_return_bps:+.2f} bps | MFE: {t.peak_mfe_bps:+.2f}b | Hold: {t.holding_sec}s | Reason: {t.exit_reason}")

if __name__ == "__main__":
    main()
