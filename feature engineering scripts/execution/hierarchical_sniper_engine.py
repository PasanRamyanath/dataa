"""
feature engineering scripts/execution/hierarchical_sniper_engine.py
9-State Hierarchical Multi-Horizon Maker Sniper Execution Engine.
Anchors Take-Profit (28-140 bps) and Stop-Loss (16-80 bps) to the Tier 3 Meso
Swing Horizon (15m-2h) while executing entries and exits on the Tier 1 (1s/10s)
microstructure tape via Post-Only (GTX) limit orders with FIFO queue tracking
and 50% intra-bar exhaustion wick retracement placement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


SniperState = Literal[
    "IDLE",
    "MACRO_VETOED",
    "MESO_SIGNAL_ARMED",
    "TACTICAL_PULLBACK_CONFIRMED",
    "MICRO_SNIPER_STALKING",
    "RESTING_MAKER_ENTRY",
    "FILLED_SWING_HOLDING",
    "DYNAMIC_TRAILING_MAKER_EXIT",
    "EMERGENCY_STOP_TAKER_EXIT",
]


@dataclass
class MultiHorizonSignalPacket:
    timestamp_ms: int
    tier4_macro_probs: np.ndarray        # [p_down, p_flat, p_up]
    tier4_hurst_4h: float
    tier4_funding_bps: float
    tier3_meso_probs: np.ndarray         # [p_down, p_flat, p_up]
    tier3_conformal_singleton: bool
    tier3_atr_15m_bps: float
    tier3_atr_1h_bps: float
    tier2_tactical_probs: np.ndarray     # [p_down, p_flat, p_up]
    tier2_pullback_zscore: float
    tier2_meta_act_prob: float
    tier1_micro_probs: np.ndarray        # [p_down, p_flat, p_up]
    tier1_spot_1s_lead_bps: float
    tier1_l1_imbalance: float
    tier1_vpin_toxicity: float
    bbo_bid: float
    bbo_ask: float
    bbo_bid_qty: float
    bbo_ask_qty: float
    intra_bar_open: float = 0.0


@dataclass
class ActiveSwingPosition:
    side: Literal["LONG", "SHORT"]
    entry_price: float
    quantity_btc: float
    entry_ts_ms: int
    target_holding_seconds: int
    tp_price: float
    sl_price: float
    initial_tp_bps: float
    initial_sl_bps: float
    peak_favorable_excursion_bps: float = 0.0
    trailing_stop_bps: float | None = None
    resting_tp_order_id: str | None = None


class HierarchicalSniperExecutionEngine:
    """
    9-State Hierarchical Maker Execution Engine combining Macro/Meso Swing Barriers
    with 1s/10s GTX Post-Only Microstructure Sniper Entry & Exit.
    """

    def __init__(
        self,
        equity_usd: float = 100_000.0,
        max_leverage: float = 3.0,
        kelly_fraction: float = 0.25,
        meta_act_threshold: float = 0.58,
        min_alignment_score: float = 0.30,
        max_stalking_window_sec: int = 180,
        maker_fee_bps: float = 2.0,
        taker_fee_bps: float = 5.0,
    ) -> None:
        self.equity_usd = equity_usd
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        self.meta_act_threshold = meta_act_threshold
        self.min_alignment_score = min_alignment_score
        self.max_stalking_window_sec = max_stalking_window_sec
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps

        self.state: SniperState = "IDLE"
        self.stalking_started_ms: int = 0
        self.pending_side: Literal["LONG", "SHORT"] | None = None
        self.pending_entry_price: float = 0.0
        self.pending_qty_btc: float = 0.0
        self.pending_queue_ahead_btc: float = 0.0
        self.pending_reprice_count: int = 0
        self.active_position: ActiveSwingPosition | None = None
        self.closed_trades: list[dict] = []

    def compute_cross_tier_alignment(self, pkt: MultiHorizonSignalPacket) -> tuple[float, Literal["LONG", "SHORT", "FLAT"]]:
        e4 = float(pkt.tier4_macro_probs[2] - pkt.tier4_macro_probs[0])
        e3 = float(pkt.tier3_meso_probs[2] - pkt.tier3_meso_probs[0])
        e2 = float(pkt.tier2_tactical_probs[2] - pkt.tier2_tactical_probs[0])

        if abs(e3) < 0.14 or not pkt.tier3_conformal_singleton:
            return 0.0, "FLAT"

        side: Literal["LONG", "SHORT"] = "LONG" if e3 > 0 else "SHORT"
        sign = 1.0 if side == "LONG" else -1.0

        if pkt.tier4_hurst_4h > 0.55 and (sign * e4) < -0.15:
            return -1.0, "FLAT"

        alignment = sign * (0.25 * e4 + 0.45 * e3 + 0.30 * e2)
        return float(alignment), side

    def compute_swing_barriers_and_size(
        self,
        pkt: MultiHorizonSignalPacket,
        side: Literal["LONG", "SHORT"],
        alignment_score: float,
    ) -> tuple[float, float, float, int]:
        composite_atr_bps = max(0.60 * pkt.tier3_atr_15m_bps + 0.40 * (pkt.tier3_atr_1h_bps * 0.5), 14.0)
        tp_bps = max(1.85 * composite_atr_bps, 28.0)
        sl_bps = max(1.15 * composite_atr_bps, 16.0)

        p_win = float(np.clip(pkt.tier2_meta_act_prob, 0.50, 0.88))
        net_tp = tp_bps - 2.0 * self.maker_fee_bps
        net_sl = sl_bps + self.maker_fee_bps + self.taker_fee_bps
        b_odds = net_tp / max(net_sl, 1.0)

        raw_kelly = max(0.0, p_win - (1.0 - p_win) / b_odds)
        vol_scalar = min(1.25, 25.0 / max(pkt.tier3_atr_15m_bps, 10.0))

        sign = 1.0 if side == "LONG" else -1.0
        funding_penalty = 0.65 if (sign * pkt.tier4_funding_bps > 3.0) else 1.0

        frac = min(self.max_leverage, self.kelly_fraction * raw_kelly * vol_scalar * funding_penalty * (1.0 + alignment_score))
        mid = 0.5 * (pkt.bbo_bid + pkt.bbo_ask)
        qty_btc = round((self.equity_usd * frac) / max(mid, 1.0), 3)

        hold_sec = 3600 if pkt.tier4_hurst_4h > 0.58 else 1800
        return tp_bps, sl_bps, qty_btc, hold_sec

    def on_10s_tick(self, pkt: MultiHorizonSignalPacket) -> dict:
        mid = 0.5 * (pkt.bbo_bid + pkt.bbo_ask)

        if self.active_position is not None:
            return self._manage_open_swing_position(pkt, mid)

        alignment, side = self.compute_cross_tier_alignment(pkt)
        if alignment < self.min_alignment_score or side == "FLAT":
            self.state = "MACRO_VETOED" if alignment < 0 else "IDLE"
            self.pending_side = None
            return {"action": "NONE", "state": self.state, "alignment": round(alignment, 4)}

        if pkt.tier2_meta_act_prob < self.meta_act_threshold:
            self.state = "MESO_SIGNAL_ARMED"
            return {"action": "WAIT_META_GATE", "state": self.state, "meta_prob": round(pkt.tier2_meta_act_prob, 4)}

        if self.state in ("IDLE", "MACRO_VETOED", "MESO_SIGNAL_ARMED", "TACTICAL_PULLBACK_CONFIRMED"):
            self.state = "MICRO_SNIPER_STALKING"
            self.stalking_started_ms = pkt.timestamp_ms
            self.pending_side = side
            self.pending_reprice_count = 0

        if self.state == "MICRO_SNIPER_STALKING":
            elapsed_sec = (pkt.timestamp_ms - self.stalking_started_ms) / 1000.0
            if elapsed_sec > self.max_stalking_window_sec:
                self.state = "IDLE"
                self.pending_side = None
                return {"action": "STALKING_EXPIRED", "state": self.state}

            sign = 1.0 if side == "LONG" else -1.0
            e1_micro = float(pkt.tier1_micro_probs[2] - pkt.tier1_micro_probs[0])
            micro_aligned = (
                (sign * e1_micro >= -0.05)
                and (sign * pkt.tier1_l1_imbalance >= -0.10)
                and (sign * pkt.tier1_spot_1s_lead_bps >= -0.50)
                and (pkt.tier1_vpin_toxicity < 0.78)
            )
            if micro_aligned:
                tp_bps, sl_bps, qty_btc, _ = self.compute_swing_barriers_and_size(pkt, side, alignment)
                if qty_btc >= 0.001:
                    self.state = "RESTING_MAKER_ENTRY"
                    # Upgrade 3 (50% intra-bar exhaustion wick retracement limit price)
                    raw_bbo_px = pkt.bbo_bid if side == "LONG" else pkt.bbo_ask
                    if pkt.intra_bar_open > 0:
                        intra_bps = abs(mid - pkt.intra_bar_open) / pkt.intra_bar_open * 1e4
                        if intra_bps > 1.2 * max(pkt.tier3_atr_15m_bps / 3.0, 2.0):
                            raw_bbo_px = round(0.5 * (pkt.intra_bar_open + mid), 1)
                    self.pending_entry_price = raw_bbo_px
                    self.pending_qty_btc = qty_btc
                    self.pending_queue_ahead_btc = pkt.bbo_bid_qty if side == "LONG" else pkt.bbo_ask_qty
                    return {
                        "action": "POST_GTX_MAKER_ENTRY",
                        "side": side,
                        "price": self.pending_entry_price,
                        "qty_btc": qty_btc,
                        "tp_bps": round(tp_bps, 2),
                        "sl_bps": round(sl_bps, 2),
                        "state": self.state,
                    }
            return {"action": "STALKING_MICRO_DIP", "state": self.state, "side": side}

        if self.state == "RESTING_MAKER_ENTRY" and self.pending_side is not None:
            side = self.pending_side
            sign = 1.0 if side == "LONG" else -1.0
            if (sign * pkt.tier1_l1_imbalance < -0.42) or (pkt.tier1_vpin_toxicity > 0.85):
                self.pending_reprice_count += 1
                if self.pending_reprice_count > 3:
                    self.state = "IDLE"
                    self.pending_side = None
                    return {"action": "CANCEL_ABORT_TOXICITY", "state": self.state}
                self.pending_entry_price = (pkt.bbo_bid - 0.10) if side == "LONG" else (pkt.bbo_ask + 0.10)
                return {"action": "REPRICE_GTX_MAKER", "price": self.pending_entry_price, "state": self.state}

            filled = (side == "LONG" and pkt.bbo_ask <= self.pending_entry_price + 0.05) or (
                side == "SHORT" and pkt.bbo_bid >= self.pending_entry_price - 0.05
            )
            if filled:
                tp_bps, sl_bps, qty_btc, hold_sec = self.compute_swing_barriers_and_size(pkt, side, alignment)
                entry_px = self.pending_entry_price
                tp_px = entry_px * (1.0 + sign * tp_bps * 1e-4)
                sl_px = entry_px * (1.0 - sign * sl_bps * 1e-4)
                self.active_position = ActiveSwingPosition(
                    side=side,
                    entry_price=entry_px,
                    quantity_btc=self.pending_qty_btc,
                    entry_ts_ms=pkt.timestamp_ms,
                    target_holding_seconds=hold_sec,
                    tp_price=tp_px,
                    sl_price=sl_px,
                    initial_tp_bps=tp_bps,
                    initial_sl_bps=sl_bps,
                )
                self.state = "FILLED_SWING_HOLDING"
                self.pending_side = None
                return {
                    "action": "MAKER_ENTRY_FILLED_POST_TP",
                    "side": side,
                    "entry_price": entry_px,
                    "tp_price": round(tp_px, 2),
                    "sl_price": round(sl_px, 2),
                    "state": self.state,
                }
        return {"action": "HOLD_RESTING_ORDER", "state": self.state}

    def _manage_open_swing_position(self, pkt: MultiHorizonSignalPacket, mid: float) -> dict:
        pos = self.active_position
        assert pos is not None
        sign = 1.0 if pos.side == "LONG" else -1.0
        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
        pos.peak_favorable_excursion_bps = max(pos.peak_favorable_excursion_bps, unrealized_bps)
        held_sec = (pkt.timestamp_ms - pos.entry_ts_ms) / 1000.0

        if unrealized_bps >= pos.initial_tp_bps:
            net_bps = pos.initial_tp_bps - 2.0 * self.maker_fee_bps
            self._record_close(pos, pkt.timestamp_ms, mid, net_bps, "MAKER_TP_HIT")
            return {"action": "EXIT_MAKER_TP_FILLED", "net_bps": round(net_bps, 2), "state": "IDLE"}

        if unrealized_bps <= -pos.initial_sl_bps:
            net_bps = -pos.initial_sl_bps - (self.maker_fee_bps + self.taker_fee_bps + 0.5)
            self._record_close(pos, pkt.timestamp_ms, mid, net_bps, "TAKER_SL_HIT")
            return {"action": "EXIT_TAKER_SL", "net_bps": round(net_bps, 2), "state": "IDLE"}

        if pos.peak_favorable_excursion_bps >= 0.60 * pos.initial_tp_bps:
            self.state = "DYNAMIC_TRAILING_MAKER_EXIT"
            lock_bps = max(6.0, 0.55 * pos.peak_favorable_excursion_bps)
            pos.trailing_stop_bps = max(pos.trailing_stop_bps or -999.0, lock_bps)
            if unrealized_bps <= pos.trailing_stop_bps:
                net_bps = unrealized_bps - 2.0 * self.maker_fee_bps
                self._record_close(pos, pkt.timestamp_ms, mid, net_bps, "TRAILING_MAKER_LOCK")
                return {"action": "EXIT_TRAILING_MAKER_PROFIT", "net_bps": round(net_bps, 2), "state": "IDLE"}

        e3 = float(pkt.tier3_meso_probs[2] - pkt.tier3_meso_probs[0])
        if (sign * e3 < -0.25) or (held_sec >= 0.85 * pos.target_holding_seconds):
            fee = 2.0 * self.maker_fee_bps if held_sec < pos.target_holding_seconds else (self.maker_fee_bps + self.taker_fee_bps)
            net_bps = unrealized_bps - fee
            self._record_close(pos, pkt.timestamp_ms, mid, net_bps, "MESO_REVERSAL_OR_TIME_UNWIND")
            return {"action": "EXIT_PASSIVE_MAKER_UNWIND", "net_bps": round(net_bps, 2), "state": "IDLE"}

        return {
            "action": "HOLDING_SWING_POSITION",
            "side": pos.side,
            "unrealized_bps": round(unrealized_bps, 2),
            "peak_mfe_bps": round(pos.peak_favorable_excursion_bps, 2),
            "held_sec": round(held_sec, 1),
            "state": self.state,
        }

    def _record_close(self, pos: ActiveSwingPosition, ts_ms: int, exit_px: float, net_bps: float, reason: str) -> None:
        pnl_usd = pos.quantity_btc * pos.entry_price * (net_bps * 1e-4)
        self.equity_usd += pnl_usd
        self.closed_trades.append({
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_px,
            "qty_btc": pos.quantity_btc,
            "net_bps": round(net_bps, 3),
            "pnl_usd": round(pnl_usd, 2),
            "duration_sec": round((ts_ms - pos.entry_ts_ms) / 1000.0, 1),
            "reason": reason,
        })
        self.active_position = None
        self.state = "IDLE"
