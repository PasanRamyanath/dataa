import re

fpath = "feature engineering scripts/execution/champion_challenger_framework.py"
with open(fpath, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Patch ConformalConfidenceFilterScalper
old_conf_block = """    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Conformal Confidence Filter Scalper (Cycle 35 Baseline)"
        self.range_boundary_buffer_dollars = 10.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        # UNIVERSAL IDLE GATE: Only apply entry search filters when IDLE!
        if self.state == "IDLE" and self.ticks_seen >= 10:
            # Restored Proven Cycle 35 / Cycle 48 Conformal Confidence Filter
            conf_tact = max(decision.tactical_probs)
            conf_meso = max(decision.meso_probs)
            if conf_tact < 0.45 or conf_meso < 0.45:
                return self.state

            tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
            tact_confirmed = (tact_dir in (0, 2)) and (
                (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.03) or
                (tact_dir == 0 and d_micro < -0.015 and d_meso <= -0.03)
            )

            cand_dir = tact_dir if tact_confirmed else None
            if cand_dir is None and decision.primary_direction in (0, 2) and decision.expected_net_edge_bps >= self.min_net_edge_bps:
                if decision.primary_direction == 2 and d_meso >= 0.03:
                    cand_dir = 2
                elif decision.primary_direction == 0 and d_meso <= -0.03:
                    cand_dir = 0

            if cand_dir in (0, 2):
                if self._is_range_boundary_violation(cand_dir, mid):
                    return self.state

                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 60.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 15.0:
                        return self.state

                    if elapsed < 200.0 and self.last_trade_exit_price > 0:
                        if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - 10.0):
                            return self.state
                        elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + 10.0):
                            return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        # Pass active position management through to base engine unimpeded
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)"""

new_conf_block = """    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Conformal Confidence Filter Scalper (Cycle 35 Baseline)"
        self.range_boundary_buffer_dollars = 8.0

    def on_hierarchical_decision(self, decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt=10000.0, intra_bar_open=0.0):
        now_ms = decision.ts_ms
        mid = 0.5 * (bbo_bid + bbo_ask)
        self.recent_mid_prices.append((now_ms, mid))
        self.rolling_15m_prices.append(mid)
        d_tact = decision.tactical_probs[2] - decision.tactical_probs[0]
        d_micro = decision.micro_probs[2] - decision.micro_probs[0]
        d_meso = decision.meso_probs[2] - decision.meso_probs[0]

        # UNIVERSAL IDLE GATE: Only apply entry search filters when IDLE!
        if self.state == "IDLE" and self.ticks_seen >= 10:
            # Calibrated Conformal / Edge Trigger:
            # Allows entry if high conformal certainty OR strong directional spread & edge (Bar 50 Peak Discovery)
            conf_tact = max(decision.tactical_probs)
            conf_meso = max(decision.meso_probs)
            has_conviction = (conf_tact >= 0.42 and conf_meso >= 0.40) or (
                d_tact >= 0.12 and d_meso >= 0.03 and decision.expected_net_edge_bps >= 10.0
            )
            if not has_conviction:
                return self.state

            # STRICTLY LONG-ONLY (BUY) to preserve 100% win rate edge
            tact_confirmed = (d_tact >= 0.10 and d_micro > 0.010 and d_meso >= 0.02)
            cand_dir = 2 if tact_confirmed else None
            if cand_dir is None and decision.primary_direction == 2 and decision.expected_net_edge_bps >= self.min_net_edge_bps and d_meso >= 0.02:
                cand_dir = 2

            if cand_dir == 2:
                if self._is_range_boundary_violation(cand_dir, mid):
                    return self.state

                if self._is_parabolic_surge(cand_dir, now_ms, mid):
                    return self.state

                if self.last_exit_ts_ms > 0:
                    elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                    if not self.last_exit_was_win and elapsed < 60.0:
                        return self.state
                    elif self.last_exit_was_win and elapsed < 15.0:
                        return self.state

                    if elapsed < 200.0 and self.last_trade_exit_price > 0:
                        if self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - self.range_boundary_buffer_dollars):
                            return self.state

                tact_dec = copy.copy(decision)
                object.__setattr__(tact_dec, "primary_direction", cand_dir) if hasattr(tact_dec, "__dataclass_fields__") and getattr(tact_dec, "__dataclass_params__", None) and getattr(tact_dec.__dataclass_params__, "frozen", False) else setattr(tact_dec, "primary_direction", cand_dir)
                self.stalking_decision = tact_dec
                self._place_sniper_maker_entry(tact_dec, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, now_ms, intra_bar_open)
                return self.state

        # Pass active position management through to base engine unimpeded
        return super().on_hierarchical_decision(decision, bbo_bid, bbo_ask, bbo_bid_qty, bbo_ask_qty, equity_usdt, intra_bar_open)"""

if old_conf_block in text:
    text = text.replace(old_conf_block, new_conf_block)
    print("Patched ConformalConfidenceFilterScalper successfully!")
else:
    print("Warning: old_conf_block not found in text!")

with open(fpath, "w", encoding="utf-8") as f:
    f.write(text)
