target_path = "feature engineering scripts/execution/champion_challenger_framework.py"
with open(target_path, "r", encoding="utf-8") as f:
    text = f.read()

old_tick = """        # 2. Position Management during micro-tick
        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and self.active_position is not None:
            decision = kwargs.get("current_decision", None)
            self._manage_active_swing_position(decision, bbo_bid, bbo_ask, ts_ms)

        return self.state"""

new_tick = """        # 2. Intra-bar Catastrophic Emergency Breaker during micro-tick
        if self.state in ("FILLED_SWING_HOLDING", "PASSIVE_MAKER_UNWIND") and self.active_position is not None:
            pos = self.active_position
            sign = 1.0 if pos.side == "BUY" else -1.0
            mid = 0.5 * (bbo_bid + bbo_ask)
            unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4
            if unrealized_bps <= -45.0:
                taker_exit_p = bbo_bid if pos.side == "BUY" else bbo_ask
                self._close_position(
                    exit_price=taker_exit_p,
                    exit_ts_ms=ts_ms,
                    exit_fee_bps=self.taker_fee_leg,
                    reason="HARD_STOP_LOSS_TAKER"
                )
                return self.state

        return self.state"""

t_norm = text.replace("\r\n", "\n")
old_norm = old_tick.replace("\r\n", "\n")
new_norm = new_tick.replace("\r\n", "\n")

if old_norm in t_norm:
    updated_norm = t_norm.replace(old_norm, new_norm, 1)
    newline = "\r\n" if "\r\n" in text else "\n"
    if newline == "\r\n":
        updated = updated_norm.replace("\n", "\r\n")
    else:
        updated = updated_norm
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print("Successfully patched on_market_micro_tick in champion_challenger_framework.py!")
else:
    print("old_norm not found!")
