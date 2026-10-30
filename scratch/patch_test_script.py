with open("scratch/test_cycle65_with_quantile_and_wide_sl.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

insert_idx = -1
for i, l in enumerate(lines):
    if "def on_market_micro_tick" in l:
        insert_idx = i
        break

manage_lines = [
    "    def _manage_active_swing_position(self, decision, bbo_bid, bbo_ask, now_ms):\n",
    "        if self.active_position is None:\n",
    "            return\n",
    "        pos = self.active_position\n",
    "        elapsed_sec = (now_ms - pos.entry_ts_ms) / 1000.0\n",
    "        sign = 1.0 if pos.side == 'BUY' else -1.0\n",
    "        mid = 0.5 * (bbo_bid + bbo_ask)\n",
    "        unrealized_bps = sign * ((mid - pos.entry_price) / pos.entry_price) * 1e4\n",
    "\n",
    "        if not hasattr(pos, 'mfe'):\n",
    "            pos.mfe = max(0.0, unrealized_bps)\n",
    "            pos.mae = min(0.0, unrealized_bps)\n",
    "        else:\n",
    "            pos.mfe = max(pos.mfe, unrealized_bps)\n",
    "            pos.mae = min(pos.mae, unrealized_bps)\n",
    "        mfe = pos.mfe\n",
    "\n",
    "        active_floor = None\n",
    "        ratchet_reason = None\n",
    "        if mfe >= self.ratchet3_mfe_hurdle:\n",
    "            active_floor = self.ratchet3_floor_gross\n",
    "            ratchet_reason = 'TIER3_RUNNER_EXPANSION_TP'\n",
    "        elif mfe >= self.ratchet2_mfe_hurdle:\n",
    "            active_floor = self.ratchet2_floor_gross\n",
    "            ratchet_reason = 'TIER2_RUNNER_RATCHET_TP'\n",
    "        elif mfe >= self.ratchet1_mfe_hurdle:\n",
    "            active_floor = self.ratchet1_floor_gross\n",
    "            ratchet_reason = 'TIER1_STEPPED_RATCHET_TP'\n",
    "        elif mfe >= self.ratchet0_5_mfe_hurdle:\n",
    "            active_floor = self.ratchet0_5_floor_gross\n",
    "            ratchet_reason = 'TIER0_5_CYCLE48_CROWN_RATCHET_TP'\n",
    "        elif mfe >= self.shield_mfe_hurdle:\n",
    "            active_floor = self.shield_floor_gross\n",
    "            ratchet_reason = 'TIER0_BREAKEVEN_MAKER_SHIELD'\n",
    "\n",
    "        # 1. Full Target TP (+35.00 bps gross / +31.00 bps net)\n",
    "        if unrealized_bps >= self.target_tp_gross:\n",
    "            exit_p = pos.entry_price * (1.0 + sign * self.target_tp_gross * 1e-4)\n",
    "            self._close_position(exit_price=exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason='TARGET_MAKER_TP')\n",
    "            return\n",
    "\n",
    "        # 2. Retrace Floor Execution via resting Maker limit order at floor price\n",
    "        if active_floor is not None and unrealized_bps <= active_floor:\n",
    "            exit_p = pos.entry_price * (1.0 + sign * active_floor * 1e-4)\n",
    "            self._close_position(exit_price=exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason=ratchet_reason)\n",
    "            return\n",
    "\n",
    "        # 3. Profitable Time Unwind (elapsed >= 900s and gross >= 4.0b)\n",
    "        if elapsed_sec >= self.max_holding_sec and unrealized_bps >= 4.0:\n",
    "            maker_exit_p = bbo_ask if pos.side == 'BUY' else bbo_bid\n",
    "            self._close_position(exit_price=maker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.maker_fee_leg, reason='PASSIVE_MAKER_TIME_UNWIND')\n",
    "            return\n",
    "\n",
    "        # 4. FIX 1: Catastrophic Emergency Stop Breaker (Aggressive Taker Stop Market at -45.0 bps)\n",
    "        if unrealized_bps <= -45.0:\n",
    "            taker_exit_p = bbo_bid if pos.side == 'BUY' else bbo_ask\n",
    "            self._close_position(exit_price=taker_exit_p, exit_ts_ms=now_ms, exit_fee_bps=self.taker_fee_leg, reason='HARD_STOP_LOSS_TAKER')\n",
    "            return\n",
    "\n"
]

updated_lines = lines[:insert_idx] + manage_lines + lines[insert_idx:]
with open("scratch/test_cycle65_with_quantile_and_wide_sl.py", "w", encoding="utf-8") as f:
    f.writelines(updated_lines)

print("Updated scratch/test_cycle65_with_quantile_and_wide_sl.py successfully!")
