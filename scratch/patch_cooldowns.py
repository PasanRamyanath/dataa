target_path = "feature engineering scripts/execution/champion_challenger_framework.py"
with open(target_path, "r", encoding="utf-8") as f:
    text = f.read()

target_str = """                    # Post-Trade Settlement & Same-Side Retrace Protection
                    if self.last_exit_ts_ms > 0:
                        elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                        if not self.last_exit_was_win and elapsed < 60.0:
                            return self.state
                        elif self.last_exit_was_win and elapsed < 15.0:
                            return self.state

                        if elapsed < 200.0 and self.last_trade_exit_price > 0:
                            if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - self.range_boundary_buffer_dollars):
                                return self.state
                            elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + self.range_boundary_buffer_dollars):
                                return self.state"""

replacement_str = """                    # Post-Trade Settlement & Same-Side Retrace Protection ($10 spacing within 200s)
                    if self.last_exit_ts_ms > 0:
                        elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                        if elapsed < 200.0 and self.last_trade_exit_price > 0:
                            if cand_dir == 2 and self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - self.range_boundary_buffer_dollars):
                                return self.state
                            elif cand_dir == 0 and self.last_trade_side == "SELL" and (mid < self.last_trade_exit_price + self.range_boundary_buffer_dollars):
                                return self.state"""

# Normalize
t_norm = text.replace("\r\n", "\n")
target_norm = target_str.replace("\r\n", "\n")
repl_norm = replacement_str.replace("\r\n", "\n")

if target_norm in t_norm:
    updated_norm = t_norm.replace(target_norm, repl_norm, 1)
    newline = "\r\n" if "\r\n" in text else "\n"
    if newline == "\r\n":
        updated = updated_norm.replace("\n", "\r\n")
    else:
        updated = updated_norm
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print("Successfully patched cooldowns in champion_challenger_framework.py!")
else:
    print("target_norm not found!")
