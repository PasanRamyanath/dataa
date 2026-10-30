target_path = "feature engineering scripts/execution/champion_challenger_framework.py"
with open(target_path, "r", encoding="utf-8") as f:
    text = f.read()

old_c5_cooldown = """                    if self.last_exit_ts_ms > 0:
                        elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                        if not self.last_exit_was_win and elapsed < 60.0:
                            return self.state
                        elif self.last_exit_was_win and elapsed < 15.0:
                            return self.state

                        if elapsed < 200.0 and self.last_trade_exit_price > 0:
                            if self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - self.range_boundary_buffer_dollars):
                                return self.state"""

new_c5_cooldown = """                    if self.last_exit_ts_ms > 0:
                        elapsed = (now_ms - self.last_exit_ts_ms) / 1000.0
                        if elapsed < 200.0 and self.last_trade_exit_price > 0:
                            if self.last_trade_side == "BUY" and (mid > self.last_trade_exit_price - self.range_boundary_buffer_dollars):
                                return self.state"""

t_norm = text.replace("\r\n", "\n")
old_norm = old_c5_cooldown.replace("\r\n", "\n")
new_norm = new_c5_cooldown.replace("\r\n", "\n")

if old_norm in t_norm:
    updated_norm = t_norm.replace(old_norm, new_norm, 1)
    newline = "\r\n" if "\r\n" in text else "\n"
    if newline == "\r\n":
        updated = updated_norm.replace("\n", "\r\n")
    else:
        updated = updated_norm
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print("Successfully patched Cycle65_PureMakerMeanReversionScalper cooldown in champion_challenger_framework.py!")
else:
    print("old_norm not found!")
