import re

path = 'feature engineering scripts/execution/champion_challenger_framework.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Cycle66 __init__
old_init = '''    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Supreme Champion: Cycle 66 Quad-Mode Multi-Tranche Peak Engine (100% WR, +348.74 bps Net, 0 Losses)"
        self.maker_fee_leg = 2.0
        self.taker_fee_leg = 5.0
        self.min_warmup_tick = 10
        self.max_entry_tick = 270'''

new_init = '''    def __init__(self, symbol: str = "BTCUSDT", session_bars: int = 0, max_entry_tick: float = float("inf")):
        super().__init__(symbol=symbol)
        self.strategy_profile = "Supreme Champion: Cycle 66 Quad-Mode Multi-Tranche Peak Engine (100% WR, +348.74 bps Net, 0 Losses)"
        self.maker_fee_leg = 2.0
        self.taker_fee_leg = 5.0
        self.min_warmup_tick = 10
        self.session_bars = session_bars
        self.max_entry_tick = max_entry_tick'''

# 2. Update rem_bars
old_rem = '''        rem_bars = max(0, 360 - self.ticks_seen)'''
new_rem = '''        rem_bars = max(0, self.session_bars - self.ticks_seen) if self.session_bars > 0 else 999'''

# 3. Update Mode 4 tick condition
old_m4 = '''                m4a = (sign == 1.0 and p_ok and self.ticks_seen <= 240 and pct <= 0.96 and rbps >= 14.0 and dt >= 0.16 and dm >= 0.195 and dmac >= 0.05 and dmic >= 0.00 and r60 <= 5.5 and r300 <= 12.0)
                m4b = (sign == 1.0 and p_ok and self.ticks_seen <= 240 and pct <= 0.96 and rbps >= 7.5 and dt >= 0.16 and dm >= 0.08 and dmac >= 0.30 and dmic >= 0.04 and r60 <= 5.5 and r300 <= 5.0)'''

new_m4 = '''                m4_tick_ok = (self.session_bars == 0 or self.ticks_seen <= max(240, self.session_bars - 60))
                m4a = (sign == 1.0 and p_ok and m4_tick_ok and pct <= 0.96 and rbps >= 14.0 and dt >= 0.16 and dm >= 0.195 and dmac >= 0.05 and dmic >= 0.00 and r60 <= 5.5 and r300 <= 12.0)
                m4b = (sign == 1.0 and p_ok and m4_tick_ok and pct <= 0.96 and rbps >= 7.5 and dt >= 0.16 and dm >= 0.08 and dmac >= 0.30 and dmic >= 0.04 and r60 <= 5.5 and r300 <= 5.0)'''

# 4. Cap mid_history length
old_hist = '''        self.mid_history.append(mid)

        if self._last_rolling_ts != now_ms:'''
new_hist = '''        self.mid_history.append(mid)
        if len(self.mid_history) > 360:
            self.mid_history = self.mid_history[-360:]

        if self._last_rolling_ts != now_ms:'''

# 5. TournamentManager __init__
old_tourn = '''    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Cycle66_QuadModeMultiTranchePeakEngine(symbol=symbol),'''

new_tourn = '''    def __init__(self, symbol: str = "BTCUSDT", session_bars: int = 0):
        self.symbol = symbol
        self.session_bars = session_bars
        self.competitors: dict[str, TournamentCompetitorRecord] = {
            "Champion": TournamentCompetitorRecord(
                name="Champion",
                engine=Cycle66_QuadModeMultiTranchePeakEngine(symbol=symbol, session_bars=session_bars, max_entry_tick=float("inf")),'''

assert old_init in content, "old_init not found"
content = content.replace(old_init, new_init, 1)

assert old_rem in content, "old_rem not found"
content = content.replace(old_rem, new_rem, 1)

assert old_m4 in content, "old_m4 not found"
content = content.replace(old_m4, new_m4, 1)

assert old_hist in content, "old_hist not found"
content = content.replace(old_hist, new_hist, 1)

assert old_tourn in content, "old_tourn not found"
content = content.replace(old_tourn, new_tourn, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully applied cutoff removal to champion_challenger_framework.py")
