import os

path = "feature engineering scripts/execution/champion_challenger_framework.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Normalize line endings
text_norm = text.replace("\r\n", "\n")

target_init = """        self.shield_mfe_hurdle = 3.80
        self.shield_floor_gross = 4.15
        self.ratchet1_mfe_hurdle = 6.00
        self.ratchet1_floor_gross = 6.25
        self.ratchet2_mfe_hurdle = 10.00
        self.ratchet2_floor_gross = 8.50
        self.target_tp_gross = 35.00"""

replacement_init = """        self.shield_mfe_hurdle = 3.80
        self.shield_floor_gross = 4.15
        self.ratchet0_5_mfe_hurdle = 4.50 # Cycle 48 Historic Crown Feature (+1.25 bps net floor)
        self.ratchet0_5_floor_gross = 5.25
        self.ratchet1_mfe_hurdle = 6.00
        self.ratchet1_floor_gross = 6.25
        self.ratchet2_mfe_hurdle = 10.00
        self.ratchet2_floor_gross = 8.50
        self.target_tp_gross = 35.00"""

target_logic = """        # Determine Active Ratchet Floor
        active_floor = None
        ratchet_reason = None
        if mfe >= self.ratchet2_mfe_hurdle:
            active_floor = self.ratchet2_floor_gross
            ratchet_reason = "TIER2_RUNNER_RATCHET_TP"
        elif mfe >= self.ratchet1_mfe_hurdle:
            active_floor = self.ratchet1_floor_gross
            ratchet_reason = "TIER1_STEPPED_RATCHET_TP"
        elif mfe >= self.shield_mfe_hurdle:
            active_floor = self.shield_floor_gross
            ratchet_reason = "TIER0_BREAKEVEN_MAKER_SHIELD" """

# Also match stripped
target_logic_clean = """        # Determine Active Ratchet Floor
        active_floor = None
        ratchet_reason = None
        if mfe >= self.ratchet2_mfe_hurdle:
            active_floor = self.ratchet2_floor_gross
            ratchet_reason = "TIER2_RUNNER_RATCHET_TP"
        elif mfe >= self.ratchet1_mfe_hurdle:
            active_floor = self.ratchet1_floor_gross
            ratchet_reason = "TIER1_STEPPED_RATCHET_TP"
        elif mfe >= self.shield_mfe_hurdle:
            active_floor = self.shield_floor_gross
            ratchet_reason = "TIER0_BREAKEVEN_MAKER_SHIELD" """

replacement_logic = """        # Determine Active Ratchet Floor (Integrated Cycle 48 + Cycle 43 + Cycle 31 Hierarchy)
        active_floor = None
        ratchet_reason = None
        if mfe >= self.ratchet2_mfe_hurdle:
            active_floor = self.ratchet2_floor_gross
            ratchet_reason = "TIER2_RUNNER_RATCHET_TP"
        elif mfe >= self.ratchet1_mfe_hurdle:
            active_floor = self.ratchet1_floor_gross
            ratchet_reason = "TIER1_STEPPED_RATCHET_TP"
        elif mfe >= self.ratchet0_5_mfe_hurdle:
            active_floor = self.ratchet0_5_floor_gross
            ratchet_reason = "TIER0_5_CYCLE48_CROWN_RATCHET_TP"
        elif mfe >= self.shield_mfe_hurdle:
            active_floor = self.shield_floor_gross
            ratchet_reason = "TIER0_BREAKEVEN_MAKER_SHIELD" """

if target_init in text_norm:
    text_norm = text_norm.replace(target_init, replacement_init)
    print("Init replaced.")
else:
    print("Init target not found.")

if target_logic.strip() in text_norm:
    text_norm = text_norm.replace(target_logic.strip(), replacement_logic.strip())
    print("Logic replaced.")
else:
    print("Logic target not found.")

with open(path, "w", encoding="utf-8") as f:
    f.write(text_norm)
print("Updated framework with Cycle 48 Tier 0.5 ratchet!")
