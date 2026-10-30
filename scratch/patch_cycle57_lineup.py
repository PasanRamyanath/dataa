import sys

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add Cycle48_Tier05_ConformalScalper if not present
new_class = '''class Cycle48_Tier05_ConformalScalper(SteppedRatchetOFIScalper):
    """
    ⚔️ Challenger 2: Cycle 48 Tier 0.5 Conformal Scalper (Long-Only Pure)
    Pure Long-Only scalper with Tier 0.5 Stepped Ratchet (+1.25 bps net lock at +4.5 bps MFE).
    Enforces strict Long-Only entries, eliminating all short-side chop bleed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_profile = "Challenger 2: Cycle 48 Tier 0.5 Conformal Scalper (Long-Only Pure)"

'''

if 'class Cycle48_Tier05_ConformalScalper' not in content:
    target = 'class Cycle48_AdaptiveSpreadScalper'
    if target in content:
        content = content.replace(target, new_class + target)
        print("Added Cycle48_Tier05_ConformalScalper class.")
    else:
        print("Target for class insertion not found.")

# Update Challenger 2 in ChampionChallengerTournamentManager
old_c2 = '''            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle35_ConformalOFIScalper(symbol=symbol),
                strategy_profile="Challenger 2: Cycle 35 Conformal OFI Depth Scalper (Long-Only Pure)",
            ),'''

new_c2 = '''            "Challenger_2": TournamentCompetitorRecord(
                name="Challenger_2",
                engine=Cycle48_Tier05_ConformalScalper(symbol=symbol),
                strategy_profile="Challenger 2: Cycle 48 Tier 0.5 Conformal Scalper (Long-Only Pure)",
            ),'''

if old_c2 in content:
    content = content.replace(old_c2, new_c2)
    print("Updated Challenger_2 in ChampionChallengerTournamentManager.")
else:
    print("old_c2 not found, checking variations...")
    # Look for any Challenger_2 record
    import re
    pattern = r'"Challenger_2":\s*TournamentCompetitorRecord\([^)]+\),'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        print("Found match:", match.group(0))
        content = content[:match.start()] + new_c2 + content[match.end():]
        print("Replaced via regex!")

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patching complete.")
