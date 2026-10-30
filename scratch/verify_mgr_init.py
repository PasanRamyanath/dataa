import sys
from pathlib import Path
REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feature engineering scripts"))

from execution.champion_challenger_framework import ChampionChallengerTournamentManager

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")
print(f"Tournament Initialized: {len(mgr.competitors)} competitors loaded:")
for name, c in mgr.competitors.items():
    print(f"  {name:12s} | Class: {c.engine.__class__.__name__:32s} | Profile: {c.strategy_profile}")
