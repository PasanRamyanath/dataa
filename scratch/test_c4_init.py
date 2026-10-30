import sys
sys.path.insert(0, "feature engineering scripts")
from execution.champion_challenger_framework import ChampionChallengerTournamentManager
import json

mgr = ChampionChallengerTournamentManager()
c4 = mgr.competitors["Challenger_4"].engine
print("C4 type:", type(c4))
print("C4 strategy_profile:", c4.strategy_profile)
print("C4 min_net_edge_bps:", c4.min_net_edge_bps)
