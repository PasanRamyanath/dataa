from execution.champion_challenger_framework import ChampionChallengerTournamentManager

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")
print(f"Tournament manager initialized successfully with {len(mgr.competitors)} competitors:")
for name, c in mgr.competitors.items():
    print(f"  {name}: {type(c.engine).__name__} | {c.strategy_profile[:60]}")
