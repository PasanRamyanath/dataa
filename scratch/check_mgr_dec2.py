with open('feature engineering scripts/execution/champion_challenger_framework.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('class ChampionChallengerTournamentManager')
if idx != -1:
    idx2 = text.find('def on_hierarchical_decision', idx)
    print(text[idx2:idx2+600])
