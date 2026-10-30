with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'def register_default_competitors' in line or 'ChampionChallengerTournamentManager' in line:
        for j in range(i, min(len(lines), i+60)):
            print(f'{j+1}: {lines[j]}', end='')
        break
