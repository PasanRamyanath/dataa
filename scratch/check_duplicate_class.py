import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('feature engineering scripts/execution/champion_challenger_framework.py', encoding='utf-8') as f:
    text = f.read()

count = text.count('class ConformalConfidenceFilterScalper')
print(f"Occurrences of 'class ConformalConfidenceFilterScalper': {count}")

idx = 0
for i in range(count):
    idx = text.find('class ConformalConfidenceFilterScalper', idx)
    print(f"Instance {i+1} at index {idx}:")
    print(text[idx:idx+400])
    idx += 1
