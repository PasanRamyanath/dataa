import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    f_text = f.read()

s = f_text.find('class Cycle65_UnifiedPeakReversionEngine')
e = f_text.find('class Cycle65_DynamicSteppedRatchetScalper', s)
framework_code = f_text[s:e]

print("=== FRAMEWORK CODE ===")
print(framework_code)
