import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('feature engineering scripts')

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    text = f.read()

start = text.find('class Cycle65_UnifiedPeakReversionEngine')
end = text.find('class Cycle65_DynamicSteppedRatchetScalper')
print(text[start:end])
