with open("scratch/test_cycle65_with_quantile_and_wide_sl.py", "r", encoding="utf-8") as f:
    text = f.read()

s = text.find("class Cycle65_UnifiedPeakReversionEngineFixed")
e = text.find("base_dir = ", s)
exact_engine_code = text[s:e].replace("Cycle65_UnifiedPeakReversionEngineFixed", "Cycle65_UnifiedPeakReversionEngine")

# Read target framework
with open("feature engineering scripts/execution/champion_challenger_framework.py", "r", encoding="utf-8") as f:
    f_text = f.read()

f_s = f_text.find("class Cycle65_UnifiedPeakReversionEngine")
f_e = f_text.find("class Cycle65_DynamicSteppedRatchetScalper", f_s)

updated_framework = f_text[:f_s] + exact_engine_code + f_text[f_e:]

with open("feature engineering scripts/execution/champion_challenger_framework.py", "w", encoding="utf-8") as f:
    f.write(updated_framework)

print("Transplanted exact verified Cycle65_UnifiedPeakReversionEngine into framework!")
