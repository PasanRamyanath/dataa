import json
from pathlib import Path

p = Path("scratch/live_hierarchical_session_ledger.jsonl")
lines = p.read_text(encoding="utf-8").strip().split("\n")
records = [json.loads(l) for l in lines]
p_start = records[0]["bbo_mid"]
p_end = records[-1]["bbo_mid"]
high_p = max(r["bbo_mid"] for r in records)
low_p = min(r["bbo_mid"] for r in records)
ret_bps = (p_end - p_start) / p_start * 10000.0

print(f"Start Mid: ${p_start:.2f}")
print(f"End Mid:   ${p_end:.2f}")
print(f"High Mid:  ${high_p:.2f}")
print(f"Low Mid:   ${low_p:.2f}")
print(f"Net Excursion: {ret_bps:+.2f} bps ({p_end - p_start:+.2f} USD)")
print(f"Max Favorable Excursion (Short): +{(p_start - low_p)/p_start*10000:.2f} bps ({p_start - low_p:.2f} USD)")
print(f"Max Adverse Excursion (Short):   -{(high_p - p_start)/p_start*10000:.2f} bps ({high_p - p_start:.2f} USD)")
