import os
import glob
import json

base = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base, d))])

schema_types = {}

for a in archives:
    p = os.path.join(base, a)
    ledgers = glob.glob(os.path.join(p, "*ledger*.jsonl"))
    if not ledgers:
        continue
    ledger_path = ledgers[0]
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("{"):
                    data = json.loads(line)
                    keys = tuple(sorted(list(data.keys())))
                    # check key features
                    has_bbo = "bbo_mid" in data or "mid" in data or "price" in data
                    has_probs = "tactical_probs" in data or "tactical" in data
                    has_dir = "primary_direction" in data or "direction" in data
                    has_edge = "expected_net_edge_bps" in data or "edge" in data
                    
                    schema_id = f"bbo:{has_bbo}_probs:{has_probs}_dir:{has_dir}_edge:{has_edge}"
                    if schema_id not in schema_types:
                        schema_types[schema_id] = []
                    schema_types[schema_id].append(a)
                    break
    except Exception as e:
        print(f"Error reading {a}: {e}")

print("Identified Schema Archetypes across 64 cycle archives:")
for sid, arch_list in schema_types.items():
    print(f"\nArchetype '{sid}' ({len(arch_list)} cycles):")
    print(f"  Examples: {arch_list[:5]} ... (total {len(arch_list)})")
