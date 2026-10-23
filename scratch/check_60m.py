import json
from pathlib import Path

lines = Path("scratch/live_hierarchical_session_ledger.jsonl").read_text(encoding="utf-8").strip().split("\n")
print(f"Total bars logged so far: {len(lines)}")

r_first = json.loads(lines[0])
r_last = json.loads(lines[-1])
print(f"Start Time: {r_first['datetime']}, Price: {r_first['bbo_mid']}")
print(f"Current Time: {r_last['datetime']}, Price: {r_last['bbo_mid']}")

modes = {}
dirs = {}
fsms = {}
for l in lines:
    r = json.loads(l)
    m = r.get("execution_mode")
    d = r.get("primary_direction")
    f = r.get("fsm_state")
    modes[m] = modes.get(m, 0) + 1
    dirs[d] = dirs.get(d, 0) + 1
    fsms[f] = fsms.get(f, 0) + 1

print("Modes:", modes)
print("Directions (0=SHORT, 1=FLAT, 2=LONG):", dirs)
print("FSM States:", fsms)
