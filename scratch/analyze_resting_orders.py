import json
from pathlib import Path

p = Path("scratch/live_hierarchical_session_ledger.jsonl")
lines = p.read_text(encoding="utf-8").strip().split("\n")
records = [json.loads(l) for l in lines]

resting_events = []
current_event = []

for i, r in enumerate(records):
    f = r.get("fsm_state")
    if f == "RESTING_ENTRY_QUEUE":
        current_event.append((i+1, r))
    else:
        if current_event:
            resting_events.append((current_event, r.get("fsm_state")))
            current_event = []

if current_event:
    resting_events.append((current_event, "SESSION_END"))

print(f"Total resting order events: {len(resting_events)}")
for idx, (event_bars, exit_state) in enumerate(resting_events[:10]):
    start_bar = event_bars[0][0]
    end_bar = event_bars[-1][0]
    dur_sec = len(event_bars) * 10
    start_p = event_bars[0][1]["bbo_mid"]
    end_p = event_bars[-1][1]["bbo_mid"]
    exit_p = records[min(len(records)-1, end_bar)]["bbo_mid"]
    print(f"Event #{idx+1:2d} (Bars {start_bar:3d}..{end_bar:3d}, {dur_sec:2d}s): StartMid=${start_p:.2f} -> EndMid=${end_p:.2f} | Exit State: {exit_state} | NextMid=${exit_p:.2f}")
