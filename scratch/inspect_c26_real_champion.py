"""
scratch/inspect_c26_real_champion.py
Find out why real Champion in Cycle 26 took only 1 trade and avoided the fake entries.
"""
import json

with open("scratch/archive_cycle26/live_hierarchical_session_1h_ledger.jsonl") as f:
    bars = [json.loads(line) for line in f]

print("Total bars in C26:", len(bars))

# Check bars where d_tact >= 0.14
candidate_bars = []
for b in bars:
    tact_d = b["tactical_probs"][2] - b["tactical_probs"][0]
    meso_d = b["meso_probs"][2] - b["meso_probs"][0]
    micro_d = b["micro_probs"][2] - b["micro_probs"][0]
    stage5_dir = b["primary_direction"]
    edge = b["expected_net_edge_bps"]
    mode = b.get("execution_mode", "")
    fsm = b.get("fsm_state", "")
    pos = b.get("active_position")
    
    if abs(tact_d) >= 0.14:
        candidate_bars.append((b["bar_idx"], b["datetime"][:19], b["bbo_mid"], tact_d, meso_d, micro_d, stage5_dir, edge, fsm, pos is not None))

print(f"{'Bar':<5}{'Time':<20}{'Mid':<10}{'Tact d':<10}{'Meso d':<10}{'Micro d':<10}{'S5 Dir':<8}{'Edge':<8}{'FSM':<12}{'InPos'}")
print("-" * 105)
for cb in candidate_bars[:25]:
    print(f"{cb[0]:<5}{cb[1]:<20}{cb[2]:<10.1f}{cb[3]:<+10.2f}{cb[4]:<+10.2f}{cb[5]:<+10.2f}{cb[6]:<8}{cb[7]:<8.2f}{cb[8]:<12}{cb[9]}")
