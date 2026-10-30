import os
import sys
import collections

sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from test_c63_candidates_59_cycles import Candidate1_MidpointRetestPassiveScalper, valid
from simulate_all_59_cycles_qualification import replay_cycle_ledger

loss_reasons = collections.Counter()
loss_gross = []
win_gross = []

for c in valid:
    p = os.path.join("scratch", c, "live_hierarchical_session_1h_ledger.jsonl")
    eng = Candidate1_MidpointRetestPassiveScalper()
    trades = replay_cycle_ledger(eng, p)
    for t in trades:
        if t["is_win"]:
            win_gross.append(t["gross_return_bps"])
        else:
            loss_reasons[t["exit_reason"]] += 1
            loss_gross.append(t["gross_return_bps"])

print("LOSS REASONS COUNT:")
for r, cnt in loss_reasons.most_common():
    print(f"  {r:35s}: {cnt}")

print(f"\nAverage Loss Gross Return: {sum(loss_gross)/len(loss_gross):.2f} bps" if loss_gross else "No losses")
print(f"Average Win Gross Return:  {sum(win_gross)/len(win_gross):.2f} bps" if win_gross else "No wins")
