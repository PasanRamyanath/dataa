import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

mids = []
with open('scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        d = json.loads(line)
        mids.append((d.get('bar_idx'), d.get('bbo_mid')))

# Let's inspect rolling prices up to Bar 129
# 15m is roughly 90 bars (10s each)
bar129_idx = 128
window = [m[1] for m in mids[max(0, bar129_idx-90):bar129_idx+1]]
print(f"At Bar 129: Min price in 15m window = {min(window)}, Max = {max(window)}")
print(f"Price at Bar 129 = {mids[bar129_idx][1]}")
print(f"Distance from Low = {mids[bar129_idx][1] - min(window):.2f} USDT")
