import os
import glob
import json
import sys
import numpy as np

base_dir = 'scratch'
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith('archive_cycle') or d.startswith('archive_v3')) and os.path.isdir(os.path.join(base_dir, d))])

total_bars = 0
edges = []
p_dirs = []
d_tacts = []
d_mesos = []
d_macros = []
spreads = []

edge_ge_8 = 0
edge_ge_6 = 0
edge_ge_5 = 0
edge_ge_4 = 0

long_opps_8 = 0
short_opps_8 = 0
long_opps_5 = 0
short_opps_5 = 0

for arch in archives:
    ledgers = glob.glob(os.path.join(base_dir, arch, '*ledger*.jsonl'))
    if not ledgers: 
        continue
    with open(ledgers[0], 'r', encoding='utf-8') as f:
        bars = [json.loads(l) for l in f if l.strip().startswith('{')]
        
    total_bars += len(bars)
    for bar in bars:
        mid = bar.get('bbo_mid', bar.get('mid', 0.0))
        bid = bar.get('bbo_bid', mid - 0.05)
        ask = bar.get('bbo_ask', mid + 0.05)
        spread_bps = ((ask - bid) / mid) * 1e4 if mid > 0 else 0.0
        tact = bar.get('tactical_probs', [0.33, 0.33, 0.33])
        meso = bar.get('meso_probs', [0.33, 0.33, 0.33])
        macro = bar.get('macro_probs', [0.33, 0.33, 0.33])
        p_dir = bar.get('primary_direction', bar.get('direction', 1))
        edge = bar.get('expected_net_edge_bps', bar.get('edge', 0.0))
        
        edges.append(edge)
        p_dirs.append(p_dir)
        spreads.append(spread_bps)
        d_tact = tact[2] - tact[0]
        d_meso = meso[2] - meso[0]
        d_macro = macro[2] - macro[0]
        d_tacts.append(d_tact)
        d_mesos.append(d_meso)
        d_macros.append(d_macro)
        
        if edge >= 8.0:
            edge_ge_8 += 1
            if p_dir == 2 and d_tact >= 0.05 and d_meso >= 0.01:
                long_opps_8 += 1
            elif p_dir == 0 and d_tact <= -0.05 and d_meso <= -0.01:
                short_opps_8 += 1
                
        if edge >= 5.0:
            edge_ge_5 += 1
            if p_dir == 2 and d_tact >= 0.03 and d_meso >= 0.0:
                long_opps_5 += 1
            elif p_dir == 0 and d_tact <= -0.03 and d_meso <= 0.0:
                short_opps_5 += 1

print(f"Total Archives: {len(archives)}")
print(f"Total 10s Bars: {total_bars} (Avg {total_bars/len(archives):.1f} bars/session)")
print(f"Expected Net Edge Percentiles: 50th={np.percentile(edges, 50):.2f}b, 75th={np.percentile(edges, 75):.2f}b, 90th={np.percentile(edges, 90):.2f}b, 95th={np.percentile(edges, 95):.2f}b")
print(f"Bars with Edge >= 8.0b: {edge_ge_8} ({edge_ge_8/total_bars*100:.1f}%)")
print(f"Bars with Edge >= 5.0b: {edge_ge_5} ({edge_ge_5/total_bars*100:.1f}%)")
print(f"Primary Direction Distribution: Dir 0 (Short)={p_dirs.count(0)} ({p_dirs.count(0)/total_bars*100:.1f}%), Dir 1 (Flat)={p_dirs.count(1)} ({p_dirs.count(1)/total_bars*100:.1f}%), Dir 2 (Long)={p_dirs.count(2)} ({p_dirs.count(2)/total_bars*100:.1f}%)")
print(f"\nPotential Entry Opportunities (Edge >= 8.0b, strict concord):")
print(f"  Long Opportunities:  {long_opps_8} bars ({long_opps_8/len(archives):.1f} per session)")
print(f"  Short Opportunities: {short_opps_8} bars ({short_opps_8/len(archives):.1f} per session)")
print(f"  Total Opportunities: {long_opps_8 + short_opps_8} bars ({(long_opps_8 + short_opps_8)/len(archives):.1f} per session)")
print(f"\nPotential Entry Opportunities (Edge >= 5.0b, moderate concord):")
print(f"  Long Opportunities:  {long_opps_5} bars ({long_opps_5/len(archives):.1f} per session)")
print(f"  Short Opportunities: {short_opps_5} bars ({short_opps_5/len(archives):.1f} per session)")
print(f"  Total Opportunities: {long_opps_5 + short_opps_5} bars ({(long_opps_5 + short_opps_5)/len(archives):.1f} per session)")
