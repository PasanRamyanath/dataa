import sys, os, glob, json, re
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def get_cycle_num(p):
    m = re.search(r'archive_cycle(\d+)', p)
    return int(m.group(1)) if m else 999

all_ledgers = sorted(
    glob.glob('scratch/archive_cycle*/live_hierarchical_session_1h_ledger.jsonl'),
    key=get_cycle_num
)

print(f"Loaded {len(all_ledgers)} cycle ledgers.")

# Collect all bars with their forward excursions
# We will compute forward Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE)
# for both BUY (LONG) and SELL (SHORT) over a 30-bar horizon (5 minutes).

long_samples = []
short_samples = []

for l_path in all_ledgers:
    c_num = get_cycle_num(l_path)
    bars = []
    with open(l_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line))
    
    n_bars = len(bars)
    for i in range(15, n_bars - 30):
        b = bars[i]
        mid = float(b['bbo_mid'])
        if mid <= 0:
            continue
        
        # Future 30 bars (5 mins)
        future_mids = [float(bars[i + k]['bbo_mid']) for k in range(1, 31)]
        max_future = max(future_mids)
        min_future = min(future_mids)
        
        long_mfe = ((max_future - mid) / mid) * 1e4
        long_mae = ((min_future - mid) / mid) * 1e4
        
        short_mfe = ((mid - min_future) / mid) * 1e4
        short_mae = ((mid - max_future) / mid) * 1e4
        
        # Compute past displacements
        disp_1m = ((mid - float(bars[i - 6]['bbo_mid'])) / float(bars[i - 6]['bbo_mid'])) * 1e4
        disp_5m = ((mid - float(bars[i - 30]['bbo_mid'])) / float(bars[i - 30]['bbo_mid'])) * 1e4 if i >= 30 else 0.0
        
        tact = b.get('tactical_probs', [0.33, 0.34, 0.33])
        meso = b.get('meso_probs', [0.33, 0.34, 0.33])
        macro = b.get('macro_probs', [0.33, 0.34, 0.33])
        micro = b.get('micro_probs', [0.33, 0.34, 0.33])
        
        d_tact = tact[2] - tact[0]
        d_meso = meso[2] - meso[0]
        d_macro = macro[2] - macro[0]
        d_micro = micro[2] - micro[0]
        
        edge = float(b.get('expected_net_edge_bps', 0.0))
        pdir = int(b.get('primary_direction', 1))
        mode = str(b.get('execution_mode', 'NONE'))
        
        sample = {
            'cycle': c_num,
            'bar_idx': i,
            'mid': mid,
            'd_macro': d_macro,
            'd_meso': d_meso,
            'd_tact': d_tact,
            'd_micro': d_micro,
            'disp_1m': disp_1m,
            'disp_5m': disp_5m,
            'edge': edge,
            'pdir': pdir,
            'mode': mode,
            'long_mfe': long_mfe,
            'long_mae': long_mae,
            'short_mfe': short_mfe,
            'short_mae': short_mae,
        }
        
        if d_tact > 0.05:
            long_samples.append(sample)
        if d_tact < -0.05:
            short_samples.append(sample)

print(f"Collected {len(long_samples)} long candidates and {len(short_samples)} short candidates.")

# Test filtering criteria for Longs
print("\n--- EVALUATING LONG FILTERS ---")
filters_long = [
    ("Baseline: d_tact > 0.10, d_meso > 0.02", lambda s: s['d_tact'] > 0.10 and s['d_meso'] > 0.02),
    ("Edge >= 10 bps", lambda s: s['d_tact'] > 0.10 and s['d_meso'] > 0.02 and s['edge'] >= 10.0),
    ("Edge >= 10 bps + disp_1m >= 0 (No falling knife)", lambda s: s['d_tact'] > 0.10 and s['d_meso'] > 0.02 and s['edge'] >= 10.0 and s['disp_1m'] >= 0.0),
    ("Edge >= 10 bps + disp_1m >= 0 + disp_5m >= 0", lambda s: s['d_tact'] > 0.10 and s['d_meso'] > 0.02 and s['edge'] >= 10.0 and s['disp_1m'] >= 0.0 and s['disp_5m'] >= 0.0),
    ("Edge >= 12 bps + disp_1m >= 0 + d_macro >= 0", lambda s: s['d_tact'] > 0.12 and s['d_meso'] > 0.04 and s['edge'] >= 12.0 and s['disp_1m'] >= 0.0 and s['d_macro'] >= 0.0),
    ("Quad-Horizon Coherence: Macro>0, Meso>0.04, Tact>0.12, Edge>=12", lambda s: s['d_macro'] > 0.02 and s['d_meso'] > 0.04 and s['d_tact'] > 0.12 and s['edge'] >= 12.0 and s['disp_1m'] >= 0.0),
]

for name, flt in filters_long:
    matched = [s for s in long_samples if flt(s)]
    if not matched:
        continue
    # Win = MFE >= 5.5 bps before MAE <= -6.0 bps
    wins = sum(1 for s in matched if s['long_mfe'] >= 5.5 and abs(s['long_mae']) < 6.0)
    # Reached +4.5 MFE
    reached_45 = sum(1 for s in matched if s['long_mfe'] >= 4.5)
    # Severe loss: MAE <= -10.0 before reaching +4.5
    severe_loss = sum(1 for s in matched if s['long_mae'] <= -10.0 and s['long_mfe'] < 4.5)
    avg_mfe = sum(s['long_mfe'] for s in matched) / len(matched)
    avg_mae = sum(s['long_mae'] for s in matched) / len(matched)
    print(f"{name:65s} | Count: {len(matched):5d} | MFE>=4.5: {reached_45/len(matched)*100:5.1f}% | Avg MFE: {avg_mfe:5.2f}b | Avg MAE: {avg_mae:5.2f}b")

# Test filtering criteria for Shorts
print("\n--- EVALUATING SHORT FILTERS ---")
filters_short = [
    ("Baseline: d_tact < -0.10, d_meso < -0.02", lambda s: s['d_tact'] < -0.10 and s['d_meso'] < -0.02),
    ("Edge >= 10 bps", lambda s: s['d_tact'] < -0.10 and s['d_meso'] < -0.02 and s['edge'] >= 10.0),
    ("Edge >= 10 bps + disp_1m <= 0 (No rising spike)", lambda s: s['d_tact'] < -0.10 and s['d_meso'] < -0.02 and s['edge'] >= 10.0 and s['disp_1m'] <= 0.0),
    ("Edge >= 10 bps + disp_1m <= 0 + disp_5m <= 0", lambda s: s['d_tact'] < -0.10 and s['d_meso'] < -0.02 and s['edge'] >= 10.0 and s['disp_1m'] <= 0.0 and s['disp_5m'] <= 0.0),
    ("Edge >= 12 bps + disp_1m <= 0 + d_macro <= 0", lambda s: s['d_tact'] < -0.12 and s['d_meso'] < -0.04 and s['edge'] >= 12.0 and s['disp_1m'] <= 0.0 and s['d_macro'] <= 0.0),
    ("Quad-Horizon Coherence: Macro<0, Meso<-0.04, Tact<-0.12, Edge>=12", lambda s: s['d_macro'] < -0.02 and s['d_meso'] < -0.04 and s['d_tact'] < -0.12 and s['edge'] >= 12.0 and s['disp_1m'] <= 0.0),
]

for name, flt in filters_short:
    matched = [s for s in short_samples if flt(s)]
    if not matched:
        continue
    reached_45 = sum(1 for s in matched if s['short_mfe'] >= 4.5)
    avg_mfe = sum(s['short_mfe'] for s in matched) / len(matched)
    avg_mae = sum(s['short_mae'] for s in matched) / len(matched)
    print(f"{name:65s} | Count: {len(matched):5d} | MFE>=4.5: {reached_45/len(matched)*100:5.1f}% | Avg MFE: {avg_mfe:5.2f}b | Avg MAE: {avg_mae:5.2f}b")
