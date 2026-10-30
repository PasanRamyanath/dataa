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

cycle_volatilities = []
for l in all_ledgers:
    c_num = get_cycle_num(l)
    bars = []
    with open(l, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line))
    
    mids = [b['bbo_mid'] for b in bars]
    high = max(mids)
    low = min(mids)
    session_range_bps = ((high - low) / low) * 1e4
    
    # 15-minute rolling ranges
    rolling_ranges = []
    for i in range(90, len(bars)):
        sub = mids[i-90:i]
        r_bps = ((max(sub) - min(sub)) / min(sub)) * 1e4
        rolling_ranges.append(r_bps)
    
    avg_15m_range_bps = sum(rolling_ranges) / len(rolling_ranges) if rolling_ranges else session_range_bps
    cycle_volatilities.append({
        'cycle': c_num,
        'session_range_bps': session_range_bps,
        'avg_15m_range_bps': avg_15m_range_bps,
        'bars': len(bars)
    })

print(f"Computed volatility for {len(cycle_volatilities)} cycles.")
print(f"Average 15m range across all cycles: {sum(c['avg_15m_range_bps'] for c in cycle_volatilities)/len(cycle_volatilities):.1f} bps")
print(f"Min 15m range: {min(c['avg_15m_range_bps'] for c in cycle_volatilities):.1f} bps | Max: {max(c['avg_15m_range_bps'] for c in cycle_volatilities):.1f} bps")

# Distribution of 15m range
low_vol = [c for c in cycle_volatilities if c['avg_15m_range_bps'] < 15.0]
med_vol = [c for c in cycle_volatilities if 15.0 <= c['avg_15m_range_bps'] < 30.0]
high_vol = [c for c in cycle_volatilities if c['avg_15m_range_bps'] >= 30.0]

print(f"Low Volatility cycles (< 15 bps 15m range):   {len(low_vol)} cycles")
print(f"Medium Volatility cycles (15 - 30 bps range): {len(med_vol)} cycles")
print(f"High Volatility cycles (>= 30 bps range):     {len(high_vol)} cycles")
