import os, glob, json, re

def get_cycle_num(p):
    m = re.search(r'archive_cycle(\d+)', p)
    return int(m.group(1)) if m else 999

metrics_files = sorted(
    glob.glob('scratch/archive_cycle*/champion_challenger_final_results.json') +
    glob.glob('scratch/archive_cycle*/live_session_final_metrics.json'),
    key=get_cycle_num
)

winning_trades = []
for mf in metrics_files:
    c_num = get_cycle_num(mf)
    with open(mf, 'r', encoding='utf-8') as f:
        try:
            d = json.load(f)
            # Check champion_challenger_final_results format
            if 'competitor_trade_ledgers' in d:
                for comp_name, trades in d['competitor_trade_ledgers'].items():
                    for t in trades:
                        if t.get('net_return_bps', 0) > 0:
                            winning_trades.append((c_num, comp_name, t))
            elif 'closed_trades' in d:
                for t in d['closed_trades']:
                    if t.get('net_return_bps', 0) > 0:
                        winning_trades.append((c_num, 'LiveSession', t))
        except Exception:
            pass

print(f"Total winning trades found in archives: {len(winning_trades)}")
# Deduplicate
unique_wins = []
seen = set()
for c, comp, t in winning_trades:
    key = (c, t.get('entry_ts_ms'), t.get('entry_price'), round(t.get('net_return_bps', 0), 2))
    if key not in seen:
        seen.add(key)
        unique_wins.append((c, comp, t))

print(f"Total unique winning trades: {len(unique_wins)}")
print("\nSample top winning trades:")
for c, comp, t in sorted(unique_wins, key=lambda x: -x[2].get('net_return_bps', 0))[:15]:
    side = t.get('side')
    entry = t.get('entry_price')
    exit_p = t.get('exit_price')
    net = t.get('net_return_bps')
    mfe = t.get('peak_mfe_bps')
    mae = t.get('max_mae_bps')
    dur = t.get('holding_duration_sec')
    reason = t.get('exit_reason')
    print(f"Cycle {c:2d} | {comp:14s} | {side} @ {entry:.1f} -> {exit_p:.1f} | Net: {net:+6.2f}b | MFE: {mfe:5.2f}b | MAE: {mae:5.2f}b | Dur: {dur:3.0f}s | {reason}")
