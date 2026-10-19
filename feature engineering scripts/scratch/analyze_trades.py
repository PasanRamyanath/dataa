import re

log_path = 'C:/Users/pasan/.gemini/antigravity-cli/brain/c79982d7-6c43-4646-ac71-494cdd8fc285/.system_generated/tasks/task-5973.log'
trades = []
with open(log_path, 'r') as f:
    for line in f:
        if '>>> TRADE #' in line:
            m = re.search(r'Gross:\s*([+-]?\d+\.\d+)\s*bps', line)
            m_id = re.search(r'TRADE #(\d+)', line)
            m_reason = re.search(r'CLOSED \[([A-Z_]+)\]', line)
            if m and m_id:
                gross = float(m.group(1))
                tid = int(m_id.group(1))
                reason = m_reason.group(1) if m_reason else 'UNKNOWN'
                trades.append({'id': tid, 'reason': reason, 'gross': gross})

print('Trade | Reason       | Gross bps | Net (14bps) | Net (10bps) | Net (5.6bps) | Net Maker (3.2bps)')
print('-' * 85)
total_gross = 0
total_net_14 = 0
total_net_10 = 0
total_net_56 = 0
total_net_32 = 0

for t in trades:
    g = t['gross']
    n14 = g - 14.0
    n10 = g - 10.0
    n56 = g - 5.6
    n32 = g - 3.2
    total_gross += g
    total_net_14 += n14
    total_net_10 += n10
    total_net_56 += n56
    total_net_32 += n32
    print(f"#{t['id']:02d}   | {t['reason']:<12s} | {g:+7.2f}   | {n14:+8.2f}    | {n10:+8.2f}    | {n56:+9.2f}    | {n32:+10.2f}")

print('=' * 85)
print(f"TOTAL (16 Trades)   | {total_gross:+7.2f}   | {total_net_14:+8.2f}    | {total_net_10:+8.2f}    | {total_net_56:+9.2f}    | {total_net_32:+10.2f}")
