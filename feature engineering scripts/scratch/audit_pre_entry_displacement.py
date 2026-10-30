import json
from pathlib import Path

# Let's inspect the archived ledgers
archive_dirs = [
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle10"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle11"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle12"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle13"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle14"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle15"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch") # current cycle 16
]

print("=== AUDIT OF PRE-ENTRY PRICE DISPLACEMENT (3m / 5m) ACROSS CYCLES ===")

for ad in archive_dirs:
    ledger_files = list(ad.glob("*hierarchical_session_1h_ledger.jsonl"))
    if not ledger_files:
        continue
    lf = ledger_files[0]
    c_name = ad.name if "archive" in ad.name else "cycle16"
    
    with open(lf, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    
    # Find entries
    for i in range(1, len(rows)):
        prev_ap = rows[i-1].get('active_position')
        curr_ap = rows[i].get('active_position')
        if prev_ap is None and curr_ap is not None:
            entry_bar = rows[i]['bar_idx']
            entry_p = curr_ap['entry_price']
            side = curr_ap['side']
            
            # Check 1m (6 bars), 3m (18 bars), 5m (30 bars) displacement in trade direction
            disp_1m = 0.0
            disp_3m = 0.0
            disp_5m = 0.0
            if i >= 6:
                p_1m = rows[i-6]['bbo_mid']
                disp_1m = ((entry_p - p_1m) / p_1m * 1e4) if side == 'BUY' else ((p_1m - entry_p) / p_1m * 1e4)
            if i >= 18:
                p_3m = rows[i-18]['bbo_mid']
                disp_3m = ((entry_p - p_3m) / p_3m * 1e4) if side == 'BUY' else ((p_3m - entry_p) / p_3m * 1e4)
            if i >= 30:
                p_5m = rows[i-30]['bbo_mid']
                disp_5m = ((entry_p - p_5m) / p_5m * 1e4) if side == 'BUY' else ((p_5m - entry_p) / p_5m * 1e4)
            
            print(f"{c_name} | Bar {entry_bar:3d} | {side:4s} @ {entry_p:.2f} | 1m_disp: {disp_1m:+5.2f} bps | 3m_disp: {disp_3m:+5.2f} bps | 5m_disp: {disp_5m:+5.2f} bps")
