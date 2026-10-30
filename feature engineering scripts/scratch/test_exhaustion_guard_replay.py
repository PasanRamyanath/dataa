import json
from pathlib import Path
from collections import deque

archive_dirs = [
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle10"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle11"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle12"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle13"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle14"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle15"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle16")
]

class ExhaustionChecker:
    def __init__(self):
        self.recent_mid_prices = deque(maxlen=36)
        self.max_30s = 10.0
        self.max_180s = 15.0
        self.max_300s = 18.0

    def add(self, ts_ms, mid):
        self.recent_mid_prices.append((ts_ms, mid))

    def is_exhausted(self, pdir, now_ms, mid):
        if pdir not in (0, 2) or not self.recent_mid_prices:
            return False, 0.0, 0.0, 0.0
        
        dir_mult = 1.0 if pdir == 2 else -1.0
        
        s30, s180, s300 = 0.0, 0.0, 0.0
        blocked_reason = []
        
        p_30s = [p for ts, p in self.recent_mid_prices if 20000 <= (now_ms - ts) <= 40000]
        if p_30s:
            s30 = dir_mult * ((mid - p_30s[0]) / p_30s[0] * 1e4)
            if s30 > self.max_30s:
                blocked_reason.append(f"30s={s30:.2f}bps>{self.max_30s}")

        p_180s = [p for ts, p in self.recent_mid_prices if 140000 <= (now_ms - ts) <= 200000]
        if p_180s:
            s180 = dir_mult * ((mid - p_180s[0]) / p_180s[0] * 1e4)
            if s180 > self.max_180s:
                blocked_reason.append(f"180s={s180:.2f}bps>{self.max_180s}")

        p_300s = [p for ts, p in self.recent_mid_prices if (now_ms - ts) >= 240000]
        if p_300s:
            s300 = dir_mult * ((mid - p_300s[0]) / p_300s[0] * 1e4)
            if s300 > self.max_300s:
                blocked_reason.append(f"300s={s300:.2f}bps>{self.max_300s}")

        return (len(blocked_reason) > 0), blocked_reason, (s30, s180, s300)

print("=== REPLAYING ALL HISTORICAL CYCLES WITH MULTI-HORIZON EXHAUSTION GUARD ===")

for ad in archive_dirs:
    ledger_files = list(ad.glob("*hierarchical_session_1h_ledger.jsonl"))
    if not ledger_files:
        continue
    lf = ledger_files[0]
    c_name = ad.name
    
    with open(lf, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
        
    checker = ExhaustionChecker()
    
    for i in range(len(rows)):
        r = rows[i]
        now_ms = r['ts_ms']
        mid = r['bbo_mid']
        
        # Check if an entry happened on this bar
        prev_ap = rows[i-1].get('active_position') if i > 0 else None
        curr_ap = r.get('active_position')
        
        if prev_ap is None and curr_ap is not None:
            side = curr_ap['side']
            pdir = 2 if side == 'BUY' else 0
            is_blocked, reasons, (s30, s180, s300) = checker.is_exhausted(pdir, now_ms, mid)
            status = "BLOCKED: " + ", ".join(reasons) if is_blocked else "ALLOWED"
            print(f"{c_name} | Bar {r['bar_idx']:3d} | {side:4s} @ {curr_ap['entry_price']:.2f} | 30s={s30:+5.2f} 180s={s180:+5.2f} 300s={s300:+5.2f} | {status}")
            
        checker.add(now_ms, mid)
