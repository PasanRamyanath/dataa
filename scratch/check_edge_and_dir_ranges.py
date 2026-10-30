import os
import glob
import json

base = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base, d))])

for a in archives[::5]: # sample every 5th cycle
    p = os.path.join(base, a)
    ledgers = glob.glob(os.path.join(p, "*ledger*.jsonl"))
    if not ledgers: continue
    with open(ledgers[0], "r", encoding="utf-8") as f:
        edges = []
        dirs = []
        for line in f:
            if line.strip().startswith("{"):
                d = json.loads(line)
                edge = d.get("expected_net_edge_bps", d.get("edge", 0.0))
                pdir = d.get("primary_direction", d.get("direction", 1))
                edges.append(edge)
                dirs.append(pdir)
        if edges:
            print(f"{a:<22} | Dir Set: {set(dirs)} | Edge min/med/max: {min(edges):.1f} / {sorted(edges)[len(edges)//2]:.1f} / {max(edges):.1f}")
