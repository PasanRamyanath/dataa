import json
import os
import re
import sys
import time

os.environ.setdefault("KAGGLE_USERNAME", "rimramyanath")
os.environ.setdefault("KAGGLE_KEY", "9a6dc5cfd44155c1e1eba7f119019e8b")

import kaggle

kernel_slug = "rimramyanath/btcusdt-prediction-engine"
print(f"Monitoring logs for {kernel_slug}...", flush=True)

url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
seen_count = 0

for attempt in range(120):
    try:
        status_res = kaggle.api.kernels_status(kernel_slug)
        status = str(status_res.get("status", "UNKNOWN") if isinstance(status_res, dict) else getattr(status_res, "status", status_res))
        
        raw_logs = kaggle.api.kernels_logs(kernel_slug)
        log_entries = []
        if isinstance(raw_logs, str) and raw_logs.strip():
            try:
                log_entries = json.loads(raw_logs)
            except Exception:
                pass
        elif isinstance(raw_logs, list):
            log_entries = raw_logs
            
        if len(log_entries) > seen_count:
            for entry in log_entries[seen_count:]:
                data = entry.get("data", "") if isinstance(entry, dict) else str(entry)
                for line in data.splitlines():
                    clean_line = line.strip()
                    if clean_line:
                        print(f"[{status}] {clean_line}", flush=True)
                        m = url_pattern.search(clean_line)
                        if m:
                            tunnel_url = m.group(0)
                            print("\n" + "=" * 75, flush=True)
                            print(f">>> FOUND LIVE CLOUDFLARE TUNNEL URL: {tunnel_url}", flush=True)
                            print("=" * 75 + "\n", flush=True)
                            with open("kaggle_tunnel_url.txt", "w", encoding="utf-8") as f:
                                f.write(tunnel_url.strip())
                            sys.exit(0)
            seen_count = len(log_entries)
        else:
            print(f"[{attempt}] Status: {status} (waiting for logs...)", flush=True)
        
        if "error" in status.lower() or "cancel" in status.lower():
            print(f"Kernel stopped with status: {status}", flush=True)
            sys.exit(1)
            
    except Exception as e:
        print(f"Notice: {e}", flush=True)
        
    time.sleep(5)

print("Timeout waiting for tunnel URL.", flush=True)
sys.exit(2)
