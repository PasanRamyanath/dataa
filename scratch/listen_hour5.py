import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KAGGLE_USERNAME'] = 'rimramyanath'
os.environ['KAGGLE_KEY'] = '9a6dc5cfd44155c1e1eba7f119019e8b'
import kaggle
api = kaggle.KaggleApi()
api.authenticate()

t0 = time.time()
print("Connecting to live log stream...", flush=True)
count = 0
recent = []
for item in api.kernels_logs_stream('rimramyanath/btcusdt-v4-ensemble-train'):
    count += 1
    data = item.get('data', '')
    if data:
        recent.append((count, data))
        if len(recent) > 50:
            recent.pop(0)
    if time.time() - t0 > 25:
        break

print(f"Total chunks seen: {count} in {time.time()-t0:.1f}s", flush=True)
print("\n--- RECENT CHUNKS ---")
for c, d in recent:
    d_str = d.strip()
    if d_str and not d_str.startswith('â') and 'it/s' not in d_str and 'eta ' not in d_str and not (d_str.endswith('%') and len(d_str) < 10):
        print(f"[{c}] {d_str}", flush=True)
