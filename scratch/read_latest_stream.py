import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KAGGLE_USERNAME'] = 'rimramyanath'
os.environ['KAGGLE_KEY'] = '9a6dc5cfd44155c1e1eba7f119019e8b'
import kaggle
api = kaggle.KaggleApi()
api.authenticate()

t0 = time.time()
idx = 0
recent = []
print("Streaming from Kaggle API...")
for item in api.kernels_logs_stream('rimramyanath/btcusdt-v4-ensemble-train'):
    idx += 1
    data = item.get('data', '')
    if idx > 3100 and data.strip():
        recent.append((idx, data.strip()))
    if time.time() - t0 > 60:
        break

print(f"Total chunks processed: {idx} in {time.time()-t0:.1f}s")
print(f"Chunks past 3100: {len(recent)}")
print("\n--- NEW CHUNKS PAST 3100 ---")
for i, d in recent:
    if not d.startswith('â') and 'it/s' not in d and 'eta ' not in d and not d.endswith('%'):
        print(f"[{i}] {d}")
