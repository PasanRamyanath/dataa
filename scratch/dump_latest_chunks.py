import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KAGGLE_USERNAME'] = 'rimramyanath'
os.environ['KAGGLE_KEY'] = '9a6dc5cfd44155c1e1eba7f119019e8b'
import kaggle
api = kaggle.KaggleApi()
api.authenticate()

t0 = time.time()
count = 0
recent = []
for item in api.kernels_logs_stream('rimramyanath/btcusdt-v4-ensemble-train'):
    count += 1
    data = item.get('data', '')
    if count >= 6850 and data:
        recent.append((count, data.strip()))
    if time.time() - t0 > 28:
        break

print(f"Total chunks: {count}")
for c, d in recent:
    if not d.startswith('â') and 'it/s' not in d and 'eta ' not in d and not (d.endswith('%') and len(d) < 10):
        print(f"[{c}] {d}")
