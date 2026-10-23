import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KAGGLE_USERNAME'] = 'rimramyanath'
os.environ['KAGGLE_KEY'] = '9a6dc5cfd44155c1e1eba7f119019e8b'
import kaggle
api = kaggle.KaggleApi()
api.authenticate()

t0 = time.time()
chunks = []
for item in api.kernels_logs_stream('rimramyanath/btcusdt-v4-ensemble-train'):
    data = item.get('data', '')
    if data:
        chunks.append(data)
    if time.time() - t0 > 25:
        break

raw_text = "".join(chunks)
lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

# Filter out progress bar ticks
substantive = [l for l in lines if not l.startswith('â') and 'it/s' not in l and 'eta ' not in l and not (l.endswith('%') and len(l) < 10)]

print(f"Total substantive lines: {len(substantive)}")
print("\n--- LAST 40 SUBSTANTIVE LINES ---")
for l in substantive[-40:]:
    print(l)
