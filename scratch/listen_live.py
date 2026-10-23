import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KAGGLE_USERNAME'] = 'rimramyanath'
os.environ['KAGGLE_KEY'] = '9a6dc5cfd44155c1e1eba7f119019e8b'
import kaggle
api = kaggle.KaggleApi()
api.authenticate()

t0 = time.time()
print("Connecting to live tail of log stream...", flush=True)
count = 0
for item in api.kernels_logs_stream('rimramyanath/btcusdt-v4-ensemble-train'):
    count += 1
    # Once we pass the historical backlog (which is ~3185 chunks), print everything!
    if count > 3180:
        data = item.get('data', '')
        if data.strip():
            print(f"LIVE [{count}]: {data.strip()}", flush=True)
    if time.time() - t0 > 25:
        break

print(f"Finished listening. Total chunks seen: {count} in {time.time()-t0:.1f}s", flush=True)
