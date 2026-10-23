import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KAGGLE_USERNAME'] = 'rimramyanath'
os.environ['KAGGLE_KEY'] = '9a6dc5cfd44155c1e1eba7f119019e8b'
import kaggle
api = kaggle.KaggleApi()
api.authenticate()

t0 = time.time()
n = 0
for item in api.kernels_logs_stream('rimramyanath/btcusdt-v4-ensemble-train'):
    line = item.get('data', '')
    if line:
        n += 1
        # only print if past the initial pip lines
        if '===' in line or '---' in line or 'Starting' in line or 'Executing' in line or 'Fold' in line or 'v4.0' in line or 'Force Retrain' in line or 'Initiating' in line or 'Tier' in line or 'Epoch' in line or 'loss' in line:
            print(line, end='', flush=True)
    if time.time() - t0 > 25:
        break
print(f"\n[Read {n} chunks in {time.time()-t0:.1f}s]")
