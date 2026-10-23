import sys, os, time, re
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
lines = raw_text.splitlines()

keywords = ["Starting", "Finished", "Uploading", "Initiating", "TIER", "wave", "patchtst", "tft", "hmm", "kronos", "Saved", "metrics", "complete"]
matches = []
for l in lines:
    l_str = l.strip()
    if any(k in l_str for k in keywords) and "it/s" not in l_str and "eta " not in l_str:
        matches.append(l_str)

print(f"Matched {len(matches)} milestone lines:")
for m in matches[-40:]:
    print(m)
