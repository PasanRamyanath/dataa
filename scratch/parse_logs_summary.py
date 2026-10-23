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
# Split by lines
lines = raw_text.splitlines()

# Filter out progress bar iterations
clean_lines = []
for l in lines:
    l_strip = l.strip()
    if not l_strip:
        continue
    if "it/s" in l_strip or "eta " in l_strip:
        continue
    if re.search(r'^\d+%', l_strip):
        continue
    clean_lines.append(l_strip)

print(f"Total substantive log lines: {len(clean_lines)}")
print("=" * 80)
# Print last 60 substantive log lines
for l in clean_lines[-60:]:
    print(l)
print("=" * 80)
