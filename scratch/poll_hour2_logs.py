import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KAGGLE_USERNAME'] = 'rimramyanath'
os.environ['KAGGLE_KEY'] = '9a6dc5cfd44155c1e1eba7f119019e8b'
import kaggle
api = kaggle.KaggleApi()
api.authenticate()

t0 = time.time()
lines = []
print("Connecting to live log stream...")
for item in api.kernels_logs_stream('rimramyanath/btcusdt-v4-ensemble-train'):
    data = item.get('data', '')
    if data:
        lines.append(data)
    # Stop after 25s so we get a good snapshot without blocking indefinitely
    if time.time() - t0 > 25:
        break

print(f"Captured {len(lines)} log chunks in {time.time()-t0:.1f}s.")
# Print the last 60 non-empty chunks
filtered = [l for l in lines if not l.startswith('   â') and not l.startswith('     â')]
print("\n--- RECENT LOG CHUNKS (LAST 50) ---")
for l in filtered[-50:]:
    print(l, end='')
