import sys
import socket
import time

old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses
socket.getaddrinfo = new_getaddrinfo

sys.path.insert(0, "D:/PrivateProjects/BTC Futures/feature engineering scripts")
sys.path.insert(0, "D:/PrivateProjects/BTC Futures")

from streaming.live_ingestion.binance_ws import BinanceLiveIngestor

received_bars = []
def on_bar(b):
    print("Received bar:", b.get("close"), b.get("close_ts"))
    received_bars.append(b)

ingestor = BinanceLiveIngestor(on_bar_completed=on_bar, symbol="btcusdt")
ingestor.start_in_background()

print("Waiting for ws connection and 1 bar...")
for i in range(15):
    time.sleep(1)
    if received_bars:
        print("Success! Got live bar from WebSocket.")
        break
    print(f"Elapsed: {i+1}s...")

ingestor.stop()
print("Ingestor stopped. Test finished.")
