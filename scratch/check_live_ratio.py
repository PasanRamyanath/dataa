import urllib.request, json
req = urllib.request.Request("https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol=BTCUSDT", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
print("Live bookTicker:", data)
bid_qty = float(data['bidQty'])
ask_qty = float(data['askQty'])
ratio = bid_qty / ask_qty
print(f"bidQty={bid_qty}, askQty={ask_qty}, ratio={ratio:.4f}")
print(f"ratio < 1.02? {ratio < 1.02}")
