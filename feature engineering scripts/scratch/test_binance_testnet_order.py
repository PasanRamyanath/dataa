import time
import hmac
import hashlib
import json
import urllib.request
import urllib.parse
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE_URL = "https://testnet.binancefuture.com"
API_KEY = "YfgxjU5MxV5SUBVQuGBOQ0FhuUMHH92gBZEuo3uknR78OJ46ZzFjkEgwMlJetLqc"
API_SECRET = "lp5ypJjEZSot8Tp57YKVj53dkmFZUcLx2kleOYUWjV5eUMdk5Ot30vDRgknbKSSA"

def send_signed_request(method, path, params=None):
    if params is None:
        params = {}
    
    t_req = urllib.request.Request(f"{BASE_URL}/fapi/v1/time", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(t_req, timeout=5, context=ctx) as r:
        server_time = json.loads(r.read().decode())["serverTime"]
        
    params["timestamp"] = server_time
    params["recvWindow"] = 10000
    
    query_str = urllib.parse.urlencode(params)
    signature = hmac.new(API_SECRET.encode("utf-8"), query_str.encode("utf-8"), hashlib.sha256).hexdigest()
    signed_query = f"{query_str}&signature={signature}"
    
    headers = {
        "X-MBX-APIKEY": API_KEY,
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    if method == "GET":
        url = f"{BASE_URL}{path}?{signed_query}"
        req = urllib.request.Request(url, headers=headers)
    elif method == "POST":
        url = f"{BASE_URL}{path}"
        req = urllib.request.Request(url, data=signed_query.encode("utf-8"), headers=headers, method="POST")
    elif method == "DELETE":
        url = f"{BASE_URL}{path}?{signed_query}"
        req = urllib.request.Request(url, headers=headers, method="DELETE")
        
    try:
        with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        return {"error": err, "code": e.code}

# 1. Fetch current ticker price to place a safe non-crossing Post-Only Limit order
p_req = urllib.request.Request(f"{BASE_URL}/fapi/v1/ticker/price?symbol=BTCUSDT", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(p_req, timeout=5, context=ctx) as r:
    current_price = float(json.loads(r.read().decode())["price"])
print(f"Current Testnet BTCUSDT Price: ${current_price:.2f}")

# Place a Post-Only BUY limit order deep below market price ($70,000)
test_price = round(current_price * 0.85, 1) # 15% below market
order_params = {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "LIMIT",
    "timeInForce": "GTX", # Post-Only Maker
    "quantity": "0.005",  # 0.005 BTC (~$400 notional)
    "price": str(test_price)
}
print(f"Placing Post-Only Test Order: BUY 0.005 BTC @ ${test_price} (timeInForce=GTX)...")
order_res = send_signed_request("POST", "/fapi/v1/order", order_params)
print("Order Result:", json.dumps(order_res, indent=2))

if "orderId" in order_res:
    oid = order_res["orderId"]
    print(f"\nOrder #{oid} PLACED SUCCESSFULLY ON BINANCE TESTNET!")
    # Now cancel it
    print(f"Canceling Order #{oid}...")
    cancel_res = send_signed_request("DELETE", "/fapi/v1/order", {"symbol": "BTCUSDT", "orderId": oid})
    print("Cancel Result:", json.dumps(cancel_res, indent=2))
