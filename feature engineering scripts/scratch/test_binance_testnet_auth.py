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

API_KEY = "YfgxjU5MxV5SUBVQuGBOQ0FhuUMHH92gBZEuo3uknR78OJ46ZzFjkEgwMlJetLqc"
API_SECRET = "lp5ypJjEZSot8Tp57YKVj53dkmFZUcLx2kleOYUWjV5eUMdk5Ot30vDRgknbKSSA"

# Test both endpoints: testnet.binancefuture.com and demo-fapi.binance.com
ENDPOINTS = [
    "https://testnet.binancefuture.com",
    "https://demo-fapi.binance.com"
]

def test_endpoint(base_url):
    print(f"\nTesting endpoint: {base_url}")
    # 1. Test server time
    try:
        t_req = urllib.request.Request(f"{base_url}/fapi/v1/time", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(t_req, timeout=5, context=ctx) as r:
            res = json.loads(r.read().decode())
            server_time = res["serverTime"]
            print(f"  Server time OK: {server_time} (diff: {abs(int(time.time()*1000) - server_time)}ms)")
    except Exception as e:
        print(f"  Failed to get server time: {e}")
        return False

    # 2. Test account balance with HMAC-SHA256 signature
    timestamp = server_time
    query_str = f"timestamp={timestamp}&recvWindow=10000"
    signature = hmac.new(API_SECRET.encode("utf-8"), query_str.encode("utf-8"), hashlib.sha256).hexdigest()
    signed_url = f"{base_url}/fapi/v2/account?{query_str}&signature={signature}"
    
    headers = {
        "X-MBX-APIKEY": API_KEY,
        "User-Agent": "Mozilla/5.0"
    }
    
    try:
        req = urllib.request.Request(signed_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5, context=ctx) as r:
            acc = json.loads(r.read().decode())
            print(f"  Account Auth SUCCESSFUL!")
            print(f"  Total Wallet Balance: {acc.get('totalWalletBalance')} USDT")
            print(f"  Total Margin Balance: {acc.get('totalMarginBalance')} USDT")
            print(f"  Available Balance: {acc.get('availableBalance')} USDT")
            # Positions
            positions = [p for p in acc.get("positions", []) if float(p.get("positionAmt", 0)) != 0]
            print(f"  Active Positions: {len(positions)}")
            for p in positions:
                print(f"    {p['symbol']}: Amt={p['positionAmt']}, EntryPrice={p['entryPrice']}, UnrealizedPnL={p['unrealizedProfit']}")
            return True
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        print(f"  HTTP Error {e.code}: {err_body}")
        return False
    except Exception as e:
        print(f"  Auth request error: {e}")
        return False

for ep in ENDPOINTS:
    if test_endpoint(ep):
        print(f"\n=> ACTIVE TESTNET ENDPOINT: {ep}")
        break
