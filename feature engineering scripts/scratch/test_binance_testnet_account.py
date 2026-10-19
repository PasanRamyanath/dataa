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
    
    # 1. Server time
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

# 1. Check current leverage and position mode for BTCUSDT
print("Checking BTCUSDT Position Risk & Leverage...")
pos_risk = send_signed_request("GET", "/fapi/v2/positionRisk", {"symbol": "BTCUSDT"})
print("Position Risk:", json.dumps(pos_risk, indent=2))

# 2. Set Leverage to 50x or 100x for BTCUSDT
print("\nSetting Leverage to 50x for BTCUSDT...")
lev_res = send_signed_request("POST", "/fapi/v1/leverage", {"symbol": "BTCUSDT", "leverage": 50})
print("Leverage Result:", lev_res)

# 3. Check Open Orders
print("\nChecking Open Orders...")
open_orders = send_signed_request("GET", "/fapi/v1/openOrders", {"symbol": "BTCUSDT"})
print("Open Orders Count:", len(open_orders))
