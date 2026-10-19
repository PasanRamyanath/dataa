import urllib.request, json, time, hmac, hashlib

k = 'YfgxjU5MxV5SUBVQuGBOQ0FhuUMHH92gBZEuo3uknR78OJ46ZzFjkEgwMlJetLqc'
s = 'lp5ypJjEZSot8Tp57YKVj53dkmFZUcLx2kleOYUWjV5eUMdk5Ot30vDRgknbKSSA'
u = 'https://testnet.binancefuture.com'

t = json.loads(urllib.request.urlopen(f'{u}/fapi/v1/time', timeout=5).read())['serverTime']
d = int(time.time()*1000) - t

def q(p, params=None):
    if params is None: params = {}
    params['timestamp'] = int(time.time()*1000) - d
    params['recvWindow'] = 10000
    qs = '&'.join(f'{k}={v}' for k, v in sorted(params.items()))
    sig = hmac.new(s.encode(), qs.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(f'{u}{p}?{qs}&signature={sig}', headers={'X-MBX-APIKEY': k})
    return json.loads(urllib.request.urlopen(req, timeout=5).read().decode())

pos = [p for p in q('/fapi/v2/positionRisk', {'symbol':'BTCUSDT'}) if float(p.get('positionAmt',0)) != 0]
print('=== POSITION RISK ===')
print('Active Positions:', pos if pos else 'FLAT (0.0000 BTC)')

open_orders = q('/fapi/v1/openOrders', {'symbol':'BTCUSDT'})
print('\n=== OPEN ORDERS ===')
print('Count:', len(open_orders))
for o in open_orders:
    print('ID:', o['orderId'], 'Side:', o['side'], 'Type:', o['type'], 'Qty:', o['origQty'], 'Price:', o['price'], 'TimeInForce:', o['timeInForce'])

orders = q('/fapi/v1/allOrders', {'symbol':'BTCUSDT', 'limit': 10})
print('\n=== RECENT 10 ORDERS ===')
for o in orders[-10:]:
    print('ID:', o['orderId'], '|', o['side'], o['type'], '| Qty:', o['origQty'], '| Status:', o['status'], '| AvgPrice:', o['avgPrice'], '| reduceOnly:', o['reduceOnly'])

acc = q('/fapi/v2/account')
print('\n=== ACCOUNT SUMMARY ===')
print('Total Wallet Balance:', acc.get('totalWalletBalance'), 'USDT')
print('Total Margin Balance:', acc.get('totalMarginBalance'), 'USDT')
print('Available Balance:', acc.get('availableBalance'), 'USDT')
