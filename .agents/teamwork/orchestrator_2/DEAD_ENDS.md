# Dead Ends Log

| Iteration | Approach Tried | Why It Failed | Files Touched |
|---|---|---|---|
| 1 | In-file toy reference implementations and static literal assertions in test suite (`tests/test_leakage_and_features.py`) | Failed Forensic Integrity Audit (Pattern 1 Hardcoding, Pattern 4 Self-Certifying / Decoupled tests). Production code was genuine, but test suite failed to test production modules directly. | `tests/test_leakage_and_features.py` |
| 1 | Selective freeze partitioning only on `btcusdt@aggTrade` in `binance_ws.py` | Failed AC3 empirical challenge: `depthUpdate`, `markPriceUpdate`, `forceOrder`, and `ethusdt@bookTicker` lacked freeze guards, allowing ticks from $T_{close} - 1000\text{ ms} < t \le T_{close}$ into features. | `feature engineering scripts/streaming/live_ingestion/binance_ws.py` |
