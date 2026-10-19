# Binance Futures Prediction & Paper Trading Cockpit

A high-performance Next.js 16 (React 19, TypeScript, Tailwind CSS) quantitative trading cockpit and paper trading validation environment designed specifically for testing the outputs of the **Kaggle Dual-T4 12-Stage Cryptocurrency Ensemble Pipeline**.

---

## 🌟 Key Features

1. **Direct Binance USDT-M Futures WebSocket Stream (`wss://fstream.binance.com`)**:
   - Streams live trades via `btcusdt@aggTrade`.
   - Microsecond order book updates via `btcusdt@bookTicker`.
   - Real-time mark price, index price, and funding rates via `btcusdt@markPrice@1s`.
   - 24h ticker metrics via `btcusdt@ticker`.
   - Integrated heartbeat watchdog (15-second ping) with automatic exponential reconnect.

2. **Full Paper Trading Engine with Triple-Barrier & Continuous Kelly Sizing**:
   - Replicates `inference/paper_trading_engine.py` exactly in TypeScript.
   - **Triple-Barrier Method Exits**: Takes Profit at **+1.5%**, Stops Loss at **-1.0%**, or exits at maximum hold horizon (360 bars = 1 hour).
   - **Continuous Kelly Sizing**: Sizes positions dynamically based on probability confidence ($f^* = (p \cdot b - q) / b \cdot 0.5$).
   - **Meta-Labeling Veto Gate ($\tau = 0.60$)**: Blocks noisy or low-confidence trade executions when $P(\text{Act}) < 0.60$.
   - **1% Turnover Deadband**: Prevents fee burn on small continuous fraction rebalances (`|new_w - current_w| < 0.01`).
   - **Realistic Friction**: Deducts 5.0 bps taker fee + 2.0 bps slippage per side (7.0 bps total).

3. **12-Stage Ensemble Model Cockpit**:
   - Inspects calibrated probabilities: $P(\text{Down})$, $P(\text{Neutral})$, $P(\text{Up})$.
   - Displays all 9 base models: **PatchTST**, **TFT**, **N-HiTS**, **LightGBM**, **Kronos**, **SSM Challenger**, **iTransformer**, **FinBERT News Sentiment**, and **HMM Regime Multiplier** ($M_{\text{regime}} \in [0.4, 1.2]$).
   - Monitors Dirichlet ODIR calibration and Level-2 Stacking Simplex.

4. **Equity Curve & Live Order Flow Tape**:
   - Responsive SVG equity curve vs. Bitcoin benchmark price.
   - Live order tape highlighting buyer-maker aggressors and trade size.
   - Filterable ledger of simulated transactions with execution reasons.

5. **Dual Modes**:
   - **Simulator Mode**: Offline high-fidelity generative process matching empirical return distributions and model correlations for instant testing.
   - **Kaggle Live Cloud Mode**: Connects directly via Cloudflare Tunnel or Webhook to your live Kaggle Dual-T4 VM.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Start the Development Server
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🔗 Connecting to Kaggle Dual-T4 VM

In your Kaggle notebook running the inference pipeline:

1. Launch a Cloudflare Tunnel:
```python
# In Kaggle notebook cell
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
!cloudflared tunnel --url http://127.0.0.1:8000
```
2. Note the generated public URL (e.g. `https://xxxx.trycloudflare.com`).
3. In the Dashboard:
   - Click the **Gear (Settings)** icon in the top right.
   - Select **Kaggle Cloud VM (Live Stream)**.
   - Paste your Cloudflare Tunnel URL.
   - Click **Save & Connect**.

---

## 📂 Project Architecture

```
futures-prediction-dashboard/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── predictions/route.ts  # Cloud proxy / fallback simulator
│   │   │   └── webhook/route.ts      # Webhook receiver for Kaggle dispatches
│   │   ├── globals.css               # Dark terminal styling & custom scrollbars
│   │   ├── layout.tsx                # App root layout & metadata
│   │   └── page.tsx                  # Main trading cockpit
│   ├── components/
│   │   ├── BinanceLiveTape.tsx       # Live trade flow & book ticker
│   │   ├── ConnectionSettingsModal.tsx # Tunnel & risk tuning dialog
│   │   ├── EnsembleCockpit.tsx       # 9 base models radar
│   │   ├── EquityChart.tsx           # SVG Equity curve vs benchmark
│   │   ├── Header.tsx                # Market bar & controls
│   │   ├── MetricCards.tsx           # PnL, Drawdown, Exposure metrics
│   │   ├── PositionCard.tsx          # Active position & Triple-Barrier track
│   │   ├── StackingGauge.tsx         # Stacking probabilities & Meta Veto
│   │   └── TradeHistoryTable.tsx     # Filterable simulated trade log
│   ├── hooks/
│   │   ├── useBinanceWebSocket.ts    # React hook for Binance stream
│   │   ├── useKaggleCloudStream.ts   # React hook for prediction stream
│   │   └── usePaperTrading.ts        # React hook for client-side paper engine
│   ├── lib/
│   │   ├── binanceWs.ts              # WebSocket client implementation
│   │   ├── mockEnsemble.ts           # Realistic stochastic ensemble simulator
│   │   └── paperEngine.ts            # Triple-barrier paper trading engine
│   └── types/
│       └── index.ts                  # Shared TypeScript interfaces
```
