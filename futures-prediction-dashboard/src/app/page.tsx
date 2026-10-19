"use client";

import React, { useState, useEffect, useCallback } from "react";
import { DashboardSettings, EnsemblePrediction } from "@/types";
import { useBinanceWebSocket } from "@/hooks/useBinanceWebSocket";
import { usePaperTrading } from "@/hooks/usePaperTrading";
import { useKaggleCloudStream } from "@/hooks/useKaggleCloudStream";
import { useAccuracyTracker } from "@/hooks/useAccuracyTracker";

import { Header } from "@/components/Header";
import { MetricCards } from "@/components/MetricCards";
import { PositionCard } from "@/components/PositionCard";
import { StackingGauge } from "@/components/StackingGauge";
import { EquityChart } from "@/components/EquityChart";
import { EnsembleCockpit } from "@/components/EnsembleCockpit";
import { BinanceLiveTape } from "@/components/BinanceLiveTape";
import { ModelAccuracyCockpit } from "@/components/ModelAccuracyCockpit";
import { TradeHistoryTable } from "@/components/TradeHistoryTable";
import { ConnectionSettingsModal } from "@/components/ConnectionSettingsModal";
import { Cpu, Server, ShieldCheck, Zap } from "lucide-react";

const DEFAULT_SETTINGS: DashboardSettings = {
  cloudMode: "KAGGLE_LIVE",
  kaggleTunnelUrl: process.env.NEXT_PUBLIC_KAGGLE_TUNNEL_URL || "https://roughly-flip-lay-accurately.trycloudflare.com",
  streamSecret: "",
  metaThreshold: 0.70,
  leverage: 100,
  marginMode: "ISOLATED",
  executionStyle: "SMART_HYBRID",
  vipTier: "VIP0",
  useBnbFeeDiscount: true,
  trailingStopCallbackPct: 0.05,
  tpPct: 0.14,
  slPct: 0.12,
  pollIntervalSec: 10,
  startingCapital: 10000,
  takerFeeBps: 5.0,
  slippageBps: 0.1,
  deadbandPct: 0.01,
  soundAlerts: false,
  executionTarget: "BINANCE_TESTNET_LIVE",
};

export default function DashboardPage() {
  const [settings, setSettings] = useState<DashboardSettings>(DEFAULT_SETTINGS);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // 1. Live Binance USDT-M Futures WebSocket (@aggTrade, @bookTicker, @markPrice@1s, @ticker)
  const { ticker, recentTrades, status: wsStatus } = useBinanceWebSocket("BTCUSDT");

  // 2. High-fidelity Paper Trading Execution Engine (Triple-Barrier, Meta veto, Kelly sizing, deadband, Binance Testnet)
  const {
    account,
    trades,
    equityHistory,
    lastExecutedTrade,
    testnetStatus,
    syncTestnetAccount,
    updatePrice,
    processPrediction,
    manualClose,
    setLeverage,
    setMarginMode,
    resetAccount,
  } = usePaperTrading(settings);

  // 3. Live Forward-Testing Accuracy Tracker (evaluating ensemble forecasts against WebSocket data)
  const {
    stats: accuracyStats,
    recordPrediction: recordAccuracyPrediction,
    recordPriceTick: recordAccuracyPriceTick,
    resetTracker,
  } = useAccuracyTracker();

  // Continuously pipe price ticks + BBO bookTicker + funding rate into the paper engine and forward-test evaluator
  useEffect(() => {
    if (ticker?.price && ticker.price > 0) {
      const bookTicker = {
        bidPrice: ticker.bidPrice || ticker.price - 0.1,
        askPrice: ticker.askPrice || ticker.price + 0.1,
        bidQty: ticker.bidQty || 1.0,
        askQty: ticker.askQty || 1.0,
      };
      const fundingInfo = {
        fundingRate: ticker.fundingRate,
        nextFundingTime: ticker.nextFundingTime,
      };

      updatePrice(ticker.price, bookTicker, fundingInfo);
      recordAccuracyPriceTick(ticker.price);
    }
  }, [ticker, updatePrice, recordAccuracyPriceTick]);

  // Combined callback when a new prediction arrives from the Kaggle VM
  const handleNewPrediction = useCallback(
    (prediction: EnsemblePrediction, price: number) => {
      const bookTicker = ticker
        ? {
            bidPrice: ticker.bidPrice || price - 0.1,
            askPrice: ticker.askPrice || price + 0.1,
          }
        : undefined;

      processPrediction(prediction, price, bookTicker);
      recordAccuracyPrediction(prediction, price);
    },
    [ticker, processPrediction, recordAccuracyPrediction]
  );

  // 4. 10-Second Kaggle Cloud / Simulator Ensemble Prediction Stream
  const {
    latestPrediction,
    isPolling,
    streamStatus,
    lastFetchTime,
    fetchLatency,
    errorMsg,
    togglePause,
    triggerManualTick,
  } = useKaggleCloudStream(
    settings,
    ticker?.price && ticker.price > 10000 ? ticker.price : (ticker?.markPrice || 80800),
    handleNewPrediction
  );

  return (
    <div className="min-h-screen flex flex-col bg-[#090d16] text-slate-100">
      {/* Top Navigation & Status Bar */}
      <Header
        ticker={ticker}
        wsStatus={wsStatus}
        streamStatus={streamStatus}
        isPolling={isPolling}
        onTogglePolling={togglePause}
        onManualTick={triggerManualTick}
        onOpenSettings={() => setIsSettingsOpen(true)}
        settings={settings}
        onSelectMode={(mode) =>
          setSettings((prev) => ({ ...prev, cloudMode: mode }))
        }
        onChangeLeverage={(lev) => {
          setSettings((prev) => ({ ...prev, leverage: lev }));
          setLeverage(lev);
        }}
        onChangeMarginMode={(mode) => {
          setSettings((prev) => ({ ...prev, marginMode: mode }));
          setMarginMode(mode);
        }}
        onToggleExecutionTarget={(target) => {
          setSettings((prev) => ({ ...prev, executionTarget: target }));
        }}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 space-y-4">
        {/* Binance Futures Testnet Active Banner */}
        {settings.executionTarget === "BINANCE_TESTNET_LIVE" && (
          <div className="p-3 bg-gradient-to-r from-emerald-950/40 via-zinc-900/60 to-emerald-950/40 border border-emerald-800/60 rounded-xl text-emerald-300 text-xs flex flex-wrap items-center justify-between gap-2 shadow-sm font-mono">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>
                <strong className="text-emerald-200">Binance USDⓈ-M Testnet Live:</strong> BTCUSDT Perpetual · {account.leverage}x Leverage · Maker Post-Only GTX Enabled
              </span>
            </div>
            <div className="flex items-center space-x-3 text-[11px] text-zinc-400">
              {testnetStatus.lastMessage && (
                <span className="text-amber-300 font-semibold truncate max-w-xs">
                  {testnetStatus.lastMessage}
                </span>
              )}
              <span className="text-zinc-500">
                API Key: YfgxjU5...JetLqc
              </span>
              <button
                onClick={() => syncTestnetAccount()}
                className="px-2 py-0.5 rounded bg-emerald-900/60 hover:bg-emerald-800 text-emerald-200 text-[10px] transition-colors"
                title="Sync balance & open position directly from Binance Testnet"
              >
                Sync Now
              </button>
            </div>
          </div>
        )}

        {/* Error / Warning Alert Banner if any */}
        {errorMsg && (
          <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-xl text-rose-300 text-xs flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
              <span>
                <strong>Cloud Connection Notice:</strong> {errorMsg}
              </span>
            </div>
            <button
              onClick={() => triggerManualTick()}
              className="px-2.5 py-1 bg-rose-900/60 hover:bg-rose-800 rounded text-[11px] font-mono font-medium transition-colors"
            >
              Retry Handshake
            </button>
          </div>
        )}

        {/* 1. Account High-Level Performance Metrics Bar */}
        <MetricCards account={account} settings={settings} />

        {/* 2. Primary Cockpit Grid: Position, Meta Stacking Simplex & Equity Curve */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Active Position & Meta-Labeling Stacking Simplex (Col 1-5) */}
          <div className="lg:col-span-5 space-y-4 flex flex-col justify-between">
            <PositionCard
              account={account}
              lastTrade={lastExecutedTrade}
              onManualClose={manualClose}
              settings={settings}
            />
            <StackingGauge prediction={latestPrediction} />
          </div>

          {/* Equity Curve vs BTC Benchmark (Col 6-12) */}
          <div className="lg:col-span-7">
            <EquityChart
              data={equityHistory}
              startingBalance={settings.startingCapital}
            />
          </div>
        </div>

        {/* 3. Secondary Grid: 12-Stage Ensemble Radar & Binance Order Flow Live Tape */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* 12-Stage Model Architecture & Calibration Radar (Col 1-8) */}
          <div className="lg:col-span-8">
            <EnsembleCockpit prediction={latestPrediction} />
          </div>

          {/* Binance Order Flow Tape (Col 9-12) */}
          <div className="lg:col-span-4">
            <BinanceLiveTape ticker={ticker} trades={recentTrades} />
          </div>
        </div>

        {/* 4. Live Forward-Testing Accuracy Engine (Testing Predictions Against WebSocket Ticks) */}
        <div>
          <ModelAccuracyCockpit
            stats={accuracyStats}
            onReset={resetTracker}
            livePrice={ticker?.price}
          />
        </div>

        {/* 5. Complete Simulated Trade History Ledger */}
        <div>
          <TradeHistoryTable trades={trades} />
        </div>
      </main>

      {/* System Footer Status Bar */}
      <footer className="border-t border-zinc-900 bg-zinc-950/60 px-4 py-3 text-xs text-zinc-500 font-mono">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>Dual NVIDIA T4 GPU Pipeline (32GB VRAM)</span>
            </div>
            <span className="text-zinc-700">|</span>
            <div className="flex items-center space-x-1.5">
              <Server className="w-3.5 h-3.5 text-emerald-400" />
              <span>
                Cadence: {settings.pollIntervalSec}s Bar · Meta Gate &tau; ={" "}
                {settings.metaThreshold.toFixed(2)}
              </span>
            </div>
            <span className="text-zinc-700">|</span>
            <div className="flex items-center space-x-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-amber-400" />
              <span>Friction: {settings.takerFeeBps + settings.slippageBps} bps/turn</span>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {fetchLatency > 0 && (
              <span className="text-zinc-400">
                Inference Latency:{" "}
                <span className="text-cyan-400 font-bold">{fetchLatency}ms</span>
              </span>
            )}
            {lastFetchTime > 0 && (
              <span className="text-zinc-500">
                Last Sync: {new Date(lastFetchTime).toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      </footer>

      {/* Connection & Risk Tuning Settings Modal */}
      <ConnectionSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSaveSettings={setSettings}
        onResetAccount={() => resetAccount(settings.startingCapital)}
      />
    </div>
  );
}
