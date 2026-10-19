"use client";

import React, { useState, useEffect } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Cloud,
  Play,
  Pause,
  RefreshCw,
  Settings,
  Zap,
  Clock,
  Layers,
} from "lucide-react";
import { BinanceTicker, DashboardSettings, MarginMode } from "@/types";
import { WsConnectionStatus } from "@/lib/binanceWs";

interface HeaderProps {
  ticker: BinanceTicker | null;
  wsStatus: WsConnectionStatus;
  streamStatus: "ACTIVE" | "IDLE" | "ERROR";
  isPolling: boolean;
  onTogglePolling: () => void;
  onManualTick: () => void;
  onOpenSettings: () => void;
  settings: DashboardSettings;
  onSelectMode: (mode: "SIMULATOR" | "KAGGLE_LIVE") => void;
  onChangeLeverage?: (leverage: number) => void;
  onChangeMarginMode?: (mode: MarginMode) => void;
  onToggleExecutionTarget?: (target: "LOCAL_SIMULATOR" | "BINANCE_TESTNET_LIVE") => void;
}

export const Header: React.FC<HeaderProps> = ({
  ticker,
  wsStatus,
  streamStatus,
  isPolling,
  onTogglePolling,
  onManualTick,
  onOpenSettings,
  settings,
  onSelectMode,
  onChangeLeverage,
  onChangeMarginMode,
  onToggleExecutionTarget,
}) => {
  const isUp = (ticker?.priceChangePercent24h ?? 0) >= 0;
  const [fundingCountdown, setFundingCountdown] = useState<string>("--:--:--");

  // Live 1-second countdown clock to next Binance 8-hour funding cash transfer
  useEffect(() => {
    const updateCountdown = () => {
      let targetMs = ticker?.nextFundingTime;
      if (!targetMs || targetMs <= Date.now()) {
        const now = new Date();
        const currentUtcHours = now.getUTCHours();
        const nextFundingHour =
          currentUtcHours < 8 ? 8 : currentUtcHours < 16 ? 16 : 24;
        const nextDate = new Date(now);
        if (nextFundingHour === 24) {
          nextDate.setUTCDate(nextDate.getUTCDate() + 1);
          nextDate.setUTCHours(0, 0, 0, 0);
        } else {
          nextDate.setUTCHours(nextFundingHour, 0, 0, 0);
        }
        targetMs = nextDate.getTime();
      }

      const diff = Math.max(0, targetMs - Date.now());
      const h = Math.floor(diff / 3600000);
      const m = Math.floor((diff % 3600000) / 60000);
      const s = Math.floor((diff % 60000) / 1000);
      setFundingCountdown(
        `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
      );
    };

    updateCountdown();
    const interval = setInterval(updateCountdown, 1000);
    return () => clearInterval(interval);
  }, [ticker?.nextFundingTime]);

  const cycleLeverage = () => {
    const levels = [20, 50, 75, 100, 125];
    const current = settings.leverage || 100;
    const idx = levels.indexOf(current);
    const next = levels[(idx + 1) % levels.length];
    onChangeLeverage?.(next);
  };

  const toggleMarginMode = () => {
    const nextMode: MarginMode =
      settings.marginMode === "ISOLATED" ? "CROSS" : "ISOLATED";
    onChangeMarginMode?.(nextMode);
  };

  return (
    <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-40 px-4 py-3">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        {/* Brand & Market Identifier */}
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-bold text-base text-zinc-100 tracking-tight">
                Project IMPJR
              </h1>
              <span className="px-2 py-0.5 text-xs font-semibold rounded bg-zinc-800 text-zinc-300 border border-zinc-700">
                BTCUSDT Perpetual
              </span>

              {/* Leverage Quick Cycle Pill */}
              <button
                type="button"
                onClick={cycleLeverage}
                title="Click to cycle Binance leverage (20x, 50x, 75x, 100x, 125x)"
                className="px-2 py-0.5 text-xs font-mono font-bold rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30 transition-all flex items-center space-x-1"
              >
                <span>{settings.leverage || 100}x</span>
              </button>

              {/* Margin Mode Quick Toggle Pill */}
              <button
                type="button"
                onClick={toggleMarginMode}
                title="Click to toggle Margin Mode (ISOLATED / CROSS)"
                className={`px-2 py-0.5 text-xs font-mono font-bold rounded border transition-all ${
                  settings.marginMode === "ISOLATED"
                    ? "bg-sky-950/80 text-sky-300 border-sky-700/60 hover:bg-sky-900/60"
                    : "bg-purple-950/80 text-purple-300 border-purple-700/60 hover:bg-purple-900/60"
                }`}
              >
                {settings.marginMode || "ISOLATED"}
              </button>

              {/* VIP Tier Badge */}
              <span className="hidden xl:inline px-1.5 py-0.5 text-[10px] font-mono rounded bg-zinc-800 text-zinc-400 border border-zinc-700">
                {settings.vipTier || "VIP0"} {settings.useBnbFeeDiscount ? "• BNB -10%" : ""}
              </span>

              {/* Execution Target Toggle Badge */}
              <button
                type="button"
                onClick={() =>
                  onToggleExecutionTarget?.(
                    settings.executionTarget === "BINANCE_TESTNET_LIVE"
                      ? "LOCAL_SIMULATOR"
                      : "BINANCE_TESTNET_LIVE"
                  )
                }
                title="Click to toggle between Binance Futures Testnet and Offline Simulator"
                className={`px-2 py-0.5 text-xs font-mono font-bold rounded border transition-all flex items-center space-x-1.5 ${
                  settings.executionTarget === "BINANCE_TESTNET_LIVE"
                    ? "bg-emerald-950/80 text-emerald-300 border-emerald-600/70 hover:bg-emerald-900/60 shadow-sm shadow-emerald-950"
                    : "bg-zinc-800 text-zinc-400 border-zinc-700 hover:bg-zinc-700"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    settings.executionTarget === "BINANCE_TESTNET_LIVE"
                      ? "bg-emerald-400 animate-pulse"
                      : "bg-zinc-500"
                  }`}
                />
                <span>
                  {settings.executionTarget === "BINANCE_TESTNET_LIVE"
                    ? "Binance Testnet"
                    : "Local Sim"}
                </span>
              </button>
            </div>
            <p className="text-xs text-zinc-400 hidden sm:block">
              12-Stage Deep Ensemble • Dirichlet ODIR • Meta-Labeling Filter
            </p>
          </div>
        </div>

        {/* Live Binance Ticker Stats with Real-time 8h Funding Countdown */}
        {ticker && ticker.price > 0 && (
          <div className="flex items-center space-x-5 border-x border-zinc-800/80 px-4 py-1 hidden md:flex">
            <div>
              <div className="text-[10px] uppercase font-mono text-zinc-500">Live Mark Price</div>
              <div className="text-lg font-mono font-bold text-zinc-100">
                ${ticker.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>

            <div>
              <div className="text-[10px] uppercase font-mono text-zinc-500">24h Change</div>
              <div className={`text-sm font-mono font-semibold ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
                {isUp ? "+" : ""}{ticker.priceChangePercent24h.toFixed(2)}%
              </div>
            </div>

            {/* 8-Hour Funding Rate & Countdown */}
            <div>
              <div className="text-[10px] uppercase font-mono text-zinc-500 flex items-center space-x-1">
                <Clock className="w-3 h-3 text-zinc-400" />
                <span>Funding ({fundingCountdown})</span>
              </div>
              <div className="text-sm font-mono text-zinc-300 font-semibold">
                {(ticker.fundingRate * 100).toFixed(4)}%
              </div>
            </div>

            {/* Top-of-Book BBO Spread */}
            <div>
              <div className="text-[10px] uppercase font-mono text-zinc-500">BBO Spread</div>
              <div className="text-xs font-mono text-zinc-400">
                ${(ticker.askPrice - ticker.bidPrice).toFixed(2)} ({(
                  ((ticker.askPrice - ticker.bidPrice) / ticker.price) *
                  10000
                ).toFixed(1)} bps)
              </div>
            </div>
          </div>
        )}

        {/* Status Indicators & Action Controls */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          {/* Binance WS Badge */}
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-mono border ${
              wsStatus === "CONNECTED"
                ? "bg-emerald-950/40 border-emerald-800/60 text-emerald-400"
                : wsStatus === "CONNECTING"
                ? "bg-amber-950/40 border-amber-800/60 text-amber-400"
                : "bg-rose-950/40 border-rose-800/60 text-rose-400"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                wsStatus === "CONNECTED"
                  ? "bg-emerald-400 animate-pulse"
                  : wsStatus === "CONNECTING"
                  ? "bg-amber-400 animate-ping"
                  : "bg-rose-400"
              }`}
            />
            <span className="hidden sm:inline">Binance WS</span>
          </div>

          {/* Cloud VM Mode Selector */}
          <div className="flex rounded-lg bg-zinc-900 border border-zinc-800 p-0.5 text-xs font-mono">
            <button
              onClick={() => onSelectMode("SIMULATOR")}
              className={`px-2.5 py-1 rounded-md transition-all ${
                settings.cloudMode === "SIMULATOR"
                  ? "bg-zinc-800 text-zinc-100 font-semibold shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Simulator
            </button>
            <button
              onClick={() => onSelectMode("KAGGLE_LIVE")}
              className={`px-2.5 py-1 rounded-md transition-all flex items-center space-x-1 ${
                settings.cloudMode === "KAGGLE_LIVE"
                  ? "bg-amber-600/30 text-amber-300 font-semibold border border-amber-500/40"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Cloud className="w-3 h-3 inline" />
              <span>Kaggle Cloud</span>
            </button>
          </div>

          {/* Trigger & Pause buttons */}
          <button
            onClick={onTogglePolling}
            title={isPolling ? "Pause 10s tick loop" : "Resume 10s tick loop"}
            className={`p-1.5 rounded-lg border transition-all ${
              isPolling
                ? "bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border-zinc-800"
                : "bg-amber-500/20 text-amber-400 border-amber-500/30"
            }`}
          >
            {isPolling ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>

          <button
            onClick={onManualTick}
            title="Trigger manual 10s prediction tick"
            className="p-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 transition-all"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          {/* Settings Modal Toggle */}
          <button
            onClick={onOpenSettings}
            title="Connection & Threshold Settings"
            className="p-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 transition-all"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
