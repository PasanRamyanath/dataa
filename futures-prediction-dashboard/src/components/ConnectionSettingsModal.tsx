"use client";

import React, { useState } from "react";
import {
  BINANCE_FUTURES_VIP_FEES,
  DashboardSettings,
  ExecutionStyle,
  MarginMode,
  VipTier,
} from "@/types";
import {
  X,
  Cloud,
  Sliders,
  RotateCcw,
  ShieldCheck,
  Check,
  Zap,
  Percent,
  Layers,
  Sparkles,
  Info,
  Radio,
  CheckCircle2,
  Key,
} from "lucide-react";

interface ConnectionSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: DashboardSettings;
  onSaveSettings: (settings: DashboardSettings) => void;
  onResetAccount: () => void;
}

export const ConnectionSettingsModal: React.FC<ConnectionSettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSaveSettings,
  onResetAccount,
}) => {
  const [localSettings, setLocalSettings] = useState<DashboardSettings>({ ...settings });
  const [savedSuccess, setSavedSuccess] = useState(false);

  if (!isOpen) return null;

  const handleSave = () => {
    // Sync taker fee bps from VIP tier if changed
    const vipInfo = BINANCE_FUTURES_VIP_FEES[localSettings.vipTier || "VIP0"];
    const baseTakerBps = vipInfo ? vipInfo.takerBps : 5.0;
    const finalTakerBps = localSettings.useBnbFeeDiscount ? baseTakerBps * 0.9 : baseTakerBps;

    const updated = {
      ...localSettings,
      takerFeeBps: finalTakerBps,
    };

    onSaveSettings(updated);
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      onClose();
    }, 600);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-zinc-950 border border-zinc-800 rounded-2xl w-full max-w-xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh]">
        {/* Modal Header */}
        <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Sliders className="w-5 h-5 text-amber-400" />
            <h2 className="text-sm uppercase font-mono font-bold text-zinc-100">
              Binance USDⓈ-M Production Settings
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 space-y-4 overflow-y-auto text-xs font-mono">
          {/* 0. Order Execution Target (Binance Testnet Live vs Simulator) */}
          <div className="space-y-2 p-3 rounded-xl bg-gradient-to-r from-zinc-900/90 to-zinc-900/50 border border-zinc-800">
            <div className="flex justify-between items-center">
              <label className="text-zinc-200 font-bold flex items-center space-x-1.5">
                <Radio className="w-4 h-4 text-emerald-400" />
                <span>Order Execution Target</span>
              </label>
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                localSettings.executionTarget === "BINANCE_TESTNET_LIVE"
                  ? "bg-emerald-950 text-emerald-300 border border-emerald-700/60"
                  : "bg-zinc-800 text-zinc-400 border border-zinc-700"
              }`}>
                {localSettings.executionTarget === "BINANCE_TESTNET_LIVE" ? "🟢 TESTNET CONNECTED" : "OFFLINE SIM"}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 pt-1">
              <button
                type="button"
                onClick={() => setLocalSettings({ ...localSettings, executionTarget: "BINANCE_TESTNET_LIVE" })}
                className={`py-2 px-3 rounded-lg border text-left transition-all ${
                  localSettings.executionTarget === "BINANCE_TESTNET_LIVE"
                    ? "bg-emerald-950/60 border-emerald-500 text-emerald-200 shadow"
                    : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                }`}
              >
                <div className="font-bold flex items-center space-x-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Binance Testnet Live</span>
                </div>
                <div className="text-[10px] text-zinc-400 mt-0.5">
                  Real API orders on Binance testnet.binancefuture.com ($5,000 USDT)
                </div>
              </button>

              <button
                type="button"
                onClick={() => setLocalSettings({ ...localSettings, executionTarget: "LOCAL_SIMULATOR" })}
                className={`py-2 px-3 rounded-lg border text-left transition-all ${
                  localSettings.executionTarget === "LOCAL_SIMULATOR"
                    ? "bg-zinc-900 border-amber-500 text-zinc-100 shadow"
                    : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                }`}
              >
                <div className="font-bold">Local Simulator</div>
                <div className="text-[10px] text-zinc-500 mt-0.5">
                  Pure in-memory backtest/paper ledger without API calls
                </div>
              </button>
            </div>

            {localSettings.executionTarget === "BINANCE_TESTNET_LIVE" && (
              <div className="mt-2 pt-2 border-t border-zinc-800/80 flex items-center justify-between text-[11px] text-zinc-400">
                <div className="flex items-center space-x-1.5">
                  <Key className="w-3 h-3 text-amber-400" />
                  <span className="font-mono text-zinc-300">API: YfgxjU5...JetLqc</span>
                </div>
                <div className="text-emerald-400 font-semibold font-mono">
                  Wallet: 5,000.00 USDT (50x)
                </div>
              </div>
            )}
          </div>

          {/* 1. Execution Mode */}
          <div className="space-y-1.5">
            <label className="text-zinc-400 block font-semibold">Signal Source & Prediction Pipeline</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setLocalSettings({ ...localSettings, cloudMode: "SIMULATOR" })}
                className={`py-2 px-3 rounded-lg border text-left transition-all ${
                  localSettings.cloudMode === "SIMULATOR"
                    ? "bg-zinc-900 border-amber-500/80 text-zinc-100 shadow"
                    : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                }`}
              >
                <div className="font-bold">Offline Simulator</div>
                <div className="text-[10px] text-zinc-500">Live Binance WS + High-Fidelity Simulator</div>
              </button>

              <button
                type="button"
                onClick={() => setLocalSettings({ ...localSettings, cloudMode: "KAGGLE_LIVE" })}
                className={`py-2 px-3 rounded-lg border text-left transition-all ${
                  localSettings.cloudMode === "KAGGLE_LIVE"
                    ? "bg-amber-950/40 border-amber-500 text-amber-200 shadow"
                    : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                }`}
              >
                <div className="font-bold flex items-center space-x-1">
                  <Cloud className="w-3.5 h-3.5 inline" />
                  <span>Kaggle Cloud VM (Dual-T4)</span>
                </div>
                <div className="text-[10px] text-zinc-500">Live Cloudflare Tunnel inference stream</div>
              </button>
            </div>
          </div>

          {/* Kaggle Tunnel Endpoint */}
          {localSettings.cloudMode === "KAGGLE_LIVE" && (
            <div className="space-y-3 p-3 rounded-xl bg-zinc-900/60 border border-zinc-800">
              <div className="space-y-1">
                <label className="text-zinc-300 block font-semibold">Cloudflare Tunnel URL</label>
                <input
                  type="text"
                  placeholder="https://accurately-lauderdale-category-colors.trycloudflare.com"
                  value={localSettings.kaggleTunnelUrl}
                  onChange={(e) => setLocalSettings({ ...localSettings, kaggleTunnelUrl: e.target.value })}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-1.5 text-zinc-200 focus:outline-none focus:border-amber-500"
                />
                <span className="text-[10px] text-zinc-500">
                  Target of `cloudflared_manager` running on Kaggle dual-T4 GPU VM.
                </span>
              </div>
            </div>
          )}

          {/* 2. Binance Margin Mode: Isolated vs Cross */}
          <div className="space-y-2 p-3 rounded-xl bg-zinc-900/40 border border-zinc-800">
            <label className="text-zinc-300 font-semibold flex items-center space-x-1.5">
              <Layers className="w-4 h-4 text-sky-400" />
              <span>Binance Margin Mode</span>
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setLocalSettings({ ...localSettings, marginMode: "ISOLATED" })}
                className={`py-2 px-3 rounded-lg border text-left transition-all ${
                  (localSettings.marginMode || "ISOLATED") === "ISOLATED"
                    ? "bg-sky-950/60 border-sky-500 text-sky-200 shadow"
                    : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                }`}
              >
                <div className="font-bold">ISOLATED MARGIN</div>
                <div className="text-[10px] text-zinc-400">
                  Risk capped to initial isolated margin. Liquidations never touch wallet cash.
                </div>
              </button>

              <button
                type="button"
                onClick={() => setLocalSettings({ ...localSettings, marginMode: "CROSS" })}
                className={`py-2 px-3 rounded-lg border text-left transition-all ${
                  localSettings.marginMode === "CROSS"
                    ? "bg-purple-950/60 border-purple-500 text-purple-200 shadow"
                    : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                }`}
              >
                <div className="font-bold">CROSS MARGIN</div>
                <div className="text-[10px] text-zinc-400">
                  Shared wallet balance prevents premature liquidation but risks total account.
                </div>
              </button>
            </div>
          </div>

          {/* 3. Binance Tiered Leverage */}
          <div className="space-y-2 p-3 rounded-xl bg-zinc-900/40 border border-zinc-800">
            <div className="flex justify-between items-center">
              <label className="text-zinc-300 font-semibold flex items-center space-x-1.5">
                <Zap className="w-4 h-4 text-amber-400" />
                <span>Futures Leverage Multiplier</span>
              </label>
              <span className="text-amber-400 font-bold text-sm">
                {(localSettings.leverage || 100)}x
              </span>
            </div>
            <div className="grid grid-cols-6 gap-2 pt-1">
              {[10, 25, 50, 75, 100, 125].map((lev) => (
                <button
                  key={lev}
                  type="button"
                  onClick={() => setLocalSettings({ ...localSettings, leverage: lev })}
                  className={`py-1.5 rounded-lg text-xs font-mono font-bold transition-colors ${
                    (localSettings.leverage || 100) === lev
                      ? "bg-amber-500 text-black shadow-lg shadow-amber-500/20"
                      : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 border border-zinc-700/60"
                  }`}
                >
                  {lev}x
                </button>
              ))}
            </div>
            <div className="flex justify-between text-[10px] text-zinc-500">
              <span>Tier 4: $5M max @ 20x</span>
              <span className="text-amber-400 font-semibold">Tier 2: $250k max @ 100x</span>
              <span>Tier 1: $50k max @ 125x</span>
            </div>
          </div>

          {/* 4. Execution Style & VIP Fee Tier */}
          <div className="grid grid-cols-2 gap-3 p-3 rounded-xl bg-zinc-900/40 border border-zinc-800">
            {/* Execution Style */}
            <div className="space-y-1">
              <label className="text-zinc-300 block font-semibold">Execution Style</label>
              <select
                value={localSettings.executionStyle || "SMART_HYBRID"}
                onChange={(e) =>
                  setLocalSettings({
                    ...localSettings,
                    executionStyle: e.target.value as ExecutionStyle,
                  })
                }
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-zinc-200 focus:outline-none focus:border-amber-500"
              >
                <option value="SMART_HYBRID">Smart Hybrid (Maker Entry / Taker Stops)</option>
                <option value="LIMIT_MAKER">Limit Maker (Post-Only, 0 Slippage)</option>
                <option value="MARKET_TAKER">Market Taker (Immediate Fill)</option>
              </select>
              <span className="text-[10px] text-zinc-500">
                Hybrid captures post-only rebates on entry while protecting exits.
              </span>
            </div>

            {/* VIP Fee Tier */}
            <div className="space-y-1">
              <label className="text-zinc-300 block font-semibold">Binance VIP Tier</label>
              <select
                value={localSettings.vipTier || "VIP0"}
                onChange={(e) => {
                  const newTier = e.target.value as VipTier;
                  const feeInfo = BINANCE_FUTURES_VIP_FEES[newTier];
                  const taker = localSettings.useBnbFeeDiscount ? feeInfo.takerBps * 0.9 : feeInfo.takerBps;
                  setLocalSettings({
                    ...localSettings,
                    vipTier: newTier,
                    takerFeeBps: taker,
                  });
                }}
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-zinc-200 focus:outline-none focus:border-amber-500"
              >
                {Object.entries(BINANCE_FUTURES_VIP_FEES).map(([tierKey, info]) => (
                  <option key={tierKey} value={tierKey}>
                    {info.name} ({info.makerBps}/{info.takerBps} bps)
                  </option>
                ))}
              </select>
              <span className="text-[10px] text-zinc-500">
                Determines exact exchange Maker/Taker commission deductions.
              </span>
            </div>

            {/* BNB Fee Discount Checkbox */}
            <div className="col-span-2 pt-1 flex items-center justify-between border-t border-zinc-800/80">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={localSettings.useBnbFeeDiscount ?? true}
                  onChange={(e) => {
                    const checked = e.target.checked;
                    const feeInfo = BINANCE_FUTURES_VIP_FEES[localSettings.vipTier || "VIP0"];
                    const taker = checked ? feeInfo.takerBps * 0.9 : feeInfo.takerBps;
                    setLocalSettings({
                      ...localSettings,
                      useBnbFeeDiscount: checked,
                      takerFeeBps: taker,
                    });
                  }}
                  className="rounded bg-zinc-950 border-zinc-700 text-amber-500 focus:ring-0 cursor-pointer"
                />
                <span className="text-zinc-300">
                  Use BNB deduction (10% Binance Futures fee discount)
                </span>
              </label>
              <span className="text-amber-400 font-bold">
                {localSettings.useBnbFeeDiscount ? "-10% ACTIVE" : "OFF"}
              </span>
            </div>
          </div>

          {/* 5. Triple Barrier & Scalping Barrier Tuning */}
          <div className="grid grid-cols-3 gap-2 p-3 rounded-xl bg-zinc-900/40 border border-zinc-800">
            <div className="space-y-1">
              <label className="text-zinc-400 block font-semibold">Take Profit (%)</label>
              <input
                type="number"
                step="0.01"
                min="0.02"
                max="1.0"
                value={localSettings.tpPct || 0.14}
                onChange={(e) =>
                  setLocalSettings({
                    ...localSettings,
                    tpPct: Math.max(0.01, parseFloat(e.target.value) || 0.14),
                  })
                }
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-2 py-1 text-zinc-200"
              />
              <span className="text-[9px] text-zinc-500">
                +{((localSettings.tpPct || 0.14) * (localSettings.leverage || 100)).toFixed(0)}% ROE
              </span>
            </div>

            <div className="space-y-1">
              <label className="text-zinc-400 block font-semibold">Stop Loss (%)</label>
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="1.0"
                value={localSettings.slPct || 0.12}
                onChange={(e) =>
                  setLocalSettings({
                    ...localSettings,
                    slPct: Math.max(0.01, parseFloat(e.target.value) || 0.12),
                  })
                }
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-2 py-1 text-zinc-200"
              />
              <span className="text-[9px] text-zinc-500">
                -{((localSettings.slPct || 0.12) * (localSettings.leverage || 100)).toFixed(0)}% ROE
              </span>
            </div>

            <div className="space-y-1">
              <label className="text-zinc-400 block font-semibold">Trailing Callback (%)</label>
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="0.5"
                value={localSettings.trailingStopCallbackPct || 0.05}
                onChange={(e) =>
                  setLocalSettings({
                    ...localSettings,
                    trailingStopCallbackPct: Math.max(0.01, parseFloat(e.target.value) || 0.05),
                  })
                }
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-2 py-1 text-zinc-200"
              />
              <span className="text-[9px] text-zinc-500">Trails favorable moves</span>
            </div>
          </div>

          {/* 6. Meta-Labeling Veto Threshold Slider */}
          <div className="space-y-2 p-3 rounded-xl bg-zinc-900/40 border border-zinc-800">
            <div className="flex justify-between items-center">
              <label className="text-zinc-300 font-semibold flex items-center space-x-1.5">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>Meta-Labeling Veto Gate (&tau;)</span>
              </label>
              <span className="text-amber-400 font-bold text-sm">
                {localSettings.metaThreshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.40"
              max="0.75"
              step="0.01"
              value={localSettings.metaThreshold}
              onChange={(e) => setLocalSettings({ ...localSettings, metaThreshold: parseFloat(e.target.value) })}
              className="w-full accent-amber-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-zinc-500">
              <span>0.50 (Aggressive)</span>
              <span className="text-emerald-400 font-semibold">0.65 (Balanced: +4.27 bps)</span>
              <span className="text-amber-400 font-bold">0.70 (Ultra Sniper: +7.96 bps)</span>
            </div>
            <div className="flex gap-2 pt-1.5">
              <button
                type="button"
                onClick={() => setLocalSettings({ ...localSettings, metaThreshold: 0.60 })}
                className={`flex-1 py-1 rounded text-[11px] font-mono border transition-all ${
                  Math.abs(localSettings.metaThreshold - 0.60) < 0.02
                    ? "bg-zinc-800 text-zinc-100 border-zinc-600 font-bold"
                    : "bg-zinc-950 text-zinc-400 border-zinc-800 hover:bg-zinc-900"
                }`}
              >
                0.60 Active (58% WR)
              </button>
              <button
                type="button"
                onClick={() => setLocalSettings({ ...localSettings, metaThreshold: 0.65 })}
                className={`flex-1 py-1 rounded text-[11px] font-mono border transition-all ${
                  Math.abs(localSettings.metaThreshold - 0.65) < 0.02
                    ? "bg-emerald-950/80 text-emerald-300 border-emerald-600 font-bold"
                    : "bg-zinc-950 text-zinc-400 border-zinc-800 hover:bg-zinc-900"
                }`}
              >
                0.65 Balanced (+4.3 bps)
              </button>
              <button
                type="button"
                onClick={() => setLocalSettings({ ...localSettings, metaThreshold: 0.70 })}
                className={`flex-1 py-1 rounded text-[11px] font-mono border transition-all ${
                  Math.abs(localSettings.metaThreshold - 0.70) < 0.02
                    ? "bg-amber-950/80 text-amber-300 border-amber-500 font-bold shadow-sm"
                    : "bg-zinc-950 text-zinc-400 border-zinc-800 hover:bg-zinc-900"
                }`}
              >
                🎯 0.70 Ultra Sniper (+8.0 bps)
              </button>
            </div>
          </div>

          {/* 7. Capital & Polling */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-zinc-400 block font-semibold">Starting Capital ($)</label>
              <input
                type="number"
                value={localSettings.startingCapital}
                onChange={(e) => setLocalSettings({ ...localSettings, startingCapital: parseFloat(e.target.value) || 10000 })}
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-1.5 text-zinc-200 focus:outline-none focus:border-amber-500"
              />
            </div>

            <div className="space-y-1">
              <label className="text-zinc-400 block font-semibold">Poll Interval (Seconds)</label>
              <input
                type="number"
                min="3"
                max="60"
                value={localSettings.pollIntervalSec}
                onChange={(e) => setLocalSettings({ ...localSettings, pollIntervalSec: parseInt(e.target.value) || 10 })}
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-1.5 text-zinc-200 focus:outline-none focus:border-amber-500"
              />
            </div>
          </div>

          {/* Reset Account Button */}
          <div className="pt-2 border-t border-zinc-800/80">
            <button
              type="button"
              onClick={() => {
                if (confirm("Reset paper trading portfolio balance back to starting capital?")) {
                  onResetAccount();
                  onClose();
                }
              }}
              className="w-full py-2 px-3 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-rose-400/90 border border-zinc-800 flex items-center justify-center space-x-2 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Portfolio & Trade History</span>
            </button>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-5 py-3 border-t border-zinc-800 flex justify-end space-x-2 bg-zinc-950">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs font-mono text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            className="px-4 py-1.5 rounded-lg text-xs font-mono font-semibold bg-amber-500 hover:bg-amber-400 text-black flex items-center space-x-1.5 transition-colors"
          >
            {savedSuccess ? <Check className="w-3.5 h-3.5" /> : null}
            <span>{savedSuccess ? "Saved!" : "Save Settings"}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
