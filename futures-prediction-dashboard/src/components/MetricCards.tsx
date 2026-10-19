"use client";

import React from "react";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Percent,
  Coins,
  ShieldCheck,
  AlertTriangle,
  Clock,
} from "lucide-react";
import { BINANCE_FUTURES_VIP_FEES, DashboardSettings, PaperAccount } from "@/types";

interface MetricCardsProps {
  account: PaperAccount;
  settings?: DashboardSettings;
}

export const MetricCards: React.FC<MetricCardsProps> = ({ account, settings }) => {
  const equity = account?.equity ?? 10000;
  const startingBalance = account?.starting_balance ?? 10000;
  const totalReturn = ((equity - startingBalance) / Math.max(1, startingBalance)) * 100;
  const isProfit = totalReturn >= 0;
  const unrealizedPnl = account?.unrealized_pnl ?? 0;
  const isUnrealizedProfit = unrealizedPnl >= 0;
  const realizedPnl = account?.realized_pnl ?? 0;
  const winCount = account?.win_count ?? 0;
  const lossCount = account?.loss_count ?? 0;
  const closedTrades = winCount + lossCount;
  const winRate = closedTrades > 0 ? (winCount / closedTrades) * 100 : 0;
  const roePct = account?.roe_pct ?? account?.unrealized_pnl_pct ?? 0;
  const positionSide = account?.position_side ?? "FLAT";
  const positionNotional = account?.position_notional ?? 0;
  const marginRatioPct = account?.margin_ratio_pct ?? 0;
  const maintenanceMargin = account?.maintenance_margin ?? 0;
  const cash = account?.cash ?? 10000;
  const totalFees = account?.total_fees ?? 0;
  const totalSlippage = account?.total_slippage ?? 0;
  const totalFundingFees = account?.total_funding_fees ?? 0;

  const vipTier = settings?.vipTier || "VIP0";
  const vipFeeInfo = BINANCE_FUTURES_VIP_FEES[vipTier] || BINANCE_FUTURES_VIP_FEES.VIP0;

  // Margin ratio health
  const isMarginSafe = marginRatioPct < 50;
  const isMarginWarning = marginRatioPct >= 50 && marginRatioPct < 80;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
      {/* 1. Total Margin Balance (Equity) */}
      <div className="p-3.5 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
        <div className="flex items-center justify-between text-zinc-400 mb-1">
          <span className="text-xs uppercase font-mono flex items-center space-x-1">
            {settings?.executionTarget === "BINANCE_TESTNET_LIVE" && (
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
            )}
            <span>{settings?.executionTarget === "BINANCE_TESTNET_LIVE" ? "Testnet Margin" : "Margin Balance"}</span>
          </span>
          <Coins className="w-4 h-4 text-amber-400" />
        </div>
        <div className="text-xl font-bold font-mono text-zinc-100">
          ${equity.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>
        <div className={`text-xs font-mono mt-1 flex items-center space-x-1 ${isProfit ? "text-emerald-400" : "text-rose-400"}`}>
          {isProfit ? <TrendingUp className="w-3 h-3 inline" /> : <TrendingDown className="w-3 h-3 inline" />}
          <span>
            {isProfit ? "+" : ""}{totalReturn.toFixed(2)}% total
          </span>
        </div>
      </div>

      {/* 2. Net Realized PnL */}
      <div className="p-3.5 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
        <div className="flex items-center justify-between text-zinc-400 mb-1">
          <span className="text-xs uppercase font-mono">Realized PnL (Net)</span>
          <DollarSign className="w-4 h-4 text-emerald-400" />
        </div>
        <div className={`text-xl font-bold font-mono ${realizedPnl >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
          {realizedPnl >= 0 ? "+" : ""}${realizedPnl.toFixed(2)}
        </div>
        <div className="text-xs font-mono text-zinc-400 mt-1">
          Win Rate: <span className="text-zinc-200 font-semibold">{winRate.toFixed(1)}%</span> ({winCount}W / {lossCount}L)
        </div>
      </div>

      {/* 3. Unrealized PnL & Binance ROE% */}
      <div className="p-3.5 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
        <div className="flex items-center justify-between text-zinc-400 mb-1">
          <span className="text-xs uppercase font-mono">Unrealized PnL</span>
          <Percent className="w-4 h-4 text-sky-400" />
        </div>
        <div className={`text-xl font-bold font-mono ${isUnrealizedProfit ? "text-emerald-400" : "text-rose-400"}`}>
          {isUnrealizedProfit ? "+" : ""}${unrealizedPnl.toFixed(2)}
        </div>
        <div className={`text-xs font-mono mt-1 font-semibold ${isUnrealizedProfit ? "text-emerald-400" : "text-rose-400"}`}>
          {isUnrealizedProfit ? "+" : ""}{roePct.toFixed(2)}% ROE
        </div>
      </div>

      {/* 4. Position Allocation & Leverage */}
      <div className="p-3.5 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
        <div className="flex items-center justify-between text-zinc-400 mb-1">
          <span className="text-xs uppercase font-mono">Position Exposure</span>
          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded font-mono ${
            positionSide === "LONG"
              ? "bg-emerald-950 text-emerald-400 border border-emerald-800/50"
              : positionSide === "SHORT"
              ? "bg-rose-950 text-rose-400 border border-rose-800/50"
              : "bg-zinc-800 text-zinc-400"
          }`}>
            {positionSide}
          </span>
        </div>
        <div className="text-xl font-bold font-mono text-zinc-100">
          ${positionNotional.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
        </div>
        <div className="text-xs font-mono text-zinc-400 mt-1">
          {positionNotional > 0
            ? `${(positionNotional / Math.max(equity, 1)).toFixed(1)}x Effective`
            : "Cash (0.0x)"}
        </div>
      </div>

      {/* 5. Binance Margin Ratio */}
      <div className="p-3.5 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
        <div className="flex items-center justify-between text-zinc-400 mb-1">
          <span className="text-xs uppercase font-mono">Margin Ratio</span>
          {isMarginSafe ? (
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          )}
        </div>
        <div className={`text-xl font-bold font-mono ${
          isMarginSafe ? "text-emerald-400" : isMarginWarning ? "text-amber-400" : "text-rose-400"
        }`}>
          {marginRatioPct.toFixed(2)}%
        </div>
        <div className="text-xs font-mono text-zinc-400 mt-1">
          MM: ${maintenanceMargin.toFixed(0)} | Avail: ${cash.toFixed(0)}
        </div>
      </div>

      {/* 6. Total Friction & VIP Fees & Funding */}
      <div className="p-3.5 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
        <div className="flex items-center justify-between text-zinc-400 mb-1">
          <span className="text-xs uppercase font-mono">Total Friction & Fees</span>
          <span className="text-[10px] font-mono text-zinc-400">
            {vipTier} ({vipFeeInfo.takerBps} bps)
          </span>
        </div>
        <div className="text-xl font-bold font-mono text-rose-400/90">
          -${(totalFees + totalSlippage).toFixed(2)}
        </div>
        <div className="text-xs font-mono text-zinc-400 mt-1 flex justify-between">
          <span>Fee: ${totalFees.toFixed(1)}</span>
          <span className={totalFundingFees >= 0 ? "text-emerald-400" : "text-rose-400"}>
            Fund: {totalFundingFees >= 0 ? "+" : ""}${totalFundingFees.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
};
