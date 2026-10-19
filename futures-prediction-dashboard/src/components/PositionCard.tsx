"use client";

import React from "react";
import {
  DashboardSettings,
  ExecutionRole,
  PaperAccount,
  PaperTrade,
  getBtcLeverageBracket,
} from "@/types";
import {
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  ShieldCheck,
  Target,
  AlertTriangle,
  Flame,
  Activity,
  Layers,
  Zap,
  TrendingDown,
  XCircle,
} from "lucide-react";

interface PositionCardProps {
  account: PaperAccount;
  lastTrade: PaperTrade | null;
  onManualClose?: (price: number, role: ExecutionRole) => void;
  settings?: DashboardSettings;
}

export const PositionCard: React.FC<PositionCardProps> = ({
  account,
  lastTrade,
  onManualClose,
  settings,
}) => {
  const isFlat = account.position_side === "FLAT" || !account.entry_price;
  const isLong = account.position_side === "LONG";
  const pnl = account.unrealized_pnl;
  const roePct = account.roe_pct ?? account.unrealized_pnl_pct;
  const isProfit = pnl >= 0;

  const tpPctRatio = (settings?.tpPct || 0.14) / 100;
  const slPctRatio = (settings?.slPct || 0.12) / 100;

  // Exact TP / SL prices from account
  const tpPrice = account.tp_price ?? (account.entry_price ? (isLong ? account.entry_price * (1.0 + tpPctRatio) : account.entry_price * (1.0 - tpPctRatio)) : 0);
  const slPrice = account.sl_price ?? (account.entry_price ? (isLong ? account.entry_price * (1.0 - slPctRatio) : account.entry_price * (1.0 + slPctRatio)) : 0);

  const priceMovePct =
    account.entry_price && account.current_price
      ? isLong
        ? ((account.current_price - account.entry_price) / account.entry_price) * 100
        : ((account.entry_price - account.current_price) / account.entry_price) * 100
      : 0;

  const effectiveLeverage =
    account.equity > 0 && account.position_notional > 0
      ? account.position_notional / account.equity
      : account.leverage || settings?.leverage || 100;

  // Normalized progress between SL (-slPct) and TP (+tpPct) -> 0% to 100%
  const tpPctVal = settings?.tpPct || 0.14;
  const slPctVal = settings?.slPct || 0.12;
  const clampedMove = Math.min(Math.max(priceMovePct, -slPctVal), tpPctVal);
  const barrierProgress = ((clampedMove + slPctVal) / (tpPctVal + slPctVal)) * 100;

  // Dynamic Binance Leverage & Margin Bracket info
  const bracket = getBtcLeverageBracket(account.position_notional || 0);

  // Margin ratio health
  const marginRatio = account.margin_ratio_pct ?? 0;
  const marginColor =
    marginRatio > 80
      ? "text-rose-400 bg-rose-950 border-rose-800"
      : marginRatio > 50
      ? "text-amber-400 bg-amber-950 border-amber-800"
      : "text-emerald-400 bg-emerald-950 border-emerald-800";

  // Liquidation distance
  const liqDistancePct =
    account.liquidation_price && account.current_price
      ? Math.abs((account.current_price - account.liquidation_price) / account.current_price) * 100
      : null;

  const marginMode = account.margin_mode || settings?.marginMode || "ISOLATED";

  return (
    <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800 flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <span className="text-xs uppercase font-mono text-zinc-400 flex items-center space-x-1.5">
              {settings?.executionTarget === "BINANCE_TESTNET_LIVE" && (
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              )}
              <span>
                {settings?.executionTarget === "BINANCE_TESTNET_LIVE"
                  ? "Binance Testnet Position"
                  : "Binance USDⓈ-M Position"}
              </span>
            </span>
            <span
              className={`px-2 py-0.5 rounded text-xs font-mono font-bold flex items-center space-x-1 ${
                isFlat
                  ? "bg-zinc-800 text-zinc-400 border border-zinc-700"
                  : isLong
                  ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                  : "bg-rose-950 text-rose-400 border border-rose-800"
              }`}
            >
              {isFlat ? (
                <>
                  <Minus className="w-3 h-3" />
                  <span>FLAT (CASH)</span>
                </>
              ) : isLong ? (
                <>
                  <ArrowUpRight className="w-3.5 h-3.5" />
                  <span>LONG BTCUSDT</span>
                </>
              ) : (
                <>
                  <ArrowDownRight className="w-3.5 h-3.5" />
                  <span>SHORT BTCUSDT</span>
                </>
              )}
            </span>
            {!isFlat && (
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold border ${
                  marginMode === "ISOLATED"
                    ? "bg-sky-500/20 text-sky-300 border-sky-500/40"
                    : "bg-purple-500/20 text-purple-300 border-purple-500/40"
                }`}
              >
                {effectiveLeverage.toFixed(1)}x {marginMode}
              </span>
            )}
          </div>

          <div className="text-xs font-mono text-zinc-400">
            {isFlat ? (
              <span className="text-zinc-500">100% USDT Balance</span>
            ) : (
              <span>
                <strong className="text-zinc-200">{account.position_size.toFixed(3)} BTC</strong>{" "}
                <span className="text-zinc-500">
                  (${account.position_notional?.toLocaleString("en-US", { maximumFractionDigits: 0 })})
                </span>
              </span>
            )}
          </div>
        </div>

        {/* Body Content */}
        {isFlat ? (
          <div className="py-6 text-center">
            <div className="inline-flex p-3 rounded-full bg-zinc-800/60 border border-zinc-700/60 text-zinc-400 mb-2">
              <ShieldCheck className="w-6 h-6 text-emerald-400" />
            </div>
            <h4 className="text-sm font-semibold text-zinc-200">100% In Cash (Capital Preserved)</h4>
            <p className="text-xs text-zinc-400 max-w-sm mx-auto mt-1">
              Binance USDⓈ-M margin balance safely in USDT. Tier 1 capacity: $50,000 max notional @ 125x / 0.40% MMR.
              Awaiting signal passing meta-labeling veto gate (&tau; = {settings?.metaThreshold.toFixed(2) || "0.55"}).
            </p>
            {lastTrade && (
              <div className="mt-3 text-[11px] font-mono text-zinc-400">
                Last Closed: <span className="text-zinc-200">{lastTrade.action} {lastTrade.side}</span> • Exit:{" "}
                <span className="text-amber-400 font-semibold">{lastTrade.reason}</span>
                {lastTrade.pnl !== undefined && (
                  <span className={lastTrade.pnl >= 0 ? " text-emerald-400 font-bold" : " text-rose-400 font-bold"}>
                    {" "}({lastTrade.pnl >= 0 ? "+" : ""}${lastTrade.pnl.toFixed(2)})
                  </span>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            {/* Price & PnL Overview Grid */}
            <div className="grid grid-cols-2 gap-2.5 p-3 rounded-lg bg-zinc-950/60 border border-zinc-800/60">
              <div>
                <span className="text-[10px] uppercase font-mono text-zinc-500">Entry / Live Mark</span>
                <div className="text-sm font-mono font-bold text-zinc-200">
                  ${account.entry_price?.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  <span className="text-xs text-zinc-400 font-normal ml-1">
                    / ${account.current_price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </span>
                </div>
              </div>
              <div className="text-right">
                <span className="text-[10px] uppercase font-mono text-zinc-500">Unrealized PnL (ROE%)</span>
                <div className={`text-sm font-mono font-bold ${isProfit ? "text-emerald-400" : "text-rose-400"}`}>
                  {isProfit ? "+" : ""}${pnl.toFixed(2)}{" "}
                  <span className="text-xs px-1.5 py-0.2 rounded bg-zinc-900 border border-zinc-800">
                    {isProfit ? "+" : ""}{roePct.toFixed(2)}% ROE
                  </span>
                </div>
              </div>
            </div>

            {/* Trailing Stop Status Pill if Active */}
            {account.trailing_stop_active && (
              <div className="p-2 rounded-lg bg-cyan-950/40 border border-cyan-800/60 flex items-center justify-between text-xs font-mono">
                <div className="flex items-center space-x-1.5 text-cyan-400 font-semibold">
                  <Flame className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
                  <span>TRAILING STOP ACTIVE</span>
                </div>
                <div className="text-[11px] text-zinc-300">
                  Stop: <span className="font-bold text-cyan-300">${account.trailing_stop_price?.toFixed(2)}</span> · Peak: ${account.trailing_stop_peak_price?.toFixed(2)}
                </div>
              </div>
            )}

            {/* Binance Isolated Risk & Margin Grid */}
            <div className="grid grid-cols-3 gap-2 text-xs font-mono">
              <div className="p-2 rounded bg-zinc-950/70 border border-zinc-800/80">
                <span className="text-[10px] text-zinc-500 uppercase block">Liq Price</span>
                <span className="font-bold text-rose-400 text-xs">
                  {account.liquidation_price
                    ? `$${account.liquidation_price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                    : "Safe"}
                </span>
                {liqDistancePct !== null && (
                  <span className="text-[9px] text-zinc-500 block">
                    {liqDistancePct.toFixed(2)}% buffer
                  </span>
                )}
              </div>

              <div className="p-2 rounded bg-zinc-950/70 border border-zinc-800/80">
                <span className="text-[10px] text-zinc-500 uppercase block">Margin Ratio</span>
                <span className={`px-1 rounded text-[11px] font-bold border ${marginColor}`}>
                  {marginRatio.toFixed(2)}%
                </span>
                <span className="text-[9px] text-zinc-500 block">
                  {marginRatio < 50 ? "Healthy" : "Attention"}
                </span>
              </div>

              <div className="p-2 rounded bg-zinc-950/70 border border-zinc-800/80">
                <span className="text-[10px] text-zinc-500 uppercase block">
                  {marginMode === "ISOLATED" ? "Isolated Margin" : "Maint. Margin"}
                </span>
                <span className="font-bold text-zinc-200 text-xs">
                  ${(marginMode === "ISOLATED" ? account.isolated_margin : account.maintenance_margin).toFixed(2)}
                </span>
                <span className="text-[9px] text-zinc-500 block">
                  Tier {bracket.tier} ({(bracket.mmr * 100).toFixed(2)}% MMR)
                </span>
              </div>
            </div>

            {/* Triple Barrier Gauge */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-[11px] font-mono">
                <span className="text-rose-400 flex items-center space-x-1">
                  <AlertTriangle className="w-3 h-3 inline" />
                  <span>SL: ${slPrice.toFixed(2)} (-0.05%)</span>
                </span>
                <span className="text-zinc-300 font-semibold">
                  Move: {priceMovePct >= 0 ? "+" : ""}{priceMovePct.toFixed(3)}%
                </span>
                <span className="text-emerald-400 flex items-center space-x-1">
                  <Target className="w-3 h-3 inline" />
                  <span>TP: ${tpPrice.toFixed(2)} (+0.08%)</span>
                </span>
              </div>

              {/* Visual Barrier Range Track */}
              <div className="relative w-full h-3 rounded-full bg-zinc-950 border border-zinc-800 overflow-hidden">
                <div
                  className="absolute top-0 bottom-0 left-0 bg-gradient-to-r from-rose-500/80 via-zinc-600 to-emerald-500/80 transition-all duration-300"
                  style={{ width: `${Math.max(Math.min(barrierProgress, 100), 0)}%` }}
                />
                {/* Breakeven marker (0.00% move) at 38.46% position */}
                <div className="absolute top-0 bottom-0 left-[38.46%] w-0.5 bg-zinc-200/60 z-10" />
              </div>
            </div>

            {/* Manual Emergency / Quick Close Action Buttons */}
            {onManualClose && (
              <div className="pt-1 flex items-center space-x-2">
                <button
                  type="button"
                  onClick={() => onManualClose(account.current_price, "TAKER")}
                  className="flex-1 py-1.5 px-2 rounded-lg bg-rose-950/60 hover:bg-rose-900 border border-rose-800/80 text-rose-300 text-xs font-mono font-semibold flex items-center justify-center space-x-1 transition-colors"
                >
                  <XCircle className="w-3.5 h-3.5" />
                  <span>Market Close (Taker)</span>
                </button>
                <button
                  type="button"
                  onClick={() => onManualClose(account.current_price, "MAKER")}
                  className="flex-1 py-1.5 px-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs font-mono font-semibold flex items-center justify-center space-x-1 transition-colors"
                >
                  <Zap className="w-3.5 h-3.5 text-amber-400" />
                  <span>Limit Close (Maker)</span>
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="pt-3 border-t border-zinc-800/80 flex items-center justify-between text-[11px] font-mono text-zinc-500">
        <span>Binance Tier {bracket.tier}: Cap ${bracket.maxNotional === Infinity ? "No Limit" : bracket.maxNotional.toLocaleString()}</span>
        <span>
          {settings?.executionTarget === "BINANCE_TESTNET_LIVE" ? "🟢 Binance Testnet GTX Engine" : "Offline Simulator"} · Lot: 0.001 BTC
        </span>
      </div>
    </div>
  );
};
