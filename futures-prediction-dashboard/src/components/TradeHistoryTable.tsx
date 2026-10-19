"use client";

import React, { useState } from "react";
import { PaperTrade } from "@/types";
import { ArrowUpRight, ArrowDownRight, Filter, Clock, Flame, ShieldAlert } from "lucide-react";

interface TradeHistoryTableProps {
  trades: PaperTrade[];
}

export const TradeHistoryTable: React.FC<TradeHistoryTableProps> = ({ trades }) => {
  const [filterReason, setFilterReason] = useState<string>("ALL");

  const filteredTrades = trades.filter((t) => {
    if (filterReason === "ALL") return true;
    return t.reason === filterReason;
  });

  return (
    <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center space-x-2">
          <h3 className="text-xs uppercase font-mono text-zinc-300 font-bold">
            Simulated Trade Ledger & Execution Log (Binance USDⓈ-M Settlement)
          </h3>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-800 text-zinc-400">
            {trades.length} Executed Events
          </span>
        </div>

        {/* Reason Filter */}
        <div className="flex items-center space-x-2 text-xs font-mono">
          <Filter className="w-3.5 h-3.5 text-zinc-400" />
          <select
            value={filterReason}
            onChange={(e) => setFilterReason(e.target.value)}
            className="bg-zinc-950 border border-zinc-800 rounded-lg px-2.5 py-1 text-zinc-300 text-xs focus:outline-none focus:border-amber-500"
          >
            <option value="ALL">All Events</option>
            <option value="SIGNAL_ENTRY">Signal Entry</option>
            <option value="TAKE_PROFIT">Take Profit (+0.08%)</option>
            <option value="STOP_LOSS">Stop Loss (-0.05%)</option>
            <option value="TRAILING_STOP">Trailing Stop Triggered</option>
            <option value="FUNDING_FEE">8h Funding Fee Settlement</option>
            <option value="VETO_EXIT">Meta-Labeling Veto Exit</option>
            <option value="MANUAL_CLOSE">Manual Position Close</option>
            <option value="LIQUIDATION">Liquidation (Insurance Clearance)</option>
          </select>
        </div>
      </div>

      <div className="overflow-x-auto max-h-[320px] overflow-y-auto pr-1">
        <table className="w-full text-left font-mono text-xs">
          <thead className="bg-zinc-950/80 sticky top-0 text-[10px] uppercase text-zinc-500 border-b border-zinc-800/80">
            <tr>
              <th className="py-2 px-2">Time (UTC)</th>
              <th className="py-2 px-2">Action / Side</th>
              <th className="py-2 px-2">Role</th>
              <th className="py-2 px-2">Exec Price</th>
              <th className="py-2 px-2">Size / Notional</th>
              <th className="py-2 px-2">Friction Fee</th>
              <th className="py-2 px-2">Net Realized PnL</th>
              <th className="py-2 px-2">Event Reason</th>
              <th className="py-2 px-2 text-right">Portfolio Balance</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-900/80 text-zinc-300">
            {filteredTrades.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-8 text-center text-zinc-500 text-xs">
                  No execution events recorded yet. Waiting for signals passing meta-labeling conviction threshold...
                </td>
              </tr>
            ) : (
              filteredTrades.map((t) => {
                const isBuy = t.side === "BUY";
                const netProfit = t.pnl ?? 0;
                const isProfitable = netProfit > 0;
                const totalFriction = t.fee + t.slippage;

                return (
                  <tr key={t.id} className="hover:bg-zinc-800/30 transition-colors">
                    <td className="py-2 px-2 text-zinc-400 text-[11px] font-mono whitespace-nowrap">
                      {new Date(t.timestamp).toLocaleTimeString("en-US", { hour12: false })}
                    </td>

                    <td className="py-2 px-2">
                      <span
                        className={`inline-flex items-center space-x-1 px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          t.reason === "FUNDING_FEE"
                            ? "bg-amber-950/80 text-amber-300 border border-amber-800/50"
                            : isBuy
                            ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/50"
                            : "bg-rose-950/80 text-rose-400 border border-rose-800/50"
                        }`}
                      >
                        {t.reason === "FUNDING_FEE" ? (
                          <Clock className="w-3 h-3" />
                        ) : isBuy ? (
                          <ArrowUpRight className="w-3 h-3" />
                        ) : (
                          <ArrowDownRight className="w-3 h-3" />
                        )}
                        <span>{t.action} {t.side}</span>
                      </span>
                    </td>

                    {/* Execution Role: Maker vs Taker */}
                    <td className="py-2 px-2">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[9px] font-bold border ${
                          t.role === "MAKER"
                            ? "bg-emerald-950/50 text-emerald-300 border-emerald-700/60"
                            : "bg-zinc-800 text-zinc-400 border-zinc-700"
                        }`}
                      >
                        {t.role || "TAKER"}
                      </span>
                      {t.executionTarget === "BINANCE_TESTNET_LIVE" && (
                        <span className="block mt-0.5 text-[8px] font-mono text-emerald-400 font-semibold">
                          TESTNET
                        </span>
                      )}
                    </td>

                    <td className="py-2 px-2 font-semibold">
                      ${t.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>

                    <td className="py-2 px-2 text-zinc-300">
                      <span>{t.size.toFixed(3)} BTC</span>{" "}
                      <span className="text-[10px] text-zinc-500">
                        (${t.notional.toLocaleString("en-US", { maximumFractionDigits: 0 })})
                      </span>
                    </td>

                    {/* Friction Fee */}
                    <td className="py-2 px-2 text-rose-400/80 text-[11px]">
                      {totalFriction > 0 ? `-$${totalFriction.toFixed(2)}` : "$0.00"}
                    </td>

                    {/* Net Realized PnL */}
                    <td className="py-2 px-2">
                      {t.pnl !== undefined ? (
                        <span className={`font-semibold ${isProfitable ? "text-emerald-400" : "text-rose-400"}`}>
                          {isProfitable ? "+" : ""}${netProfit.toFixed(2)}
                          {t.pnl_pct !== undefined && (
                            <span className="text-[10px] ml-1">
                              ({isProfitable ? "+" : ""}{t.pnl_pct.toFixed(2)}%)
                            </span>
                          )}
                        </span>
                      ) : (
                        <span className="text-zinc-600">—</span>
                      )}
                    </td>

                    {/* Event Reason Badge */}
                    <td className="py-2 px-2">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[10px] flex items-center space-x-1 w-fit ${
                          t.reason === "TAKE_PROFIT"
                            ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
                            : t.reason === "STOP_LOSS"
                            ? "bg-rose-950 text-rose-300 border border-rose-800"
                            : t.reason === "TRAILING_STOP"
                            ? "bg-cyan-950 text-cyan-300 border border-cyan-800 font-semibold"
                            : t.reason === "LIQUIDATION"
                            ? "bg-rose-900 text-rose-100 border border-rose-700 font-bold animate-pulse"
                            : t.reason === "FUNDING_FEE"
                            ? "bg-amber-950 text-amber-300 border border-amber-800"
                            : t.reason === "VETO_EXIT"
                            ? "bg-amber-950/60 text-amber-300 border border-amber-800/80"
                            : t.reason === "MANUAL_CLOSE"
                            ? "bg-purple-950 text-purple-300 border border-purple-800"
                            : "bg-zinc-800 text-zinc-400 border border-zinc-700"
                        }`}
                      >
                        {t.reason === "TRAILING_STOP" && <Flame className="w-2.5 h-2.5 text-cyan-400 inline mr-0.5" />}
                        {t.reason === "LIQUIDATION" && <ShieldAlert className="w-2.5 h-2.5 text-rose-300 inline mr-0.5" />}
                        <span>{t.reason}</span>
                      </span>
                    </td>

                    <td className="py-2 px-2 text-right font-semibold text-zinc-200">
                      ${t.equity_after?.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) ?? "—"}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
