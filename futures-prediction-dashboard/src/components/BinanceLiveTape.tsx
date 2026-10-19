"use client";

import React from "react";
import { AggTrade, BinanceTicker } from "@/types";
import { Radio, ArrowUp, ArrowDown } from "lucide-react";

interface BinanceLiveTapeProps {
  ticker: BinanceTicker | null;
  trades: AggTrade[];
}

export const BinanceLiveTape: React.FC<BinanceLiveTapeProps> = ({ ticker, trades }) => {
  return (
    <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800 flex flex-col h-full">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Radio className="w-4 h-4 text-emerald-400 animate-pulse" />
          <h3 className="text-xs uppercase font-mono text-zinc-300 font-bold">
            Binance Live Order Flow Tape
          </h3>
        </div>
        <span className="text-[10px] font-mono text-zinc-500">
          fstream.binance.com @aggTrade
        </span>
      </div>

      {/* Best Bid / Ask Card */}
      {ticker && ticker.bidPrice > 0 && (
        <div className="grid grid-cols-2 gap-2 p-2.5 rounded-lg bg-zinc-950/70 border border-zinc-800/80 mb-3 text-xs font-mono">
          <div>
            <span className="text-zinc-500 block text-[10px]">Best Bid (Buy)</span>
            <span className="text-emerald-400 font-bold">${ticker.bidPrice.toFixed(2)}</span>
            <span className="text-zinc-500 text-[10px] ml-1">({ticker.bidQty.toFixed(3)} BTC)</span>
          </div>
          <div className="text-right">
            <span className="text-zinc-500 block text-[10px]">Best Ask (Sell)</span>
            <span className="text-rose-400 font-bold">${ticker.askPrice.toFixed(2)}</span>
            <span className="text-zinc-500 text-[10px] ml-1">({ticker.askQty.toFixed(3)} BTC)</span>
          </div>
        </div>
      )}

      {/* Scrolling Trade Stream */}
      <div className="flex-1 overflow-hidden relative">
        <div className="space-y-1.5 overflow-y-auto max-h-[220px] pr-1 font-mono text-xs">
          {trades.length === 0 ? (
            <div className="text-center py-8 text-zinc-500 text-xs">
              Connecting to Binance Futures tape...
            </div>
          ) : (
            trades.slice(0, 15).map((t, idx) => {
              const isBuy = !t.isBuyerMaker; // taker bought = green
              return (
                <div
                  key={`${t.tradeId}-${idx}`}
                  className="flex items-center justify-between py-1 px-2 rounded bg-zinc-950/40 border border-zinc-900/60 hover:bg-zinc-900/60 transition-colors"
                >
                  <span className={`font-semibold flex items-center space-x-1 ${isBuy ? "text-emerald-400" : "text-rose-400"}`}>
                    {isBuy ? <ArrowUp className="w-3 h-3 inline" /> : <ArrowDown className="w-3 h-3 inline" />}
                    <span>${t.price.toFixed(2)}</span>
                  </span>

                  <span className="text-zinc-300">
                    {t.quantity.toFixed(3)} BTC
                  </span>

                  <span className="text-[10px] text-zinc-500">
                    {new Date(t.timestamp).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
