"use client";

import React, { useState } from "react";
import { ModelAccuracyStats } from "@/types";
import {
  CheckCircle2,
  XCircle,
  TrendingUp,
  TrendingDown,
  Target,
  ShieldCheck,
  Activity,
  Layers,
  ArrowUpRight,
  ArrowDownRight,
  Award,
  Zap,
  RotateCcw,
} from "lucide-react";

interface ModelAccuracyCockpitProps {
  stats: ModelAccuracyStats;
  onReset?: () => void;
  livePrice?: number;
}

export const ModelAccuracyCockpit: React.FC<ModelAccuracyCockpitProps> = ({ stats, onReset, livePrice }) => {
  const [filterHorizon, setFilterHorizon] = useState<number | "ALL">("ALL");

  const filteredResolved = stats.recentResolved.filter((r) => {
    if (filterHorizon === "ALL") return true;
    return r.horizonSec === filterHorizon;
  });

  const alphaOverRandom = stats.overallAccuracy - 50.0;
  const metaFilterBoost = stats.authorizedAccuracy - stats.vetoedAccuracy;

  return (
    <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800 space-y-4">
      {/* 1. Header Section */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-3">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-lg bg-emerald-950/60 border border-emerald-800/60 text-emerald-400">
            <Target className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-sm uppercase font-mono font-bold text-zinc-100">
                Live Forward-Testing Accuracy Engine
              </h3>
              <span className="flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-950 text-emerald-400 border border-emerald-800/80 animate-pulse">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>FORWARD TESTING ON BINANCE WS</span>
              </span>
            </div>
            <p className="text-xs text-zinc-400 mt-0.5">
              Evaluating 12-stage deep learning forecasts against live Binance futures mark price movements.
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono text-zinc-400">
          <div className="px-2.5 py-1 rounded bg-zinc-950 border border-zinc-800">
            <span className="text-zinc-500">Evaluated Bars: </span>
            <span className="text-zinc-200 font-bold">{stats.totalEvaluated}</span>
          </div>
          <div className="px-2.5 py-1 rounded bg-zinc-950 border border-zinc-800">
            <span className="text-zinc-500">Alpha: </span>
            <span className={`font-bold ${alphaOverRandom >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
              {alphaOverRandom >= 0 ? "+" : ""}{alphaOverRandom.toFixed(1)}%
            </span>
          </div>
          {onReset && (
            <button
              onClick={onReset}
              title="Reset accuracy counters to 0 for a clean recording session"
              className="px-2 py-1 rounded bg-zinc-950 hover:bg-zinc-800 text-zinc-400 hover:text-amber-400 border border-zinc-800 transition-colors flex items-center space-x-1"
            >
              <RotateCcw className="w-3 h-3" />
              <span className="text-[11px] font-mono">Reset</span>
            </button>
          )}
        </div>
      </div>

      {/* 2. Primary KPI Gauge Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Card A: Overall Directional Hit Rate */}
        <div className="p-3.5 rounded-xl bg-zinc-950/60 border border-zinc-800/70 relative overflow-hidden flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-zinc-400 text-xs font-mono mb-1.5">
              <span>Overall Directional Hit Rate</span>
              <Award className="w-4 h-4 text-amber-400" />
            </div>
            <div className="flex items-baseline space-x-2">
              <span className="text-3xl font-mono font-extrabold text-zinc-100">
                {stats.overallAccuracy.toFixed(1)}%
              </span>
              <span className="text-xs font-mono text-emerald-400 font-semibold">
                +{alphaOverRandom.toFixed(1)}% vs Random
              </span>
            </div>
            <div className="w-full h-2 rounded-full bg-zinc-900 border border-zinc-800 mt-2 overflow-hidden">
              <div
                style={{ width: `${Math.min(stats.overallAccuracy, 100)}%` }}
                className="h-full bg-gradient-to-r from-amber-500 to-emerald-500 transition-all duration-500"
              />
            </div>
          </div>
          <div className="mt-3 text-[11px] font-mono text-zinc-500 flex justify-between">
            <span>{stats.correctPredictions} Hits / {stats.totalEvaluated} Forward Bars</span>
            <span className="text-zinc-400 font-semibold">Baseline: 50%</span>
          </div>
        </div>

        {/* Card B: Meta-Labeling Filter Precision Boost */}
        <div className="p-3.5 rounded-xl bg-zinc-950/60 border border-zinc-800/70 relative overflow-hidden flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-zinc-400 text-xs font-mono mb-1.5">
              <span>Meta-Labeling Veto Precision</span>
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="flex items-baseline space-x-2">
              <span className="text-3xl font-mono font-extrabold text-emerald-400">
                {stats.authorizedAccuracy.toFixed(1)}%
              </span>
              <span className="text-xs font-mono text-amber-400 font-semibold">
                +{metaFilterBoost.toFixed(1)}% Boost
              </span>
            </div>
            <div className="space-y-1 mt-2">
              <div className="flex justify-between text-[10px] font-mono text-zinc-400">
                <span>Authorized (Passed &tau;): <strong className="text-emerald-400">{stats.authorizedAccuracy.toFixed(1)}%</strong></span>
                <span>Vetoed: <strong className="text-zinc-500">{stats.vetoedAccuracy.toFixed(1)}%</strong></span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-zinc-900 border border-zinc-800 overflow-hidden flex">
                <div
                  style={{ width: `${stats.authorizedAccuracy}%` }}
                  className="bg-emerald-500"
                />
              </div>
            </div>
          </div>
          <div className="mt-2 text-[11px] font-mono text-zinc-500">
            {stats.authorizedCorrect}/{stats.authorizedEvaluated} trades profitable after veto filter
          </div>
        </div>

        {/* Card C: Long vs Short Precision Breakdown */}
        <div className="p-3.5 rounded-xl bg-zinc-950/60 border border-zinc-800/70 relative overflow-hidden flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-zinc-400 text-xs font-mono mb-1.5">
              <span>Directional Precision (Long vs Short)</span>
              <Activity className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="grid grid-cols-2 gap-2 mt-1">
              <div className="p-2 rounded bg-zinc-900/80 border border-zinc-800">
                <div className="flex items-center space-x-1 text-[10px] font-mono text-emerald-400">
                  <ArrowUpRight className="w-3 h-3" />
                  <span>LONG ACC</span>
                </div>
                <div className="text-lg font-mono font-bold text-zinc-100">
                  {stats.upAccuracy.toFixed(1)}%
                </div>
                <div className="text-[10px] font-mono text-zinc-500">
                  {stats.upCorrect}/{stats.upEvaluated} correct
                </div>
              </div>

              <div className="p-2 rounded bg-zinc-900/80 border border-zinc-800">
                <div className="flex items-center space-x-1 text-[10px] font-mono text-rose-400">
                  <ArrowDownRight className="w-3 h-3" />
                  <span>SHORT ACC</span>
                </div>
                <div className="text-lg font-mono font-bold text-zinc-100">
                  {stats.downAccuracy.toFixed(1)}%
                </div>
                <div className="text-[10px] font-mono text-zinc-500">
                  {stats.downCorrect}/{stats.downEvaluated} correct
                </div>
              </div>
            </div>
          </div>
          <div className="mt-2 text-[11px] font-mono text-zinc-500">
            Regime sensitivity matches directional flow
          </div>
        </div>

        {/* Card D: Multi-Horizon Accuracy Decay */}
        <div className="p-3.5 rounded-xl bg-zinc-950/60 border border-zinc-800/70 relative overflow-hidden flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-zinc-400 text-xs font-mono mb-1.5">
              <span>Horizon Alpha Decay</span>
              <Zap className="w-4 h-4 text-amber-400" />
            </div>
            <div className="grid grid-cols-3 gap-1.5 mt-1 text-center font-mono">
              <div className="p-1.5 rounded bg-zinc-900 border border-zinc-800">
                <span className="text-[10px] text-zinc-500 block">10s (1 Bar)</span>
                <span className="text-sm font-bold text-emerald-400">{stats.horizon10sAccuracy.toFixed(1)}%</span>
              </div>
              <div className="p-1.5 rounded bg-zinc-900 border border-zinc-800">
                <span className="text-[10px] text-zinc-500 block">30s (3 Bars)</span>
                <span className="text-sm font-bold text-zinc-200">{stats.horizon30sAccuracy.toFixed(1)}%</span>
              </div>
              <div className="p-1.5 rounded bg-zinc-900 border border-zinc-800">
                <span className="text-[10px] text-zinc-500 block">60s (6 Bars)</span>
                <span className="text-sm font-bold text-zinc-200">{stats.horizon60sAccuracy.toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div className="mt-3 text-[11px] font-mono text-zinc-500 flex justify-between">
            <span>Peak Micro-Alpha: 10s Horizon</span>
          </div>
        </div>
      </div>

      {/* 3. 8-Member Model Accuracy Leaderboard */}
      <div className="p-3.5 rounded-xl bg-zinc-950/60 border border-zinc-800">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-xs uppercase font-mono text-zinc-400 font-semibold flex items-center space-x-1.5">
            <Layers className="w-3.5 h-3.5 text-amber-400" />
            <span>8-Member Model Live Accuracy Leaderboard (Ranked on WebSocket Test Feed)</span>
          </span>
          <span className="text-[11px] font-mono text-zinc-500">Live Dual Tesla T4 Evaluation</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
          {Object.entries(stats.memberAccuracies)
            .sort((a, b) => b[1].accuracy - a[1].accuracy)
            .map(([mName, s]) => {
              const isTop = s.accuracy >= 70;
              return (
                <div
                  key={mName}
                  className={`p-2 rounded-lg border text-center font-mono ${
                    isTop
                      ? "bg-zinc-900/90 border-emerald-800/60 shadow"
                      : "bg-zinc-950/80 border-zinc-800/80"
                  }`}
                >
                  <div className="text-[10px] truncate text-zinc-400 mb-0.5" title={mName}>
                    {mName}
                  </div>
                  <div
                    className={`text-sm font-bold ${
                      s.accuracy >= 70
                        ? "text-emerald-400"
                        : s.accuracy >= 60
                        ? "text-zinc-200"
                        : "text-zinc-400"
                    }`}
                  >
                    {s.accuracy.toFixed(1)}%
                  </div>
                  <div className="text-[9px] text-zinc-500">
                    {s.correct}/{s.total} Hits
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* 4. Live Forward-Verification Log Stream Table */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <div className="flex items-center space-x-2">
            <span className="text-xs uppercase font-mono text-zinc-300 font-bold">
              Forward Verification Ledger (Real-Time Settlement on Binance Mark Price)
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-800 text-zinc-400">
              {filteredResolved.length} Resolved Events
            </span>
          </div>

          <div className="flex items-center space-x-1.5 text-xs font-mono">
            <span className="text-zinc-500 text-[11px]">Filter Horizon:</span>
            {(["ALL", 10, 30, 60] as const).map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => setFilterHorizon(h)}
                className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                  filterHorizon === h
                    ? "bg-amber-500 text-black font-bold"
                    : "bg-zinc-900 hover:bg-zinc-800 text-zinc-400 border border-zinc-800"
                }`}
              >
                {h === "ALL" ? "All" : `${h}s`}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto max-h-[260px] overflow-y-auto rounded-lg border border-zinc-800/80">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-zinc-950 sticky top-0 text-[10px] uppercase text-zinc-500 border-b border-zinc-800">
              <tr>
                <th className="py-2 px-2.5">Target Time</th>
                <th className="py-2 px-2.5">Horizon</th>
                <th className="py-2 px-2.5">Predicted Forecast</th>
                <th className="py-2 px-2.5">WebSocket Entry Mark</th>
                <th className="py-2 px-2.5">WebSocket Settlement</th>
                <th className="py-2 px-2.5">Price Delta (%)</th>
                <th className="py-2 px-2.5 text-right">Verification Outcome</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-900 bg-zinc-950/40 text-zinc-300">
              {/* Active In-Flight Evaluation Bar */}
              {stats.activePending && (
                <tr className="bg-amber-950/25 border-b border-amber-900/40">
                  <td className="py-2 px-3 text-amber-300 text-[11px] font-mono whitespace-nowrap">
                    <span className="flex items-center space-x-1.5">
                      <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                      <span>{new Date(stats.activePending.timestamp).toLocaleTimeString("en-US", { hour12: false })}</span>
                    </span>
                  </td>

                  <td className="py-2 px-3 text-amber-400 text-[11px] font-mono whitespace-nowrap">
                    <span className="px-2 py-0.5 rounded bg-amber-950/80 border border-amber-700/60 text-amber-300 font-semibold text-[10px]">
                      10s (In Flight)
                    </span>
                  </td>

                  <td className="py-2 px-2.5">
                    <span
                      className={`inline-flex items-center space-x-1 px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        stats.activePending.predictedDirection === 1
                          ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/50"
                          : stats.activePending.predictedDirection === -1
                          ? "bg-rose-950/80 text-rose-400 border border-rose-800/50"
                          : "bg-zinc-800 text-zinc-400"
                      }`}
                    >
                      {stats.activePending.predictedDirection === 1 ? (
                        <ArrowUpRight className="w-3 h-3" />
                      ) : stats.activePending.predictedDirection === -1 ? (
                        <ArrowDownRight className="w-3 h-3" />
                      ) : null}
                      <span>
                        {stats.activePending.predictedDirection === 1
                          ? "LONG (UP)"
                          : stats.activePending.predictedDirection === -1
                          ? "SHORT (DOWN)"
                          : "FLAT"}
                      </span>
                    </span>
                    <span className="text-[10px] text-zinc-400 ml-1.5 font-normal">
                      {(stats.activePending.confidence * 100).toFixed(0)}% Conf
                    </span>
                  </td>

                  <td className="py-2 px-2.5 font-semibold text-zinc-300">
                    ${stats.activePending.entryPrice.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>

                  <td className="py-2 px-2.5 font-semibold text-amber-300">
                    {livePrice ? `$${livePrice.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "Ticking..."}
                  </td>

                  <td className="py-2 px-2.5 font-semibold">
                    {livePrice && stats.activePending.entryPrice ? (
                      <span className={livePrice >= stats.activePending.entryPrice ? "text-emerald-400" : "text-rose-400"}>
                        {livePrice >= stats.activePending.entryPrice ? "+" : ""}
                        {(((livePrice - stats.activePending.entryPrice) / stats.activePending.entryPrice) * 100).toFixed(3)}%
                      </span>
                    ) : (
                      <span className="text-zinc-500">0.000%</span>
                    )}
                  </td>

                  <td className="py-2 px-2.5 text-right">
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950/80 text-amber-300 border border-amber-700/60">
                      <Activity className="w-3 h-3 animate-pulse" />
                      <span>EVALUATING</span>
                    </span>
                  </td>
                </tr>
              )}

              {filteredResolved.length === 0 && !stats.activePending ? (
                <tr>
                  <td colSpan={7} className="py-6 text-center text-zinc-500 text-xs">
                    Listening for incoming Binance WebSocket price settlements...
                  </td>
                </tr>
              ) : (
                filteredResolved.slice(0, 50).map((r) => {
                  const isLong = r.predictedDirection === 1;
                  const isShort = r.predictedDirection === -1;
                  const isPriceUp = r.priceChangePct >= 0;

                  return (
                    <tr key={r.id} className="hover:bg-zinc-800/30 transition-colors">
                      <td className="py-2 px-3 text-zinc-300 text-[11px] font-mono whitespace-nowrap">
                        {new Date(r.timestamp).toLocaleTimeString("en-US", { hour12: false })}
                      </td>

                      <td className="py-2 px-3 text-zinc-400 text-[11px] font-mono whitespace-nowrap">
                        <span className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-amber-400 font-semibold">
                          {r.horizonSec}s
                        </span>
                      </td>

                      <td className="py-2 px-2.5">
                        <span
                          className={`inline-flex items-center space-x-1 px-1.5 py-0.5 rounded text-[10px] font-bold ${
                            isLong
                              ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/50"
                              : isShort
                              ? "bg-rose-950/80 text-rose-400 border border-rose-800/50"
                              : "bg-zinc-800 text-zinc-400"
                          }`}
                        >
                          {isLong ? <ArrowUpRight className="w-3 h-3" /> : isShort ? <ArrowDownRight className="w-3 h-3" /> : null}
                          <span>{isLong ? "LONG (UP)" : isShort ? "SHORT (DOWN)" : "FLAT"}</span>
                        </span>
                        <span className="text-[10px] text-zinc-500 ml-1.5 font-normal">
                          {(r.confidence * 100).toFixed(0)}% Conf
                        </span>
                      </td>

                      <td className="py-2 px-2.5 font-semibold text-zinc-300">
                        ${r.entryPrice.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>

                      <td className="py-2 px-2.5 font-semibold text-zinc-200">
                        ${r.resolvedPrice.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>

                      <td className="py-2 px-2.5 font-semibold">
                        <span className={isPriceUp ? "text-emerald-400" : "text-rose-400"}>
                          {isPriceUp ? "+" : ""}{r.priceChangePct.toFixed(3)}%
                        </span>
                      </td>

                      <td className="py-2 px-2.5 text-right">
                        <span
                          className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold ${
                            r.isCorrect
                              ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
                              : "bg-rose-950 text-rose-300 border border-rose-800"
                          }`}
                        >
                          {r.isCorrect ? (
                            <>
                              <CheckCircle2 className="w-3 h-3" />
                              <span>HIT (CORRECT)</span>
                            </>
                          ) : (
                            <>
                              <XCircle className="w-3 h-3" />
                              <span>MISS (INCORRECT)</span>
                            </>
                          )}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
