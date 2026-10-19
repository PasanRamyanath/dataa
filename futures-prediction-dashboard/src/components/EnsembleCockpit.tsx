"use client";

import React from "react";
import { EnsemblePrediction } from "@/types";
import { ArrowUp, ArrowDown, Minus, Cpu, Compass, Layers } from "lucide-react";

interface EnsembleCockpitProps {
  prediction: EnsemblePrediction | null;
}

export const EnsembleCockpit: React.FC<EnsembleCockpitProps> = ({ prediction }) => {
  if (!prediction) {
    return (
      <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800 flex items-center justify-center min-h-[260px]">
        <div className="text-zinc-500 font-mono text-xs animate-pulse">
          Loading 12-stage model ensemble...
        </div>
      </div>
    );
  }

  const memberEntries = Object.values(prediction.members || {});

  return (
    <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center space-x-2">
          <Cpu className="w-4 h-4 text-sky-400" />
          <h3 className="text-xs uppercase font-mono text-zinc-300 font-bold">
            12-Stage Ensemble ({memberEntries.length} Voting Models + 4 Fusion Stages)
          </h3>
        </div>

        {/* HMM Regime State Indicator */}
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-zinc-950 border border-zinc-800 text-xs font-mono">
          <Compass className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-zinc-400">Regime:</span>
          <span
            className={`font-semibold ${
              prediction.regime_state === 0
                ? "text-emerald-400"
                : prediction.regime_state === 1
                ? "text-sky-400"
                : "text-amber-400"
            }`}
          >
            {prediction.regime_name}
          </span>
          <span className="text-zinc-500 text-[10px]">
            (M={prediction.regime_multiplier}x)
          </span>
        </div>
      </div>

      {/* Grid of the 8 Directional Members */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-4 gap-2.5">
        {memberEntries.map((member) => {
          const isUp = member.primary_direction === 1;
          const isDown = member.primary_direction === -1;
          const isFlat = member.primary_direction === 0;

          return (
            <div
              key={member.name}
              className="p-2.5 rounded-lg bg-zinc-950/60 border border-zinc-800/60 hover:border-zinc-700/80 transition-all flex flex-col justify-between"
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-mono font-semibold text-zinc-200 truncate">
                  {member.name}
                </span>
                <span
                  className={`p-1 rounded text-[10px] font-mono font-bold flex items-center ${
                    isUp
                      ? "bg-emerald-950 text-emerald-400 border border-emerald-800/60"
                      : isDown
                      ? "bg-rose-950 text-rose-400 border border-rose-800/60"
                      : "bg-zinc-800 text-zinc-400"
                  }`}
                >
                  {isUp ? <ArrowUp className="w-3 h-3" /> : isDown ? <ArrowDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
                </span>
              </div>

              {/* Probabilities Mini Bar */}
              <div className="space-y-1">
                <div className="w-full h-1.5 rounded-full bg-zinc-900 flex overflow-hidden">
                  <div style={{ width: `${member.prob_down * 100}%` }} className="bg-rose-500" />
                  <div style={{ width: `${member.prob_flat * 100}%` }} className="bg-zinc-700" />
                  <div style={{ width: `${member.prob_up * 100}%` }} className="bg-emerald-500" />
                </div>

                <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500">
                  <span>D: {(member.prob_down * 100).toFixed(0)}%</span>
                  <span>U: {(member.prob_up * 100).toFixed(0)}%</span>
                  <span>{member.latency_ms}ms</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
