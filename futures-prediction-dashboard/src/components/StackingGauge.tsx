"use client";

import React from "react";
import { EnsemblePrediction } from "@/types";
import { CheckCircle2, XCircle, ShieldCheck, Gauge, Layers } from "lucide-react";

interface StackingGaugeProps {
  prediction: EnsemblePrediction | null;
}

export const StackingGauge: React.FC<StackingGaugeProps> = ({ prediction }) => {
  if (!prediction) {
    return (
      <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800 flex items-center justify-center min-h-[200px]">
        <div className="text-center text-zinc-500 font-mono text-xs animate-pulse">
          Waiting for 10-second ensemble prediction...
        </div>
      </div>
    );
  }

  const pDown = prediction.calibrated_proba[0] * 100;
  const pFlat = prediction.calibrated_proba[1] * 100;
  const pUp = prediction.calibrated_proba[2] * 100;

  const rawDown = prediction.raw_proba[0] * 100;
  const rawFlat = prediction.raw_proba[1] * 100;
  const rawUp = prediction.raw_proba[2] * 100;

  const tau = prediction.meta_threshold;
  const actP = prediction.act_probability;
  const isAuthorized = prediction.is_trade_allowed;

  return (
    <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <span className="text-xs uppercase font-mono text-zinc-400">Level-2 Fusion & Veto Gate</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700">
              Dirichlet ODIR
            </span>
          </div>
          <span className="text-[11px] font-mono text-zinc-500">
            ECE &lt; 6.22%
          </span>
        </div>

        {/* 3-Class Probability Simplex Bar */}
        <div className="space-y-1.5 mb-4">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-rose-400 font-semibold">Down: {pDown.toFixed(1)}%</span>
            <span className="text-zinc-400">Flat: {pFlat.toFixed(1)}%</span>
            <span className="text-emerald-400 font-semibold">Up: {pUp.toFixed(1)}%</span>
          </div>

          <div className="w-full h-3 rounded-full bg-zinc-950 border border-zinc-800 flex overflow-hidden">
            <div style={{ width: `${pDown}%` }} className="bg-rose-500 transition-all duration-300" title={`Down: ${pDown.toFixed(1)}%`} />
            <div style={{ width: `${pFlat}%` }} className="bg-zinc-700 transition-all duration-300" title={`Flat: ${pFlat.toFixed(1)}%`} />
            <div style={{ width: `${pUp}%` }} className="bg-emerald-500 transition-all duration-300" title={`Up: ${pUp.toFixed(1)}%`} />
          </div>

          <div className="flex justify-between text-[10px] font-mono text-zinc-500">
            <span>Raw: {rawDown.toFixed(1)}%</span>
            <span>Raw: {rawFlat.toFixed(1)}%</span>
            <span>Raw: {rawUp.toFixed(1)}%</span>
          </div>
          <div className="text-[10px] font-mono text-zinc-500/80 pt-0.5">
            * Triple-barrier returns resolve to Up/Down. FLAT (Cash) is determined by the Meta-Filter below.
          </div>
        </div>

        {/* Meta-Labeling Veto Gate Status */}
        <div className="p-3 rounded-lg bg-zinc-950/70 border border-zinc-800/80 space-y-2 mb-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-zinc-400 flex items-center space-x-1.5">
              <ShieldCheck className="w-4 h-4 text-amber-400" />
              <span>Meta-Labeling Veto Gate (&tau; = {tau.toFixed(2)})</span>
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold flex items-center space-x-1 ${
                isAuthorized
                  ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                  : "bg-amber-950/60 text-amber-400 border border-amber-800/50"
              }`}
            >
              {isAuthorized ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>AUTHORIZED (79.2% PREC)</span>
                </>
              ) : (
                <>
                  <XCircle className="w-3.5 h-3.5" />
                  <span>VETOED (NOISE GATE)</span>
                </>
              )}
            </span>
          </div>

          {/* Act Probability Progress Gauge */}
          <div className="space-y-1">
            <div className="flex justify-between text-[11px] font-mono">
              <span className="text-zinc-500">P(Act): <strong className={isAuthorized ? "text-emerald-400" : "text-amber-400"}>{(actP * 100).toFixed(1)}%</strong></span>
              <span className="text-zinc-500">Threshold: {(tau * 100).toFixed(0)}%</span>
            </div>

            <div className="relative w-full h-2 rounded-full bg-zinc-900 border border-zinc-800 overflow-hidden">
              <div
                style={{ width: `${Math.min(actP * 100, 100)}%` }}
                className={`h-full transition-all duration-300 ${
                  isAuthorized ? "bg-emerald-500" : "bg-amber-500/80"
                }`}
              />
              {/* Vertical line at tau threshold */}
              <div
                style={{ left: `${tau * 100}%` }}
                className="absolute top-0 bottom-0 w-0.5 bg-amber-400 z-10"
                title={`Veto Gate Threshold tau = ${tau}`}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Target Sizing Stats */}
      <div className="pt-3 border-t border-zinc-800/80 grid grid-cols-3 gap-2 text-[11px] font-mono text-zinc-400">
        <div>
          <span className="text-zinc-500 block">Confidence</span>
          <span className="font-semibold text-zinc-200">{(prediction.confidence * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-zinc-500 block">Kelly Bet Size</span>
          <span className="font-semibold text-zinc-200">{(prediction.bet_size * 100).toFixed(1)}%</span>
        </div>
        <div className="text-right">
          <span className="text-zinc-500 block">Target Sizing</span>
          <span className={`font-semibold ${prediction.target_fraction > 0 ? "text-emerald-400" : prediction.target_fraction < 0 ? "text-rose-400" : "text-zinc-400"}`}>
            {(prediction.target_fraction * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
};
