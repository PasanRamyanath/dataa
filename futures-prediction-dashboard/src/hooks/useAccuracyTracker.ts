"use client";

import { useRef, useState, useCallback } from "react";
import { EnsemblePrediction, ModelAccuracyStats, ResolvedPrediction } from "@/types";
import { LiveAccuracyTracker } from "@/lib/accuracyTracker";

export function useAccuracyTracker() {
  const trackerRef = useRef<LiveAccuracyTracker | null>(null);

  if (!trackerRef.current) {
    trackerRef.current = new LiveAccuracyTracker();
  }

  const [stats, setStats] = useState<ModelAccuracyStats>(() => trackerRef.current!.getStats());
  const [latestResolution, setLatestResolution] = useState<ResolvedPrediction | null>(null);

  // Pipe new ensemble prediction into the tracker
  const recordPrediction = useCallback((prediction: EnsemblePrediction, currentPrice: number) => {
    if (!trackerRef.current || !currentPrice || currentPrice <= 0) return;
    const resolved = trackerRef.current.onNewPrediction(prediction, currentPrice);
    setStats(trackerRef.current.getStats());
    if (resolved && resolved.length > 0) {
      setLatestResolution(resolved[0]);
    }
  }, []);

  // Pipe continuous live WebSocket price ticks into the tracker
  const recordPriceTick = useCallback((currentPrice: number) => {
    if (!trackerRef.current || !currentPrice || currentPrice <= 0) return;
    const resolved = trackerRef.current.onLivePrice(currentPrice);
    if (resolved && resolved.length > 0) {
      setStats(trackerRef.current.getStats());
      setLatestResolution(resolved[0]);
    }
  }, []);

  // Reset tracker state
  const resetTracker = useCallback(() => {
    if (trackerRef.current) {
      trackerRef.current.reset();
      setStats(trackerRef.current.getStats());
      setLatestResolution(null);
    }
  }, []);

  return {
    stats,
    latestResolution,
    recordPrediction,
    recordPriceTick,
    resetTracker,
  };
}
