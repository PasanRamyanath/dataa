"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { DashboardSettings, EnsemblePrediction } from "@/types";

export function useKaggleCloudStream(
  settings: DashboardSettings,
  currentPrice: number,
  onNewPrediction: (prediction: EnsemblePrediction, price: number) => void
) {
  const [latestPrediction, setLatestPrediction] = useState<EnsemblePrediction | null>(null);
  const [isPolling, setIsPolling] = useState(true);
  const [streamStatus, setStreamStatus] = useState<"ACTIVE" | "IDLE" | "ERROR">("ACTIVE");
  const [lastFetchTime, setLastFetchTime] = useState<number>(0);
  const [fetchLatency, setFetchLatency] = useState<number>(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const priceRef = useRef(currentPrice);
  priceRef.current = currentPrice;

  const onNewPredictionRef = useRef(onNewPrediction);
  onNewPredictionRef.current = onNewPrediction;
  const lastBarTsRef = useRef<string | number | null>(null);

  const fetchPrediction = useCallback(async () => {
    const t0 = performance.now();
    try {
      const price = priceRef.current || 95000;
      const params = new URLSearchParams({
        mode: settings.cloudMode,
        tunnelUrl: settings.kaggleTunnelUrl,
        secret: settings.streamSecret,
        price: price.toString(),
        metaThreshold: settings.metaThreshold.toString(),
      });

      const res = await fetch(`/api/predictions?${params.toString()}`);
      const latency = Math.round(performance.now() - t0);
      setFetchLatency(latency);

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const json = await res.json();
      if (json && json.data) {
        const pred: EnsemblePrediction = json.data;
        setLatestPrediction(pred);
        setStreamStatus("ACTIVE");
        setLastFetchTime(Date.now());
        setErrorMsg(json.error || null);

        // Deduplicate by source_bar_ts so fast 2.5s polling picks up every new 10s bar
        // within <1.5s of completion without ever double-scoring the same bar
        const barKey = pred.source_bar_ts ?? pred.timestamp;
        if (barKey && lastBarTsRef.current === barKey && settings.cloudMode === "KAGGLE_LIVE") {
          return;
        }
        if (barKey) {
          lastBarTsRef.current = barKey;
        }

        const anchorPrice = (typeof pred.source_close === "number" && pred.source_close > 0)
          ? pred.source_close
          : (priceRef.current || price);
        onNewPredictionRef.current(pred, anchorPrice);
      }
    } catch (e: any) {
      console.warn("[Prediction Stream] Fetch notice:", e?.message || e);
      setStreamStatus("ERROR");
      setErrorMsg(e?.message || "Failed to fetch prediction");
    }
  }, [settings.cloudMode, settings.kaggleTunnelUrl, settings.streamSecret, settings.metaThreshold]);

  // Fast 2.5s polling in KAGGLE_LIVE mode (with source_bar_ts deduplication) to eliminate polling phase lag
  useEffect(() => {
    if (!isPolling) {
      setStreamStatus("IDLE");
      return;
    }

    // Trigger initial tick immediately
    fetchPrediction();

    const intervalMs = settings.cloudMode === "KAGGLE_LIVE"
      ? 2500
      : Math.max(settings.pollIntervalSec * 1000, 3000);
    const timer = setInterval(() => {
      fetchPrediction();
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPolling, settings.cloudMode, settings.pollIntervalSec, fetchPrediction]);

  const togglePause = useCallback(() => {
    setIsPolling((prev) => !prev);
  }, []);

  const triggerManualTick = useCallback(() => {
    fetchPrediction();
  }, [fetchPrediction]);

  return {
    latestPrediction,
    isPolling,
    streamStatus,
    lastFetchTime,
    fetchLatency,
    errorMsg,
    togglePause,
    triggerManualTick,
  };
}
