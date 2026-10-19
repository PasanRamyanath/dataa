"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { AggTrade, BinanceTicker } from "@/types";
import { BinanceFuturesWebSocket, WsConnectionStatus } from "@/lib/binanceWs";

export function useBinanceWebSocket(symbol: string = "BTCUSDT") {
  const [ticker, setTicker] = useState<BinanceTicker | null>(null);
  const [recentTrades, setRecentTrades] = useState<AggTrade[]>([]);
  const [status, setStatus] = useState<WsConnectionStatus>("CONNECTING");
  const [error, setError] = useState<string | null>(null);
  const [priceHistory, setPriceHistory] = useState<number[]>([]);

  const wsClientRef = useRef<BinanceFuturesWebSocket | null>(null);
  const isWsConnectedRef = useRef(false);

  const lastMsgTimeRef = useRef<number>(0);

  // 1. Establish native WebSocket multiplex stream
  useEffect(() => {
    const client = new BinanceFuturesWebSocket(symbol);
    wsClientRef.current = client;

    client.setCallbacks({
      onTicker: (newTicker) => {
        lastMsgTimeRef.current = Date.now();
        isWsConnectedRef.current = true;
        setTicker(newTicker);
        setPriceHistory((prev) => {
          const next = [...prev, newTicker.price];
          if (next.length > 60) next.shift();
          return next;
        });
      },
      onTrade: (newTrade) => {
        lastMsgTimeRef.current = Date.now();
        isWsConnectedRef.current = true;
        setRecentTrades((prev) => {
          const next = [newTrade, ...prev];
          if (next.length > 40) next.pop();
          return next;
        });
      },
      onStatus: (newStatus, err) => {
        setStatus(newStatus);
        if (newStatus === "CONNECTED") {
          setError(null);
        } else {
          isWsConnectedRef.current = false;
          if (err) setError(err);
        }
      },
    });

    client.connect();

    return () => {
      client.disconnect();
    };
  }, [symbol]);

  // 2. Continuous Health Watchdog & HTTP Fallback Poller:
  // If native WebSocket has not delivered a message in > 1500ms, seamlessly poll local /api/binance-ticker
  useEffect(() => {
    let isMounted = true;

    const pollFallback = async () => {
      const msSinceLastWsMsg = Date.now() - lastMsgTimeRef.current;
      // Only fetch HTTP fallback if WebSocket has been quiet for > 1500ms
      if (msSinceLastWsMsg < 1500 && isWsConnectedRef.current) return;

      try {
        const res = await fetch(`/api/binance-ticker?symbol=${symbol}`);
        if (!res.ok) return;
        const json = await res.json();
        if (isMounted && json.success) {
          if (json.ticker) {
            setTicker(json.ticker);
            setPriceHistory((prev) => {
              const next = [...prev, json.ticker.price];
              if (next.length > 60) next.shift();
              return next;
            });
          }
          if (Array.isArray(json.trades) && json.trades.length > 0) {
            setRecentTrades(json.trades);
          }
        }
      } catch {
        // Silently handle fallback errors
      }
    };

    // Immediate initial fetch to prevent blank UI while WS connects
    pollFallback();

    const interval = setInterval(pollFallback, 1000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [symbol]);

  const reconnect = useCallback(() => {
    if (wsClientRef.current) {
      wsClientRef.current.disconnect();
      wsClientRef.current.connect();
    }
  }, []);

  return {
    ticker,
    recentTrades,
    status,
    error,
    priceHistory,
    reconnect,
  };
}
