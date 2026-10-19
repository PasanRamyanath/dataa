"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  DashboardSettings,
  EnsemblePrediction,
  EquityPoint,
  ExecutionRole,
  ExecutionStyle,
  MarginMode,
  PaperAccount,
  PaperTrade,
  PositionSide,
} from "@/types";
import { ClientPaperTradingEngine } from "@/lib/paperEngine";

export interface TestnetStatus {
  connected: boolean;
  lastSync: number;
  lastOrderId: string | number | null;
  lastMessage: string | null;
  error: string | null;
}

export function usePaperTrading(settings: DashboardSettings) {
  const engineRef = useRef<ClientPaperTradingEngine | null>(null);

  if (!engineRef.current) {
    engineRef.current = new ClientPaperTradingEngine(settings);
  }

  const [account, setAccount] = useState<PaperAccount>(() => engineRef.current!.getAccount());
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [lastExecutedTrade, setLastExecutedTrade] = useState<PaperTrade | null>(null);
  const [testnetStatus, setTestnetStatus] = useState<TestnetStatus>({
    connected: false,
    lastSync: 0,
    lastOrderId: null,
    lastMessage: null,
    error: null,
  });

  // Settings ref to access fresh settings in callbacks without stale closures
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  // Synchronize settings to paper engine when changed
  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.updateSettings(settings);
      if (settings.executionTarget === "LOCAL_SIMULATOR") {
        setAccount(engineRef.current.getAccount());
      }
    }
  }, [settings]);

  // Synchronize account state from Binance Futures Testnet
  const syncTestnetAccount = useCallback(async () => {
    if (settingsRef.current.executionTarget !== "BINANCE_TESTNET_LIVE") return;

    try {
      const res = await fetch("/api/binance-testnet?action=account", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`Testnet HTTP ${res.status}`);
      }
      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || "Failed to fetch testnet account");
      }

      setTestnetStatus((prev) => ({
        ...prev,
        connected: true,
        lastSync: Date.now(),
        error: null,
      }));

      setAccount((prev) => {
        const walletBal = typeof data.totalWalletBalance === "number" ? data.totalWalletBalance : prev.wallet_balance;
        const marginBal = typeof data.totalMarginBalance === "number" ? data.totalMarginBalance : prev.margin_balance;
        const availBal = typeof data.availableBalance === "number" ? data.availableBalance : prev.cash;
        const unPnl = typeof data.totalUnrealizedProfit === "number" ? data.totalUnrealizedProfit : 0;
        const pos = data.position;

        const hasPosition = pos && Math.abs(pos.positionAmt) > 0.0001;
        const posSide: PositionSide = hasPosition ? (pos.positionAmt > 0 ? "LONG" : "SHORT") : "FLAT";
        const posSize = hasPosition ? Math.abs(pos.positionAmt) : 0;
        const entryPrice = hasPosition ? pos.entryPrice : null;
        const currentPrice = pos && pos.markPrice > 0 ? pos.markPrice : prev.current_price;
        const posNotional = hasPosition ? parseFloat((posSize * currentPrice).toFixed(2)) : 0;
        const posLev = pos?.leverage || prev.leverage;
        const liqPrice = hasPosition && pos.liquidationPrice > 0 ? pos.liquidationPrice : null;
        const posMarginType: MarginMode = pos?.marginType?.toUpperCase() === "ISOLATED" ? "ISOLATED" : "CROSS";

        const initialMargin = (posNotional > 0 && posLev > 0) ? parseFloat((posNotional / posLev).toFixed(2)) : 0;
        const roePct = initialMargin > 0 ? (unPnl / initialMargin) * 100 : 0;

        return {
          ...prev,
          starting_balance: prev.starting_balance === 10000 && walletBal === 5000 ? 5000 : prev.starting_balance,
          wallet_balance: walletBal,
          equity: marginBal,
          margin_balance: marginBal,
          cash: availBal,
          current_price: currentPrice,
          unrealized_pnl: unPnl,
          unrealized_pnl_pct: roePct,
          roe_pct: roePct,
          position_side: posSide,
          position_size: posSize,
          position_notional: posNotional,
          entry_price: entryPrice,
          leverage: posLev,
          margin_mode: posMarginType,
          liquidation_price: liqPrice,
          initial_margin: initialMargin,
          isolated_margin: posMarginType === "ISOLATED" ? initialMargin : 0,
        };
      });
    } catch (e: any) {
      console.warn("[Binance Testnet Sync] Notice:", e?.message || e);
      setTestnetStatus((prev) => ({
        ...prev,
        connected: false,
        error: e?.message || "Testnet sync failed",
      }));
    }
  }, []);

  // Periodic account sync when BINANCE_TESTNET_LIVE is active
  useEffect(() => {
    if (settings.executionTarget !== "BINANCE_TESTNET_LIVE") return;

    syncTestnetAccount();
    const interval = setInterval(syncTestnetAccount, 4000);
    return () => clearInterval(interval);
  }, [settings.executionTarget, syncTestnetAccount]);

  // Place order on Binance Futures Testnet
  const placeTestnetOrder = useCallback(
    async (
      side: "BUY" | "SELL",
      sizeBtc: number,
      price: number,
      style: ExecutionStyle = "SMART_HYBRID"
    ) => {
      // Step: 0.001 BTC, minimum testnet order: 0.002 BTC
      const qty = Math.max(0.002, Math.floor(sizeBtc * 1000) / 1000);
      const orderPrice = Math.round(price * 10) / 10;

      const tryMaker = style !== "MARKET_TAKER";

      if (tryMaker) {
        try {
          const res = await fetch("/api/binance-testnet", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              action: "ORDER",
              symbol: "BTCUSDT",
              side,
              type: "LIMIT",
              quantity: qty.toFixed(3),
              price: orderPrice.toFixed(1),
              timeInForce: "GTX", // Post-Only Maker
            }),
          });
          const data = await res.json();

          if (data.success && data.order) {
            setTestnetStatus((prev) => ({
              ...prev,
              lastOrderId: data.order.orderId,
              lastMessage: `Maker GTX Limit Order #${data.order.orderId} Placed (${side} ${qty} BTC @ $${orderPrice})`,
            }));
            return { success: true, order: data.order, role: "MAKER" as ExecutionRole };
          }

          // If GTX rejected with -5022 (would match taker) and SMART_HYBRID is active: fallback to MARKET!
          if (data.code === -5022 && style === "SMART_HYBRID") {
            console.log("[Testnet SMART_HYBRID] Post-Only GTX crossed book; routing as MARKET taker...");
            const mRes = await fetch("/api/binance-testnet", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                action: "ORDER",
                symbol: "BTCUSDT",
                side,
                type: "MARKET",
                quantity: qty.toFixed(3),
              }),
            });
            const mData = await mRes.json();
            if (mData.success && mData.order) {
              setTestnetStatus((prev) => ({
                ...prev,
                lastOrderId: mData.order.orderId,
                lastMessage: `Hybrid Market Order #${mData.order.orderId} Filled (${side} ${qty} BTC)`,
              }));
              return { success: true, order: mData.order, role: "TAKER" as ExecutionRole };
            }
          }

          console.warn("[Testnet Order] Order placement failed:", data.error);
          setTestnetStatus((prev) => ({ ...prev, error: data.error }));
          return { success: false, error: data.error };
        } catch (e: any) {
          console.warn("[Testnet Order] Error:", e);
          setTestnetStatus((prev) => ({ ...prev, error: e?.message || String(e) }));
          return { success: false, error: e?.message || String(e) };
        }
      } else {
        // Direct MARKET Taker execution
        try {
          const res = await fetch("/api/binance-testnet", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              action: "ORDER",
              symbol: "BTCUSDT",
              side,
              type: "MARKET",
              quantity: qty.toFixed(3),
            }),
          });
          const data = await res.json();
          if (data.success && data.order) {
            setTestnetStatus((prev) => ({
              ...prev,
              lastOrderId: data.order.orderId,
              lastMessage: `Market Order #${data.order.orderId} Filled (${side} ${qty} BTC)`,
            }));
            return { success: true, order: data.order, role: "TAKER" as ExecutionRole };
          }
          setTestnetStatus((prev) => ({ ...prev, error: data.error }));
          return { success: false, error: data.error };
        } catch (e: any) {
          setTestnetStatus((prev) => ({ ...prev, error: e?.message || String(e) }));
          return { success: false, error: e?.message || String(e) };
        }
      }
    },
    []
  );

  // Close position on Binance Futures Testnet
  const closeTestnetPosition = useCallback(async () => {
    try {
      const res = await fetch("/api/binance-testnet", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "CLOSE_POSITION",
          symbol: "BTCUSDT",
        }),
      });
      const data = await res.json();
      setTestnetStatus((prev) => ({
        ...prev,
        lastMessage: data.success ? "Position flattened on Binance Testnet" : data.error,
      }));
      await syncTestnetAccount();
      return data;
    } catch (e: any) {
      console.warn("[Testnet Close] Error closing position:", e);
      setTestnetStatus((prev) => ({ ...prev, error: e?.message || "Failed to close testnet position" }));
    }
  }, [syncTestnetAccount]);

  // Update live price for continuous unrealized PnL ticking and instant intra-tick TP/SL execution
  const updatePrice = useCallback(
    (
      price: number,
      bookTicker?: { bidPrice: number; askPrice: number; bidQty?: number; askQty?: number },
      fundingInfo?: { fundingRate?: number; nextFundingTime?: number }
    ) => {
      if (!engineRef.current || !price || price <= 0) return;
      const tickTrade = engineRef.current.onLivePrice(price, bookTicker, fundingInfo);

      if (settingsRef.current.executionTarget === "LOCAL_SIMULATOR") {
        setAccount(engineRef.current.getAccount());
      } else {
        // In Testnet mode, update current_price and intra-tick unrealized PnL
        setAccount((prev) => {
          if (prev.position_side === "FLAT" || !prev.entry_price || prev.position_size === 0) {
            return { ...prev, current_price: price };
          }
          const isLong = prev.position_side === "LONG";
          const deltaPrice = isLong ? price - prev.entry_price : prev.entry_price - price;
          const unPnl = parseFloat((deltaPrice * prev.position_size).toFixed(2));
          const marginBal = parseFloat((prev.wallet_balance + unPnl).toFixed(2));
          const initialMargin = prev.initial_margin > 0 ? prev.initial_margin : (prev.position_notional / Math.max(1, prev.leverage));
          const roe = initialMargin > 0 ? (unPnl / initialMargin) * 100 : 0;
          return {
            ...prev,
            current_price: price,
            unrealized_pnl: unPnl,
            margin_balance: marginBal,
            equity: marginBal,
            unrealized_pnl_pct: roe,
            roe_pct: roe,
          };
        });
      }

      if (tickTrade) {
        tickTrade.executionTarget = settingsRef.current.executionTarget;
        setTrades(engineRef.current.getTrades());
        setEquityHistory(engineRef.current.getEquityHistory());
        setLastExecutedTrade(tickTrade);

        // If intra-tick TP/SL triggered a CLOSE in Testnet mode: execute on Binance Testnet
        if (settingsRef.current.executionTarget === "BINANCE_TESTNET_LIVE" && tickTrade.action === "CLOSE") {
          closeTestnetPosition();
        }
      }
    },
    [closeTestnetPosition]
  );

  // Process a new 10-second bar ensemble prediction
  const processPrediction = useCallback(
    async (
      prediction: EnsemblePrediction,
      currentPrice: number,
      bookTicker?: { bidPrice: number; askPrice: number }
    ) => {
      if (!engineRef.current || !currentPrice || currentPrice <= 0) return;
      const trade = engineRef.current.processBar(prediction, currentPrice, bookTicker);

      if (settingsRef.current.executionTarget === "LOCAL_SIMULATOR") {
        setAccount(engineRef.current.getAccount());
      }
      setTrades(engineRef.current.getTrades());
      setEquityHistory(engineRef.current.getEquityHistory());

      if (trade) {
        trade.executionTarget = settingsRef.current.executionTarget;
        setLastExecutedTrade(trade);

        // Route live order to Binance Futures Testnet if active
        if (settingsRef.current.executionTarget === "BINANCE_TESTNET_LIVE") {
          if (trade.action === "OPEN") {
            const orderRes = await placeTestnetOrder(
              trade.side,
              trade.size,
              trade.price,
              settingsRef.current.executionStyle
            );
            if (orderRes?.success && orderRes.order?.orderId) {
              trade.orderId = orderRes.order.orderId;
              trade.role = orderRes.role;
            }
            await syncTestnetAccount();
          } else if (trade.action === "CLOSE") {
            await closeTestnetPosition();
          }
        }
      }
    },
    [placeTestnetOrder, closeTestnetPosition, syncTestnetAccount]
  );

  const manualClose = useCallback(
    async (price: number, role: ExecutionRole = "TAKER") => {
      if (!engineRef.current || !price || price <= 0) return;
      const trade = engineRef.current.manualClosePosition(price, role);

      if (settingsRef.current.executionTarget === "BINANCE_TESTNET_LIVE") {
        await closeTestnetPosition();
      } else {
        setAccount(engineRef.current.getAccount());
      }

      if (trade) {
        trade.executionTarget = settingsRef.current.executionTarget;
        setTrades(engineRef.current.getTrades());
        setEquityHistory(engineRef.current.getEquityHistory());
        setLastExecutedTrade(trade);
      }
    },
    [closeTestnetPosition]
  );

  const setLeverage = useCallback(
    async (leverage: number) => {
      if (!engineRef.current) return;
      engineRef.current.setLeverage(leverage);

      if (settingsRef.current.executionTarget === "BINANCE_TESTNET_LIVE") {
        try {
          await fetch("/api/binance-testnet", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "SET_LEVERAGE", symbol: "BTCUSDT", leverage }),
          });
          await syncTestnetAccount();
        } catch (e) {
          console.warn("[Testnet] Set leverage failed:", e);
        }
      } else {
        setAccount(engineRef.current.getAccount());
      }
    },
    [syncTestnetAccount]
  );

  const setMarginMode = useCallback((mode: MarginMode) => {
    if (!engineRef.current) return;
    engineRef.current.setMarginMode(mode);
    setAccount(engineRef.current.getAccount());
  }, []);

  const resetAccount = useCallback(
    async (capital?: number) => {
      if (engineRef.current) {
        engineRef.current.reset(capital);
        if (settingsRef.current.executionTarget === "BINANCE_TESTNET_LIVE") {
          await closeTestnetPosition();
          await syncTestnetAccount();
        } else {
          setAccount(engineRef.current.getAccount());
        }
        setTrades([]);
        setEquityHistory([]);
        setLastExecutedTrade(null);
      }
    },
    [closeTestnetPosition, syncTestnetAccount]
  );

  return {
    account,
    trades,
    equityHistory,
    lastExecutedTrade,
    testnetStatus,
    syncTestnetAccount,
    updatePrice,
    processPrediction,
    manualClose,
    setLeverage,
    setMarginMode,
    resetAccount,
  };
}
