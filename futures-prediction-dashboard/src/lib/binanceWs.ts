import { AggTrade, BinanceTicker } from "@/types";

export type WsConnectionStatus = "CONNECTING" | "CONNECTED" | "DISCONNECTED" | "ERROR";

export class BinanceFuturesWebSocket {
  private ws: WebSocket | null = null;
  private symbol: string;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 20;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private lastMessageTime = 0;
  private isExplicitlyClosed = false;

  private onTickerCallback?: (ticker: BinanceTicker) => void;
  private onTradeCallback?: (trade: AggTrade) => void;
  private onStatusCallback?: (status: WsConnectionStatus, error?: string) => void;

  private tickerState: Partial<BinanceTicker> = {
    symbol: "BTCUSDT",
    price: 0,
    bidPrice: 0,
    askPrice: 0,
    bidQty: 0,
    askQty: 0,
    markPrice: 0,
    fundingRate: 0,
    nextFundingTime: 0,
    high24h: 0,
    low24h: 0,
    volume24h: 0,
    priceChange24h: 0,
    priceChangePercent24h: 0,
    lastUpdate: 0,
  };

  constructor(symbol: string = "BTCUSDT") {
    this.symbol = symbol.toLowerCase();
    // Standard combined multiplexed stream endpoint:
    this.url = `wss://fstream.binance.com/stream?streams=${this.symbol}@aggTrade/${this.symbol}@bookTicker/${this.symbol}@markPrice@1s/${this.symbol}@ticker`;
  }

  public setCallbacks(callbacks: {
    onTicker?: (ticker: BinanceTicker) => void;
    onTrade?: (trade: AggTrade) => void;
    onStatus?: (status: WsConnectionStatus, error?: string) => void;
  }) {
    this.onTickerCallback = callbacks.onTicker;
    this.onTradeCallback = callbacks.onTrade;
    this.onStatusCallback = callbacks.onStatus;
  }

  public connect() {
    if (typeof window === "undefined") return;
    this.isExplicitlyClosed = false;

    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.onStatusCallback?.("CONNECTING");

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        if (this.isExplicitlyClosed) {
          this.ws?.close();
          return;
        }
        this.reconnectAttempts = 0;
        this.lastMessageTime = Date.now();
        this.onStatusCallback?.("CONNECTED");
        this.startHeartbeatWatcher();
      };

      this.ws.onmessage = (event) => {
        if (this.isExplicitlyClosed) return;
        this.lastMessageTime = Date.now();
        this.handleMessage(event.data);
      };

      this.ws.onerror = (err) => {
        // In React Strict Mode, fast component unmounting triggers onerror on connecting websockets.
        // We suppress console.error if the connection was closed explicitly or unmounted.
        if (this.isExplicitlyClosed) return;
        console.warn("[Binance WS] Connection issue noticed, attempting reconnection...");
        this.onStatusCallback?.("ERROR", "WebSocket connection error");
      };

      this.ws.onclose = (event) => {
        this.stopHeartbeatWatcher();
        if (this.isExplicitlyClosed) return;
        this.onStatusCallback?.("DISCONNECTED");
        this.scheduleReconnect();
      };
    } catch (e: any) {
      if (this.isExplicitlyClosed) return;
      console.warn("[Binance WS] Initialization exception:", e?.message || e);
      this.onStatusCallback?.("ERROR", e?.message || "Failed to initialize WebSocket");
      this.scheduleReconnect();
    }
  }

  public disconnect() {
    this.isExplicitlyClosed = true;
    this.stopHeartbeatWatcher();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      // Detach event listeners before closing to prevent React StrictMode from logging onerror events
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onerror = null;
      this.ws.onclose = null;
      try {
        this.ws.close();
      } catch {
        // Silently catch browser close error
      }
      this.ws = null;
    }
    this.onStatusCallback?.("DISCONNECTED");
  }

  private handleMessage(raw: string) {
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || !parsed.data) return;

      const stream: string = parsed.stream || "";
      const data = parsed.data;

      // 1. Individual Symbol Book Ticker (@bookTicker)
      if (stream.includes("@bookTicker") || data.e === "bookTicker") {
        this.tickerState.bidPrice = parseFloat(data.b);
        this.tickerState.askPrice = parseFloat(data.a);
        this.tickerState.bidQty = parseFloat(data.B);
        this.tickerState.askQty = parseFloat(data.A);
        const mid = (this.tickerState.bidPrice + this.tickerState.askPrice) / 2;
        if (mid > 0) {
          this.tickerState.price = mid;
        }
        this.tickerState.lastUpdate = Date.now();
        this.emitTicker();
      }

      // 2. Aggregate Trade (@aggTrade)
      else if (stream.includes("@aggTrade") || data.e === "aggTrade") {
        const tradePrice = parseFloat(data.p);
        const tradeQty = parseFloat(data.q);
        if (tradePrice > 0) {
          this.tickerState.price = tradePrice;
        }
        this.tickerState.lastUpdate = Date.now();

        const trade: AggTrade = {
          tradeId: data.a,
          price: tradePrice,
          quantity: tradeQty,
          timestamp: data.T,
          isBuyerMaker: data.m,
        };
        this.onTradeCallback?.(trade);
        this.emitTicker();
      }

      // 3. Mark Price (@markPrice@1s)
      else if (stream.includes("@markPrice") || data.e === "markPriceUpdate") {
        this.tickerState.markPrice = parseFloat(data.p);
        this.tickerState.fundingRate = parseFloat(data.r);
        this.tickerState.nextFundingTime = data.T;
        if (!this.tickerState.price || this.tickerState.price === 0) {
          this.tickerState.price = this.tickerState.markPrice;
        }
        this.tickerState.lastUpdate = Date.now();
        this.emitTicker();
      }

      // 4. 24hr Mini Ticker (@ticker)
      else if (stream.includes("@ticker") || data.e === "24hrTicker") {
        this.tickerState.high24h = parseFloat(data.h);
        this.tickerState.low24h = parseFloat(data.l);
        this.tickerState.volume24h = parseFloat(data.v);
        this.tickerState.priceChange24h = parseFloat(data.p);
        this.tickerState.priceChangePercent24h = parseFloat(data.P);
        if (data.c) {
          const cPrice = parseFloat(data.c);
          if (cPrice > 0) {
            this.tickerState.price = cPrice;
          }
        }
        this.tickerState.lastUpdate = Date.now();
        this.emitTicker();
      }
    } catch {
      // Ignored non-fatal parse errors
    }
  }

  private emitTicker() {
    if (this.onTickerCallback && this.tickerState.price && this.tickerState.price > 0) {
      this.onTickerCallback(this.tickerState as BinanceTicker);
    }
  }

  private scheduleReconnect() {
    if (this.isExplicitlyClosed) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.onStatusCallback?.("ERROR", "Max reconnect attempts reached");
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 10000);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startHeartbeatWatcher() {
    this.stopHeartbeatWatcher();
    this.heartbeatTimer = setInterval(() => {
      if (Date.now() - this.lastMessageTime > 15000) {
        if (this.ws && !this.isExplicitlyClosed) {
          try {
            this.ws.close();
          } catch {
            // Ignore
          }
        }
      }
    }, 5000);
  }

  private stopHeartbeatWatcher() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }
}
