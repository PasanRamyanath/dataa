export type PositionSide = "LONG" | "SHORT" | "FLAT";

export type MarginMode = "ISOLATED" | "CROSS";

export type ExecutionRole = "MAKER" | "TAKER";

export type ExecutionStyle = "LIMIT_MAKER" | "MARKET_TAKER" | "SMART_HYBRID";

export type VipTier =
  | "VIP0"
  | "VIP1"
  | "VIP2"
  | "VIP3"
  | "VIP4"
  | "VIP5"
  | "VIP7"
  | "VIP9_MM";

export type TradeReason =
  | "SIGNAL_ENTRY"
  | "VETO_EXIT"
  | "TAKE_PROFIT"
  | "STOP_LOSS"
  | "TRAILING_STOP"
  | "FUNDING_FEE"
  | "MAX_HOLD_EXPIRED"
  | "DEADBAND_REBALANCE"
  | "MANUAL_CLOSE"
  | "LIQUIDATION";

export interface BinanceLeverageBracket {
  tier: number;
  maxNotional: number; // in USDT
  maxLeverage: number; // e.g. 125, 100, 50, 20...
  mmr: number; // Maintenance Margin Rate, e.g. 0.004 = 0.4%
  cumMaintenanceAmount: number; // Maintenance Amount in USDT
}

/**
 * Official Binance USDⓈ-M Futures BTCUSDT Tiered Margin & Leverage Bracket Table
 */
export const BINANCE_BTCUSDT_BRACKETS: BinanceLeverageBracket[] = [
  { tier: 1, maxNotional: 50000, maxLeverage: 125, mmr: 0.0040, cumMaintenanceAmount: 0 },
  { tier: 2, maxNotional: 250000, maxLeverage: 100, mmr: 0.0050, cumMaintenanceAmount: 50 },
  { tier: 3, maxNotional: 1000000, maxLeverage: 50, mmr: 0.0100, cumMaintenanceAmount: 1300 },
  { tier: 4, maxNotional: 5000000, maxLeverage: 20, mmr: 0.0250, cumMaintenanceAmount: 16300 },
  { tier: 5, maxNotional: 20000000, maxLeverage: 10, mmr: 0.0500, cumMaintenanceAmount: 141300 },
  { tier: 6, maxNotional: 50000000, maxLeverage: 5, mmr: 0.1000, cumMaintenanceAmount: 1141300 },
  { tier: 7, maxNotional: 100000000, maxLeverage: 4, mmr: 0.1250, cumMaintenanceAmount: 2391300 },
  { tier: 8, maxNotional: 200000000, maxLeverage: 3, mmr: 0.1500, cumMaintenanceAmount: 4891300 },
  { tier: 9, maxNotional: 300000000, maxLeverage: 2, mmr: 0.2500, cumMaintenanceAmount: 24891300 },
  { tier: 10, maxNotional: Infinity, maxLeverage: 1, mmr: 0.5000, cumMaintenanceAmount: 99891300 },
];

export function getBtcLeverageBracket(notional: number): BinanceLeverageBracket {
  const absNotional = Math.abs(notional);
  for (const b of BINANCE_BTCUSDT_BRACKETS) {
    if (absNotional <= b.maxNotional) return b;
  }
  return BINANCE_BTCUSDT_BRACKETS[BINANCE_BTCUSDT_BRACKETS.length - 1];
}

export function getMaxNotionalForLeverage(leverage: number): number {
  let maxCap = 50000;
  for (const b of BINANCE_BTCUSDT_BRACKETS) {
    if (leverage <= b.maxLeverage) {
      maxCap = b.maxNotional;
    }
  }
  return maxCap;
}

/**
 * Official Binance USDⓈ-M Futures VIP Fee Schedules (basis points: 1 bps = 0.01% = 0.0001)
 */
export const BINANCE_FUTURES_VIP_FEES: Record<
  VipTier,
  { makerBps: number; takerBps: number; name: string }
> = {
  VIP0: { makerBps: 2.0, takerBps: 5.0, name: "VIP 0 (Regular)" },
  VIP1: { makerBps: 1.6, takerBps: 4.0, name: "VIP 1" },
  VIP2: { makerBps: 1.4, takerBps: 3.5, name: "VIP 2" },
  VIP3: { makerBps: 1.2, takerBps: 3.2, name: "VIP 3" },
  VIP4: { makerBps: 1.0, takerBps: 3.0, name: "VIP 4" },
  VIP5: { makerBps: 0.8, takerBps: 2.7, name: "VIP 5 (HFT Pro)" },
  VIP7: { makerBps: 0.4, takerBps: 2.3, name: "VIP 7 (Institutional)" },
  VIP9_MM: { makerBps: 0.0, takerBps: 1.7, name: "VIP 9 / Market Maker (0 bps Maker)" },
};

// Binance Contract Specifications for BTCUSDT Perpetual
export const CONTRACT_STEP_BTC = 0.001; // Lot size: 0.001 BTC
export const PRICE_TICK_USDT = 0.10; // Tick size: 0.10 USDT
export const MIN_NOTIONAL_USDT = 5.0; // Minimum notional: 5 USDT
export const LIQUIDATION_FEE_RATE = 0.0125; // 1.25% Insurance Fund Clearance Fee

export interface BinanceTicker {
  symbol: string;
  price: number;
  bidPrice: number;
  askPrice: number;
  bidQty: number;
  askQty: number;
  markPrice: number;
  fundingRate: number;
  nextFundingTime: number;
  high24h: number;
  low24h: number;
  volume24h: number;
  priceChange24h: number;
  priceChangePercent24h: number;
  lastUpdate: number;
}

export interface AggTrade {
  tradeId: number;
  price: number;
  quantity: number;
  timestamp: number;
  isBuyerMaker: boolean; // true = sell taker, false = buy taker
}

export interface MemberPrediction {
  name: string;
  prob_down: number;
  prob_flat: number;
  prob_up: number;
  primary_direction: -1 | 0 | 1;
  predicted_return: number;
  latency_ms: number;
  weight: number;
}

export interface EnsemblePrediction {
  timestamp: string;
  source_bar_ts?: string | number;
  source_close?: number;
  source_open?: number;
  bar_index?: number;
  symbol: string;
  members: Record<string, MemberPrediction>;
  regime_state: 0 | 1 | 2; // 0: Low Vol Mean-Rev, 1: Momentum Trend, 2: High Vol Liquidation
  regime_name: string;
  regime_multiplier: number;
  raw_proba: [number, number, number]; // [down, flat, up]
  calibrated_proba: [number, number, number]; // Dirichlet calibrated
  primary_direction: -1 | 0 | 1;
  confidence: number;
  act_probability: number; // Meta-labeling filter output
  bet_size: number;
  is_trade_allowed: boolean; // act_probability >= tau
  meta_threshold: number;
  target_fraction: number; // Kelly sized target position (-0.25 to +0.25)
  fusion_stage: string;
  latency_seconds: number;
}

export interface PaperTrade {
  id: string;
  timestamp: string;
  side: "BUY" | "SELL";
  action: "OPEN" | "CLOSE" | "REBALANCE" | "FUNDING";
  position_side: PositionSide;
  price: number;
  size: number; // contracts in BTC
  notional: number; // USD
  fee: number; // USD
  slippage: number; // USD
  role?: ExecutionRole; // MAKER or TAKER
  margin_mode?: MarginMode;
  leverage?: number;
  funding_amount?: number;
  pnl?: number; // Net Realized PnL in USD
  pnl_pct?: number; // % Return on Position Margin (ROE%)
  gross_pnl?: number; // Gross PnL before fees
  reason: TradeReason;
  equity_after: number;
  confidence: number;
  act_probability: number;
  meta_threshold: number;
  executionTarget?: "LOCAL_SIMULATOR" | "BINANCE_TESTNET_LIVE";
  orderId?: string | number;
}

export interface PaperAccount {
  starting_balance: number;
  equity: number; // Total Margin Balance (Wallet Balance + Unrealized PnL)
  wallet_balance: number; // Cash + Realized PnL - Fees +/- Funding
  margin_balance: number; // Wallet Balance + Unrealized PnL
  cash: number; // Available Balance for new orders
  leverage: number; // Target / configured isolated leverage
  margin_mode: MarginMode; // ISOLATED or CROSS
  tier: number; // Binance Bracket Tier (1 to 10)
  max_notional_tier: number; // Max position notional for current tier
  cum_maintenance_amount: number; // Maintenance cum amount
  position_side: PositionSide;
  position_size: number; // in BTC contracts (0.001 step)
  position_notional: number; // in USD
  entry_price: number | null;
  entry_timestamp: string | null;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number; // ROE%
  roe_pct: number;
  initial_margin: number; // Margin allocated to open position
  isolated_margin: number; // Dedicated isolated margin
  maintenance_margin: number; // Minimum margin required by Binance Tier MMR
  margin_ratio_pct: number; // Maintenance Margin / Margin Balance * 100%
  liquidation_price: number | null; // Exact Binance liquidation price
  tp_price: number | null; // Take-profit trigger price
  sl_price: number | null; // Stop-loss trigger price
  trailing_stop_active: boolean;
  trailing_stop_peak_price: number | null;
  trailing_stop_price: number | null;
  last_funding_time: number;
  total_funding_fees: number;
  realized_pnl: number; // Net Realized PnL across all closed trades
  total_fees: number;
  total_slippage: number;
  win_count: number;
  loss_count: number;
  total_trades: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  peak_equity: number;
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
  btc_price: number;
  drawdown_pct: number;
}

export interface DashboardSettings {
  cloudMode: "SIMULATOR" | "KAGGLE_LIVE" | "HISTORICAL_REPLAY";
  kaggleTunnelUrl: string;
  streamSecret: string;
  metaThreshold: number; // default 0.55
  leverage: number; // default 100
  marginMode: MarginMode; // ISOLATED or CROSS
  executionStyle: ExecutionStyle; // LIMIT_MAKER, MARKET_TAKER, SMART_HYBRID
  vipTier: VipTier; // VIP0 to VIP9_MM
  useBnbFeeDiscount: boolean; // 10% discount on futures fees
  trailingStopCallbackPct: number; // default 0.05%
  tpPct: number; // default +0.14%
  slPct: number; // default -0.12%
  pollIntervalSec: number; // default 10
  startingCapital: number; // default 10,000
  takerFeeBps: number; // custom or derived from VIP tier
  slippageBps: number; // default 0.1
  deadbandPct: number; // default 0.01 (1%)
  soundAlerts: boolean;
  executionTarget: "LOCAL_SIMULATOR" | "BINANCE_TESTNET_LIVE";
}

export interface ResolvedPrediction {
  id: string;
  timestamp: string;
  resolvedTimestamp: string;
  horizonSec: number;
  entryPrice: number;
  resolvedPrice: number;
  priceChangePct: number;
  predictedDirection: -1 | 0 | 1;
  actualDirection: -1 | 0 | 1;
  isCorrect: boolean;
  confidence: number;
  actProbability: number;
  isAuthorized: boolean;
  memberVotes?: Record<string, -1 | 0 | 1>;
}

export interface ModelAccuracyStats {
  totalEvaluated: number;
  correctPredictions: number;
  overallAccuracy: number; // %
  upEvaluated: number;
  upCorrect: number;
  upAccuracy: number; // %
  downEvaluated: number;
  downCorrect: number;
  downAccuracy: number; // %
  authorizedEvaluated: number;
  authorizedCorrect: number;
  authorizedAccuracy: number; // %
  vetoedEvaluated: number;
  vetoedCorrect: number;
  vetoedAccuracy: number; // %
  avgPriceReturnPct: number;
  horizon10sAccuracy: number;
  horizon30sAccuracy: number;
  horizon60sAccuracy: number;
  recentResolved: ResolvedPrediction[];
  memberAccuracies: Record<string, { total: number; correct: number; accuracy: number }>;
  activePending?: {
    id: string;
    timestamp: string;
    createdAtMs: number;
    entryPrice: number;
    predictedDirection: -1 | 0 | 1;
    confidence: number;
    isAuthorized: boolean;
  } | null;
}
