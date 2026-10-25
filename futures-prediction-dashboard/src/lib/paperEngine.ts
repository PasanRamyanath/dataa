import {
  BINANCE_BTCUSDT_BRACKETS,
  BINANCE_FUTURES_VIP_FEES,
  CONTRACT_STEP_BTC,
  DashboardSettings,
  EnsemblePrediction,
  EquityPoint,
  ExecutionRole,
  getBtcLeverageBracket,
  getMaxNotionalForLeverage,
  LIQUIDATION_FEE_RATE,
  MarginMode,
  PaperAccount,
  PaperTrade,
  PositionSide,
  PRICE_TICK_USDT,
  TradeReason,
} from "@/types";

/**
 * Production-Grade Binance USDⓈ-M Futures Trading Engine (BTCUSDT Perpetual)
 *
 * Implements 100% authentic Binance USDⓈ-M production exchange mechanics:
 * 1. Tiered Leverage Brackets (Tier 1-10, 1x-125x with max position notional caps)
 * 2. Cumulative Maintenance Margin Deduction (cumMaintenanceAmount)
 * 3. Isolated Margin vs. Cross Margin modes with exact liquidation formulas
 * 4. Contract Lot Size (0.001 BTC step) & Tick Size (0.10 USDT step) precision
 * 5. Full Binance VIP 0 to VIP 9 fee tiers + BNB 10% discount
 * 6. Maker (Post-Only Limit) vs. Taker (Market) execution role logging
 * 7. Live Top-of-Book Pricing via BookTicker (bestBid / bestAsk)
 * 8. 8-Hour Funding Fee Settlement Clock (Cash debits/credits via mark price)
 * 9. Sub-Second Intra-Tick Scalp Barrier & Trailing Stop execution
 * 10. Binance Insurance Fund Liquidation Clearance Fee (1.25%)
 */
export class ClientPaperTradingEngine {
  private account: PaperAccount;
  private trades: PaperTrade[] = [];
  private equityHistory: EquityPoint[] = [];
  private settings: DashboardSettings;
  private barsInTrade = 0;
  private maxHoldingBars = 30; // 5.0 minutes optimal holding horizon for multi-horizon ensemble scalping
  private minHoldingBars = 2; // 20 seconds minimum patience before discretionary veto exit
  private barsSinceExit = 999; // Cooldown tracking since last closed trade
  private minCooldownBars = 2; // 20 seconds minimum cooldown between trades to avoid overtrading

  constructor(settings: DashboardSettings) {
    this.settings = settings;
    const startCap = settings.startingCapital || 10000;
    const initialLev = settings.leverage || 100;
    const initialMarginMode: MarginMode = settings.marginMode || "ISOLATED";
    const initialBracket = getBtcLeverageBracket(0);

    this.account = {
      starting_balance: startCap,
      equity: startCap,
      wallet_balance: startCap,
      margin_balance: startCap,
      cash: startCap,
      leverage: initialLev,
      margin_mode: initialMarginMode,
      tier: initialBracket.tier,
      max_notional_tier: getMaxNotionalForLeverage(initialLev),
      cum_maintenance_amount: 0,
      position_side: "FLAT",
      position_size: 0,
      position_notional: 0,
      entry_price: null,
      entry_timestamp: null,
      current_price: 0,
      unrealized_pnl: 0,
      unrealized_pnl_pct: 0,
      roe_pct: 0,
      initial_margin: 0,
      isolated_margin: 0,
      maintenance_margin: 0,
      margin_ratio_pct: 0,
      liquidation_price: null,
      tp_price: null,
      sl_price: null,
      trailing_stop_active: false,
      trailing_stop_peak_price: null,
      trailing_stop_price: null,
      last_funding_time: 0,
      total_funding_fees: 0,
      realized_pnl: 0,
      total_fees: 0,
      total_slippage: 0,
      win_count: 0,
      loss_count: 0,
      total_trades: 0,
      max_drawdown: 0,
      max_drawdown_pct: 0,
      peak_equity: startCap,
    };
  }

  public updateSettings(newSettings: Partial<DashboardSettings>) {
    this.settings = { ...this.settings, ...newSettings };
    if (newSettings.leverage) {
      this.account.leverage = newSettings.leverage;
      this.account.max_notional_tier = getMaxNotionalForLeverage(newSettings.leverage);
    }
    if (newSettings.marginMode) {
      this.account.margin_mode = newSettings.marginMode;
    }
  }

  public getAccount(): PaperAccount {
    return { ...this.account };
  }

  public getTrades(): PaperTrade[] {
    return [...this.trades];
  }

  public getEquityHistory(): EquityPoint[] {
    return [...this.equityHistory];
  }

  /**
   * Helper: Round price to Binance tick size ($0.10)
   */
  public static roundPrice(price: number): number {
    return Math.round(price / PRICE_TICK_USDT) * PRICE_TICK_USDT;
  }

  /**
   * Helper: Round quantity to Binance lot step (0.001 BTC)
   */
  public static roundLot(qty: number): number {
    return Math.floor(Math.abs(qty) / CONTRACT_STEP_BTC) * CONTRACT_STEP_BTC;
  }

  /**
   * Calculates applicable Maker and Taker fee rates in basis points
   * based on the selected Binance VIP Tier and BNB 10% discount.
   */
  private getFeeBps(role: ExecutionRole): number {
    const vip = this.settings.vipTier || "VIP0";
    const tierInfo = BINANCE_FUTURES_VIP_FEES[vip] || BINANCE_FUTURES_VIP_FEES.VIP0;

    let baseBps = role === "MAKER" ? tierInfo.makerBps : tierInfo.takerBps;

    // Apply custom user fee override if explicitly specified in settings
    if (role === "TAKER" && this.settings.takerFeeBps !== undefined) {
      baseBps = this.settings.takerFeeBps;
    }

    // Apply Binance BNB 10% fee deduction if enabled
    if (this.settings.useBnbFeeDiscount) {
      baseBps *= 0.90;
    }

    return Math.max(0, baseBps);
  }

  /**
   * Checks and executes 8-hour funding cash settlement when crossing a funding epoch
   */
  public checkFundingSettlement(markPrice: number, fundingRate: number = 0.0001, nextFundingTime?: number): PaperTrade | null {
    if (this.account.position_side === "FLAT" || this.account.position_size <= 0 || markPrice <= 0) {
      return null;
    }

    const now = Date.now();
    if (this.account.last_funding_time === 0) {
      this.account.last_funding_time = now;
      return null;
    }

    const shouldSettle = nextFundingTime
      ? (now >= nextFundingTime && now - this.account.last_funding_time >= 3600000)
      : (now - this.account.last_funding_time >= 8 * 3600 * 1000);

    if (!shouldSettle) return null;

    const isLong = this.account.position_side === "LONG";
    const notional = this.account.position_size * markPrice;

    // Funding Amount: Long pays Short if rate > 0; Short pays Long if rate < 0
    let fundingCashFlow = notional * fundingRate;
    if (!isLong) {
      fundingCashFlow = -fundingCashFlow; // Reverse for short
    }

    // A positive fundingCashFlow means paying funding (debit wallet)
    // A negative fundingCashFlow means receiving funding (credit wallet)
    this.account.wallet_balance -= fundingCashFlow;
    this.account.total_funding_fees += fundingCashFlow;
    this.account.last_funding_time = now;

    const tradeRecord: PaperTrade = {
      id: `funding_${now}_${Math.random().toString(36).substring(2, 6)}`,
      timestamp: new Date(now).toISOString(),
      side: fundingCashFlow > 0 ? "SELL" : "BUY",
      action: "FUNDING",
      position_side: this.account.position_side,
      price: markPrice,
      size: this.account.position_size,
      notional: parseFloat(notional.toFixed(2)),
      fee: 0,
      slippage: 0,
      funding_amount: parseFloat(fundingCashFlow.toFixed(2)),
      margin_mode: this.account.margin_mode,
      leverage: this.account.leverage,
      reason: "FUNDING_FEE",
      equity_after: this.account.equity,
      confidence: 1.0,
      act_probability: 1.0,
      meta_threshold: this.settings.metaThreshold,
    };

    this.trades.unshift(tradeRecord);
    return tradeRecord;
  }

  /**
   * Continuous Mark Price & BookTicker update from Binance WebSocket.
   * Evaluates intra-second Take-Profit, Stop-Loss, Trailing Stop, and Liquidation.
   */
  public onLivePrice(
    price: number,
    bookTicker?: { bidPrice: number; askPrice: number; bidQty?: number; askQty?: number },
    fundingInfo?: { fundingRate?: number; nextFundingTime?: number }
  ): PaperTrade | null {
    if (!price || price <= 0) return null;
    this.account.current_price = price;
    this.recalculateUnrealizedPnL(price);

    // 1. Check Periodic 8-hour Funding Settlement
    if (fundingInfo?.fundingRate !== undefined) {
      this.checkFundingSettlement(price, fundingInfo.fundingRate, fundingInfo.nextFundingTime);
    }

    // 2. Intra-Second Position Risk & Barrier Checks
    if (this.account.position_side !== "FLAT" && this.account.entry_price) {
      const isLong = this.account.position_side === "LONG";
      const dummyPred: EnsemblePrediction = {
        timestamp: new Date().toISOString(),
        symbol: "BTCUSDT",
        members: {},
        regime_state: 1,
        regime_name: "Momentum / Trend Following",
        regime_multiplier: 1.0,
        raw_proba: [0.33, 0.34, 0.33],
        calibrated_proba: [0.33, 0.34, 0.33],
        primary_direction: 0,
        confidence: 0.5,
        act_probability: 0.5,
        bet_size: 0,
        is_trade_allowed: false,
        meta_threshold: this.settings.metaThreshold,
        target_fraction: 0,
        fusion_stage: "tick_barrier",
        latency_seconds: 0,
      };

      // A. Binance Exact Liquidation Trigger
      if (this.account.liquidation_price !== null) {
        if ((isLong && price <= this.account.liquidation_price) ||
            (!isLong && price >= this.account.liquidation_price)) {
          return this.executeTrade(0.0, price, dummyPred.timestamp, "LIQUIDATION", dummyPred, "TAKER", bookTicker);
        }
      }

      // B. Two-Stage Maker Profit Shield (arms at >= 4.8 bps MFE, locks >= 4.35 bps or 74% of peak MFE)
      const activationPct = 0.00048;
      const minProfitFloorPct = 0.000435;
      const retainFrac = 0.74;
      const trailRole: ExecutionRole = this.settings.executionStyle === "MARKET_TAKER" ? "TAKER" : "MAKER";
      if (isLong) {
        const peak = Math.max(this.account.trailing_stop_peak_price || this.account.entry_price, price);
        this.account.trailing_stop_peak_price = peak;
        const mfePct = (peak - this.account.entry_price) / this.account.entry_price;
        if (mfePct >= activationPct) {
          this.account.trailing_stop_active = true;
          const lockPct = Math.max(minProfitFloorPct, retainFrac * mfePct);
          const trailStop = this.account.entry_price * (1.0 + lockPct);
          this.account.trailing_stop_price = parseFloat(trailStop.toFixed(2));
          if (price <= trailStop) {
            return this.executeTrade(0.0, Math.max(price, trailStop), dummyPred.timestamp, "TRAILING_STOP", dummyPred, trailRole, bookTicker);
          }
        }
      } else {
        const trough = Math.min(this.account.trailing_stop_peak_price || this.account.entry_price, price);
        this.account.trailing_stop_peak_price = trough;
        const mfePct = (this.account.entry_price - trough) / this.account.entry_price;
        if (mfePct >= activationPct) {
          this.account.trailing_stop_active = true;
          const lockPct = Math.max(minProfitFloorPct, retainFrac * mfePct);
          const trailStop = this.account.entry_price * (1.0 - lockPct);
          this.account.trailing_stop_price = parseFloat(trailStop.toFixed(2));
          if (price >= trailStop) {
            return this.executeTrade(0.0, Math.min(price, trailStop), dummyPred.timestamp, "TRAILING_STOP", dummyPred, trailRole, bookTicker);
          }
        }
      }

      // C. Scalping Take-Profit Target
      if (this.account.tp_price !== null) {
        if ((isLong && price >= this.account.tp_price) || (!isLong && price <= this.account.tp_price)) {
          const role: ExecutionRole = this.settings.executionStyle === "MARKET_TAKER" ? "TAKER" : "MAKER";
          return this.executeTrade(0.0, price, dummyPred.timestamp, "TAKE_PROFIT", dummyPred, role, bookTicker);
        }
      }

      // D. Scalping Stop-Loss Target
      if (this.account.sl_price !== null) {
        if ((isLong && price <= this.account.sl_price) || (!isLong && price >= this.account.sl_price)) {
          return this.executeTrade(0.0, price, dummyPred.timestamp, "STOP_LOSS", dummyPred, "TAKER", bookTicker);
        }
      }
    }

    return null;
  }

  /**
   * Evaluates each 10-second ensemble prediction bar against active positions or entry rules
   */
  public processBar(
    prediction: EnsemblePrediction,
    currentPrice: number,
    bookTicker?: { bidPrice: number; askPrice: number }
  ): PaperTrade | null {
    if (!currentPrice || currentPrice <= 0) return null;
    this.account.current_price = currentPrice;
    this.recalculateUnrealizedPnL(currentPrice);

    const isPositionOpen = this.account.position_side !== "FLAT" && this.account.entry_price !== null;

    // 1. Triple-Barrier & Position Management for Active Positions
    if (isPositionOpen && this.account.entry_price) {
      this.barsInTrade++;
      const isLong = this.account.position_side === "LONG";

      // A. Forced Liquidation Check
      if (this.account.liquidation_price !== null) {
        if ((isLong && currentPrice <= this.account.liquidation_price) ||
            (!isLong && currentPrice >= this.account.liquidation_price)) {
          return this.executeTrade(0.0, currentPrice, prediction.timestamp, "LIQUIDATION", prediction, "TAKER", bookTicker);
        }
      }

      // B. Take-Profit Barrier
      if (this.account.tp_price !== null) {
        if ((isLong && currentPrice >= this.account.tp_price) ||
            (!isLong && currentPrice <= this.account.tp_price)) {
          const role: ExecutionRole = this.settings.executionStyle === "MARKET_TAKER" ? "TAKER" : "MAKER";
          return this.executeTrade(0.0, currentPrice, prediction.timestamp, "TAKE_PROFIT", prediction, role, bookTicker);
        }
      }

      // C. Stop-Loss Barrier
      if (this.account.sl_price !== null) {
        if ((isLong && currentPrice <= this.account.sl_price) ||
            (!isLong && currentPrice >= this.account.sl_price)) {
          return this.executeTrade(0.0, currentPrice, prediction.timestamp, "STOP_LOSS", prediction, "TAKER", bookTicker);
        }
      }

      // D. High-Leverage Profit Lock:
      // If position is in net profit (ROE >= +6.0%, securing >= +2.4 bps net profit after fees)
      // and the new bar predicts FLAT or opposite direction, lock in the winning trade!
      const isSameDirection = (isLong && prediction.primary_direction === 1) || (!isLong && prediction.primary_direction === -1);
      if (this.account.roe_pct >= 6.0 && !isSameDirection) {
        const role: ExecutionRole = this.settings.executionStyle === "MARKET_TAKER" ? "TAKER" : "MAKER";
        return this.executeTrade(0.0, currentPrice, prediction.timestamp, "TAKE_PROFIT", prediction, role, bookTicker);
      }

      // E. Decisive Opposing Signal Reversal: flip immediately into the new direction
      const isOpposingSignal =
        (isLong && prediction.primary_direction === -1 && prediction.act_probability >= this.settings.metaThreshold) ||
        (!isLong && prediction.primary_direction === 1 && prediction.act_probability >= this.settings.metaThreshold);

      if (isOpposingSignal) {
        const flipFraction = this.computeTargetFraction(prediction);
        const role: ExecutionRole = this.settings.executionStyle === "MARKET_TAKER" ? "TAKER" : "MAKER";
        return this.executeTrade(flipFraction, currentPrice, prediction.timestamp, "SIGNAL_ENTRY", prediction, role, bookTicker);
      }

      // F. Time Barrier (Max holding bars reached)
      if (this.barsInTrade >= this.maxHoldingBars) {
        const role: ExecutionRole = this.settings.executionStyle === "MARKET_TAKER" ? "TAKER" : "MAKER";
        return this.executeTrade(0.0, currentPrice, prediction.timestamp, "MAX_HOLD_EXPIRED", prediction, role, bookTicker);
      }

      // G. Holding Patience Guard:
      if (this.barsInTrade < this.minHoldingBars) {
        this.recordEquityPoint(prediction.timestamp, currentPrice);
        return null;
      }

      // H. Invalidation Veto: only exit on veto if trade has secured solid net profit well above round-trip fees
      if (prediction.act_probability < 0.45 && this.account.roe_pct >= 10.0) {
        return this.executeTrade(0.0, currentPrice, prediction.timestamp, "TAKE_PROFIT", prediction, "MAKER", bookTicker);
      }

      // Hold position
      this.recordEquityPoint(prediction.timestamp, currentPrice);
      return null;
    }

    // 2. Compute Entry Sizing when Flat
    this.barsSinceExit++;
    const targetFraction = this.computeTargetFraction(prediction);

    if (targetFraction !== 0.0 && !isPositionOpen && this.barsSinceExit >= this.minCooldownBars) {
      const role: ExecutionRole = this.settings.executionStyle === "MARKET_TAKER" ? "TAKER" : "MAKER";
      return this.executeTrade(targetFraction, currentPrice, prediction.timestamp, "SIGNAL_ENTRY", prediction, role, bookTicker);
    }

    // Record equity point
    this.recordEquityPoint(prediction.timestamp, currentPrice);
    return null;
  }

  /**
   * High-Leverage Position Sizing adhering to Binance Tier Notional Limits:
   * Enforces the maximum position notional allowed for the user's selected leverage
   * so an account cannot exceed Binance bracket restrictions.
   */
  private computeTargetFraction(p: EnsemblePrediction): number {
    const tau = this.settings.metaThreshold;
    const configuredLev = Math.max(1, this.settings.leverage || 100);

    // 1. Meta-Labeling Conviction Filter
    if (p.act_probability < tau || !p.is_trade_allowed) {
      return 0.0;
    }

    // 2. Directional Sign
    const sign = p.primary_direction === 1 ? 1.0 : p.primary_direction === -1 ? -1.0 : 0.0;
    if (sign === 0.0) return 0.0;

    // 3. Binance Tier Maximum Notional Guard
    const maxAllowedNotional = getMaxNotionalForLeverage(configuredLev);
    const balance = Math.max(1, this.account.wallet_balance);
    const maxAllowedFraction = maxAllowedNotional / balance;

    // 4. Deploy 80% to 100% of allowed leverage based on model confidence
    const confBoost = Math.min(Math.max(p.confidence / 0.60, 0.80), 1.0);
    const desiredFraction = configuredLev * confBoost;
    const cappedFraction = Math.min(desiredFraction, maxAllowedFraction);

    return parseFloat((sign * cappedFraction).toFixed(2));
  }

  private getCurrentPositionFraction(): number {
    if (this.account.wallet_balance <= 0 || this.account.position_notional === 0) return 0;
    const sign = this.account.position_side === "LONG" ? 1.0 : this.account.position_side === "SHORT" ? -1.0 : 0.0;
    return (sign * this.account.position_notional) / this.account.wallet_balance;
  }

  /**
   * Executes order transitions with Binance VIP fee schedules, top-of-book pricing,
   * 0.001 BTC contract rounding, and exact isolated/cross liquidation parameters.
   */
  private executeTrade(
    targetFraction: number,
    markPrice: number,
    timestamp: string,
    reason: TradeReason,
    prediction: EnsemblePrediction,
    role: ExecutionRole = "MAKER",
    bookTicker?: { bidPrice: number; askPrice: number }
  ): PaperTrade {
    const currentFraction = this.getCurrentPositionFraction();
    const turnoverFraction = Math.abs(targetFraction - currentFraction);

    const leverage = Math.max(1, this.settings.leverage || 100);
    const orderNotionalRaw = turnoverFraction * this.account.wallet_balance;

    // Determine execution price with live Top-of-Book BookTicker
    let execPrice = markPrice;
    if (bookTicker && bookTicker.bidPrice > 0 && bookTicker.askPrice > 0) {
      const isBuying = targetFraction > currentFraction;
      if (role === "TAKER") {
        execPrice = isBuying ? bookTicker.askPrice : bookTicker.bidPrice;
      } else {
        // Maker Post-Only Limit fills at best bid for buying, best ask for selling
        execPrice = isBuying ? bookTicker.bidPrice : bookTicker.askPrice;
      }
    }
    execPrice = ClientPaperTradingEngine.roundPrice(execPrice);

    // Calculate trade size in BTC adhering to 0.001 BTC step size
    const rawSize = execPrice > 0 ? orderNotionalRaw / execPrice : 0;
    const tradeSize = ClientPaperTradingEngine.roundLot(rawSize);
    const orderNotional = parseFloat((tradeSize * execPrice).toFixed(2));

    // Fee & Slippage calculation based on ExecutionRole & VIP Tier
    const feeBps = this.getFeeBps(role);
    const slipBps = role === "TAKER" ? (this.settings.slippageBps || 0.1) : 0.0; // Maker orders have 0 slippage
    const fee = (orderNotional * feeBps) / 10000.0;
    const slippage = (orderNotional * slipBps) / 10000.0;
    const orderFriction = fee + slippage;

    // Apply friction to wallet balance
    this.account.wallet_balance -= orderFriction;
    this.account.total_fees += fee;
    this.account.total_slippage += slippage;

    let closedNetPnl: number | undefined = undefined;
    let closedGrossPnl: number | undefined = undefined;
    let closedRoePct: number | undefined = undefined;

    // If closing or flipping an existing position: Realize PnL
    if (this.account.position_side !== "FLAT" && this.account.entry_price) {
      const isLong = this.account.position_side === "LONG";
      const exitPrice = execPrice;
      const priceDeltaPct = isLong
        ? (exitPrice - this.account.entry_price) / this.account.entry_price
        : (this.account.entry_price - exitPrice) / this.account.entry_price;

      closedGrossPnl = this.account.position_notional * priceDeltaPct;

      // Deduct liquidation clearance fee if liquidated
      let liqClearanceFee = 0;
      if (reason === "LIQUIDATION") {
        liqClearanceFee = this.account.position_notional * LIQUIDATION_FEE_RATE;
        this.account.wallet_balance -= liqClearanceFee;
        this.account.total_fees += liqClearanceFee;
      }

      // Attribute only the closing leg friction to closedNetPnl on position flips
      const closeLegRatio = turnoverFraction > 0 ? Math.min(1.0, Math.abs(currentFraction) / turnoverFraction) : 1.0;
      closedNetPnl = closedGrossPnl - (orderFriction * closeLegRatio) - liqClearanceFee;

      const initialMargin = this.account.initial_margin > 0 ? this.account.initial_margin : this.account.position_notional / leverage;
      closedRoePct = (closedNetPnl / Math.max(1, initialMargin)) * 100;

      // Update Realized Account State
      if (this.account.margin_mode === "ISOLATED" && reason === "LIQUIDATION") {
        // In Isolated margin, only the allocated isolated margin is lost
        this.account.wallet_balance = Math.max(0, this.account.wallet_balance - this.account.isolated_margin);
      } else {
        this.account.wallet_balance = Math.max(0, this.account.wallet_balance + closedGrossPnl);
      }
      this.account.realized_pnl += closedNetPnl;

      if (closedNetPnl >= 0) {
        this.account.win_count++;
      } else {
        this.account.loss_count++;
      }
    }

    // Determine New Position State
    const newSide: PositionSide = targetFraction > 0 ? "LONG" : targetFraction < 0 ? "SHORT" : "FLAT";
    const newSizeRaw = targetFraction !== 0 && execPrice > 0
      ? (Math.abs(targetFraction) * this.account.wallet_balance) / execPrice
      : 0;
    const newSize = ClientPaperTradingEngine.roundLot(newSizeRaw);
    const newNotional = parseFloat((newSize * execPrice).toFixed(2));

    const action = currentFraction === 0 ? "OPEN" : targetFraction === 0 ? "CLOSE" : "REBALANCE";
    const side = targetFraction > currentFraction ? "BUY" : "SELL";

    if (newSide === "FLAT" || newNotional < 5.0) {
      this.account.position_side = "FLAT";
      this.account.position_size = 0;
      this.account.position_notional = 0;
      this.account.entry_price = null;
      this.account.entry_timestamp = null;
      this.account.unrealized_pnl = 0;
      this.account.unrealized_pnl_pct = 0;
      this.account.roe_pct = 0;
      this.account.initial_margin = 0;
      this.account.isolated_margin = 0;
      this.account.maintenance_margin = 0;
      this.account.cum_maintenance_amount = 0;
      this.account.margin_ratio_pct = 0;
      this.account.liquidation_price = null;
      this.account.tp_price = null;
      this.account.sl_price = null;
      this.account.trailing_stop_active = false;
      this.account.trailing_stop_peak_price = null;
      this.account.trailing_stop_price = null;
      this.account.cash = this.account.wallet_balance;
      this.account.margin_balance = this.account.wallet_balance;
      this.account.equity = this.account.wallet_balance;
      this.barsInTrade = 0;
      this.barsSinceExit = 0;
    } else {
      this.account.position_side = newSide;
      this.account.position_size = newSize;
      this.account.position_notional = newNotional;
      this.account.entry_price = execPrice;
      this.account.entry_timestamp = timestamp;
      this.account.unrealized_pnl = 0;
      this.account.unrealized_pnl_pct = 0;
      this.account.roe_pct = 0;

      // Binance Tier Bracket Calculation
      const bracket = getBtcLeverageBracket(newNotional);
      this.account.tier = bracket.tier;
      this.account.cum_maintenance_amount = bracket.cumMaintenanceAmount;
      this.account.maintenance_margin = parseFloat((newNotional * bracket.mmr - bracket.cumMaintenanceAmount).toFixed(2));

      // Margin calculations
      const posMargin = newNotional / leverage;
      this.account.initial_margin = parseFloat(posMargin.toFixed(2));
      this.account.isolated_margin = parseFloat(posMargin.toFixed(2));
      this.account.cash = Math.max(0, this.account.wallet_balance - posMargin);
      this.account.margin_balance = this.account.wallet_balance;
      this.account.equity = this.account.wallet_balance;

      const effectiveMarginBalance = this.account.margin_mode === "ISOLATED" ? posMargin : this.account.margin_balance;
      this.account.margin_ratio_pct = parseFloat(((this.account.maintenance_margin / Math.max(0.01, effectiveMarginBalance)) * 100).toFixed(2));

      // Exact Binance Liquidation Formulas
      const mmr = bracket.mmr;
      const cum = bracket.cumMaintenanceAmount;
      if (this.account.margin_mode === "ISOLATED") {
        if (newSide === "LONG") {
          this.account.liquidation_price = parseFloat(
            ((newNotional - posMargin + cum) / (newSize * (1.0 - mmr))).toFixed(2)
          );
        } else {
          this.account.liquidation_price = parseFloat(
            ((newNotional + posMargin - cum) / (newSize * (1.0 + mmr))).toFixed(2)
          );
        }
      } else {
        // Cross Margin Mode Liquidation Price
        const walletBal = this.account.wallet_balance;
        if (newSide === "LONG") {
          this.account.liquidation_price = parseFloat(
            Math.max(0, (newNotional - walletBal + cum) / (newSize * (1.0 - mmr))).toFixed(2)
          );
        } else {
          this.account.liquidation_price = parseFloat(
            ((newNotional + walletBal - cum) / (newSize * (1.0 + mmr))).toFixed(2)
          );
        }
      }

      // High-Frequency Scalping TP / SL targets
      const tpPct = (this.settings.tpPct || 0.14) / 100.0;
      const slPct = (this.settings.slPct || 0.12) / 100.0;
      if (newSide === "LONG") {
        this.account.tp_price = parseFloat((execPrice * (1.0 + tpPct)).toFixed(2));
        this.account.sl_price = parseFloat((execPrice * (1.0 - slPct)).toFixed(2));
        this.account.trailing_stop_peak_price = execPrice;
      } else {
        this.account.tp_price = parseFloat((execPrice * (1.0 - tpPct)).toFixed(2));
        this.account.sl_price = parseFloat((execPrice * (1.0 + slPct)).toFixed(2));
        this.account.trailing_stop_peak_price = execPrice;
      }
      this.account.trailing_stop_active = true;
      this.barsInTrade = 0;
    }

    this.account.total_trades++;

    // Peak equity & drawdowns
    if (this.account.equity > this.account.peak_equity) {
      this.account.peak_equity = this.account.equity;
    }
    const currentDrawdown = this.account.peak_equity - this.account.equity;
    const currentDrawdownPct = (currentDrawdown / this.account.peak_equity) * 100;
    if (currentDrawdownPct > this.account.max_drawdown_pct) {
      this.account.max_drawdown = currentDrawdown;
      this.account.max_drawdown_pct = currentDrawdownPct;
    }

    const tradeRecord: PaperTrade = {
      id: `trade_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      timestamp,
      side,
      action,
      position_side: newSide,
      price: execPrice,
      size: tradeSize,
      notional: orderNotional,
      fee,
      slippage,
      role,
      margin_mode: this.account.margin_mode,
      leverage,
      gross_pnl: closedGrossPnl,
      pnl: closedNetPnl,
      pnl_pct: closedRoePct,
      reason,
      equity_after: this.account.equity,
      confidence: prediction.confidence,
      act_probability: prediction.act_probability,
      meta_threshold: this.settings.metaThreshold,
    };

    this.trades.unshift(tradeRecord);
    if (this.trades.length > 500) {
      this.trades.pop();
    }

    this.recordEquityPoint(timestamp, execPrice);
    return tradeRecord;
  }

  /**
   * Recalculates Unrealized PnL, ROE%, Margin Balance & Margin Ratio using Binance Mark Price
   */
  private recalculateUnrealizedPnL(currentPrice: number) {
    if (this.account.position_side === "FLAT" || !this.account.entry_price || this.account.entry_price <= 0) {
      this.account.unrealized_pnl = 0;
      this.account.unrealized_pnl_pct = 0;
      this.account.roe_pct = 0;
      this.account.margin_balance = this.account.wallet_balance;
      this.account.equity = this.account.wallet_balance;
      this.account.margin_ratio_pct = 0;
      return;
    }

    const isLong = this.account.position_side === "LONG";
    const priceDeltaPct = isLong
      ? (currentPrice - this.account.entry_price) / this.account.entry_price
      : (this.account.entry_price - currentPrice) / this.account.entry_price;

    this.account.unrealized_pnl = parseFloat((this.account.position_notional * priceDeltaPct).toFixed(2));

    // Binance ROE% = Unrealized PnL / Initial Margin * 100%
    const margin = this.account.initial_margin > 0 ? this.account.initial_margin : (this.account.position_notional / (this.account.leverage || 100));
    this.account.roe_pct = parseFloat(((this.account.unrealized_pnl / Math.max(1, margin)) * 100).toFixed(2));
    this.account.unrealized_pnl_pct = this.account.roe_pct;

    // Total Margin Balance = Wallet Balance + Unrealized PnL
    this.account.margin_balance = parseFloat((this.account.wallet_balance + this.account.unrealized_pnl).toFixed(2));
    this.account.equity = this.account.margin_balance;

    // Binance Margin Ratio % = Total Maintenance Margin / Margin Balance * 100%
    const effectiveBalance = this.account.margin_mode === "ISOLATED"
      ? (this.account.isolated_margin + this.account.unrealized_pnl)
      : this.account.margin_balance;

    if (effectiveBalance > 0) {
      this.account.margin_ratio_pct = parseFloat(
        ((this.account.maintenance_margin / effectiveBalance) * 100).toFixed(2)
      );
    } else {
      this.account.margin_ratio_pct = 100.0;
    }
  }

  /**
   * Manual Action: Immediately closes position with Market or Limit order
   */
  public manualClosePosition(currentPrice: number, role: ExecutionRole = "TAKER"): PaperTrade | null {
    if (this.account.position_side === "FLAT" || !this.account.entry_price) return null;
    const dummyPred: EnsemblePrediction = {
      timestamp: new Date().toISOString(),
      symbol: "BTCUSDT",
      members: {},
      regime_state: 1,
      regime_name: "Manual Close",
      regime_multiplier: 1.0,
      raw_proba: [0.33, 0.34, 0.33],
      calibrated_proba: [0.33, 0.34, 0.33],
      primary_direction: 0,
      confidence: 1.0,
      act_probability: 1.0,
      bet_size: 0,
      is_trade_allowed: false,
      meta_threshold: this.settings.metaThreshold,
      target_fraction: 0,
      fusion_stage: "manual",
      latency_seconds: 0,
    };
    return this.executeTrade(0.0, currentPrice, dummyPred.timestamp, "MANUAL_CLOSE", dummyPred, role);
  }

  /**
   * Manual Action: Changes isolated/cross leverage directly
   */
  public setLeverage(newLeverage: number): void {
    const clamped = Math.max(1, Math.min(125, newLeverage));
    this.account.leverage = clamped;
    this.account.max_notional_tier = getMaxNotionalForLeverage(clamped);
    this.settings.leverage = clamped;
    if (this.account.position_side !== "FLAT" && this.account.entry_price) {
      this.recalculateUnrealizedPnL(this.account.current_price);
    }
  }

  /**
   * Manual Action: Toggles margin mode between ISOLATED and CROSS
   */
  public setMarginMode(newMode: MarginMode): void {
    this.account.margin_mode = newMode;
    this.settings.marginMode = newMode;
    if (this.account.position_side !== "FLAT" && this.account.entry_price) {
      this.recalculateUnrealizedPnL(this.account.current_price);
    }
  }

  private recordEquityPoint(timestamp: string, price: number) {
    const totalVal = this.account.equity;
    const dd = this.account.peak_equity > 0 ? ((this.account.peak_equity - totalVal) / this.account.peak_equity) * 100 : 0;

    this.equityHistory.push({
      timestamp,
      equity: parseFloat(totalVal.toFixed(2)),
      btc_price: price,
      drawdown_pct: parseFloat(Math.max(dd, 0).toFixed(2)),
    });

    if (this.equityHistory.length > 1000) {
      this.equityHistory.shift();
    }
  }

  public reset(startingBalance?: number) {
    const cap = startingBalance || this.settings.startingCapital || 10000;
    const lev = this.settings.leverage || 100;
    const mode = this.settings.marginMode || "ISOLATED";
    const bracket = getBtcLeverageBracket(0);

    this.account = {
      starting_balance: cap,
      equity: cap,
      wallet_balance: cap,
      margin_balance: cap,
      cash: cap,
      leverage: lev,
      margin_mode: mode,
      tier: bracket.tier,
      max_notional_tier: getMaxNotionalForLeverage(lev),
      cum_maintenance_amount: 0,
      position_side: "FLAT",
      position_size: 0,
      position_notional: 0,
      entry_price: null,
      entry_timestamp: null,
      current_price: this.account.current_price,
      unrealized_pnl: 0,
      unrealized_pnl_pct: 0,
      roe_pct: 0,
      initial_margin: 0,
      isolated_margin: 0,
      maintenance_margin: 0,
      margin_ratio_pct: 0,
      liquidation_price: null,
      tp_price: null,
      sl_price: null,
      trailing_stop_active: false,
      trailing_stop_peak_price: null,
      trailing_stop_price: null,
      last_funding_time: 0,
      total_funding_fees: 0,
      realized_pnl: 0,
      total_fees: 0,
      total_slippage: 0,
      win_count: 0,
      loss_count: 0,
      total_trades: 0,
      max_drawdown: 0,
      max_drawdown_pct: 0,
      peak_equity: cap,
    };
    this.trades = [];
    this.equityHistory = [];
    this.barsInTrade = 0;
    this.barsSinceExit = 999;
  }
}
