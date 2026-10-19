import { NextRequest, NextResponse } from "next/server";
import { AggTrade, BinanceTicker } from "@/types";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const symbol = (searchParams.get("symbol") || "BTCUSDT").toUpperCase();

  try {
    const [tickerRes, bookRes, premiumRes, tradesRes] = await Promise.all([
      fetch(`https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=${symbol}`, {
        cache: "no-store",
        headers: { "User-Agent": "Mozilla/5.0" },
      }),
      fetch(`https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol=${symbol}`, {
        cache: "no-store",
        headers: { "User-Agent": "Mozilla/5.0" },
      }),
      fetch(`https://fapi.binance.com/fapi/v1/premiumIndex?symbol=${symbol}`, {
        cache: "no-store",
        headers: { "User-Agent": "Mozilla/5.0" },
      }),
      fetch(`https://fapi.binance.com/fapi/v1/aggTrades?symbol=${symbol}&limit=20`, {
        cache: "no-store",
        headers: { "User-Agent": "Mozilla/5.0" },
      }).catch(() => null),
    ]);

    if (!tickerRes.ok) {
      throw new Error(`Binance API error: ${tickerRes.status}`);
    }

    const tData = await tickerRes.json();
    const bData = bookRes.ok ? await bookRes.json() : {};
    const pData = premiumRes.ok ? await premiumRes.json() : {};
    const trData = tradesRes && tradesRes.ok ? await tradesRes.json() : [];

    const price = parseFloat(tData.lastPrice) || 0;
    const bidPrice = parseFloat(bData.bidPrice) || price;
    const askPrice = parseFloat(bData.askPrice) || price;

    const ticker: BinanceTicker = {
      symbol: symbol,
      price: price,
      bidPrice: bidPrice,
      askPrice: askPrice,
      bidQty: parseFloat(bData.bidQty) || 0,
      askQty: parseFloat(bData.askQty) || 0,
      markPrice: parseFloat(pData.markPrice) || price,
      fundingRate: parseFloat(pData.lastFundingRate) || 0.0001,
      nextFundingTime: pData.nextFundingTime || Date.now() + 8 * 3600 * 1000,
      high24h: parseFloat(tData.highPrice) || 0,
      low24h: parseFloat(tData.lowPrice) || 0,
      volume24h: parseFloat(tData.volume) || 0,
      priceChange24h: parseFloat(tData.priceChange) || 0,
      priceChangePercent24h: parseFloat(tData.priceChangePercent) || 0,
      lastUpdate: Date.now(),
    };

    const trades: AggTrade[] = Array.isArray(trData)
      ? trData.map((d: any) => ({
          tradeId: d.a,
          price: parseFloat(d.p),
          quantity: parseFloat(d.q),
          timestamp: d.T,
          isBuyerMaker: d.m,
        }))
      : [];

    return NextResponse.json({ success: true, ticker, trades });
  } catch (err: any) {
    return NextResponse.json(
      { success: false, error: err.message || "Failed to fetch Binance ticker" },
      { status: 500 }
    );
  }
}
