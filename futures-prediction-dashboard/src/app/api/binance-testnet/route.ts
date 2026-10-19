import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";

const DEFAULT_BASE_URL = "https://testnet.binancefuture.com";

function getCredentials(req?: NextRequest) {
  const apiKey = process.env.BINANCE_TESTNET_API_KEY || "";
  const apiSecret = process.env.BINANCE_TESTNET_API_SECRET || "";
  const baseUrl = process.env.BINANCE_TESTNET_BASE_URL || DEFAULT_BASE_URL;
  return { apiKey, apiSecret, baseUrl };
}

async function sendBinanceSignedRequest(
  method: "GET" | "POST" | "DELETE",
  path: string,
  params: Record<string, any> = {},
  apiKey: string,
  apiSecret: string,
  baseUrl: string
) {
  // 1. Fetch server time
  let serverTime = Date.now();
  try {
    const timeRes = await fetch(`${baseUrl}/fapi/v1/time`, { cache: "no-store" });
    if (timeRes.ok) {
      const timeData = await timeRes.json();
      serverTime = timeData.serverTime;
    }
  } catch (e) {
    // fallback to local time
  }

  const queryParams = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) {
      queryParams.append(k, String(v));
    }
  }
  queryParams.append("timestamp", String(serverTime));
  queryParams.append("recvWindow", "10000");

  const queryString = queryParams.toString();
  const signature = crypto.createHmac("sha256", apiSecret).update(queryString).digest("hex");
  const signedQuery = `${queryString}&signature=${signature}`;

  const headers: Record<string, string> = {
    "X-MBX-APIKEY": apiKey,
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "FuturesPredictionDashboard/1.0",
  };

  let url = `${baseUrl}${path}`;
  let fetchOptions: RequestInit = {
    method,
    headers,
    cache: "no-store",
  };

  if (method === "GET" || method === "DELETE") {
    url = `${url}?${signedQuery}`;
  } else if (method === "POST") {
    fetchOptions.body = signedQuery;
  }

  const resp = await fetch(url, fetchOptions);
  const data = await resp.json();

  if (!resp.ok) {
    const err: any = new Error(data.msg || `Binance API Error: ${resp.status}`);
    err.code = data.code;
    err.status = resp.status;
    throw err;
  }

  return data;
}

export async function GET(req: NextRequest) {
  const { apiKey, apiSecret, baseUrl } = getCredentials(req);
  if (!apiKey || !apiSecret) {
    return NextResponse.json(
      { success: false, error: "Binance Testnet API credentials not configured." },
      { status: 400 }
    );
  }

  const { searchParams } = new URL(req.url);
  const action = searchParams.get("action") || "account";

  try {
    if (action === "account") {
      const acc = await sendBinanceSignedRequest("GET", "/fapi/v2/account", {}, apiKey, apiSecret, baseUrl);
      const posRisk = await sendBinanceSignedRequest(
        "GET",
        "/fapi/v2/positionRisk",
        { symbol: "BTCUSDT" },
        apiKey,
        apiSecret,
        baseUrl
      );

      const btcPos = Array.isArray(posRisk) && posRisk.length > 0 ? posRisk[0] : null;

      return NextResponse.json({
        success: true,
        totalWalletBalance: parseFloat(acc.totalWalletBalance || "0"),
        totalMarginBalance: parseFloat(acc.totalMarginBalance || "0"),
        availableBalance: parseFloat(acc.availableBalance || "0"),
        totalUnrealizedProfit: parseFloat(acc.totalUnrealizedProfit || "0"),
        position: btcPos
          ? {
              symbol: btcPos.symbol,
              positionAmt: parseFloat(btcPos.positionAmt || "0"),
              entryPrice: parseFloat(btcPos.entryPrice || "0"),
              markPrice: parseFloat(btcPos.markPrice || "0"),
              unRealizedProfit: parseFloat(btcPos.unRealizedProfit || "0"),
              leverage: parseInt(btcPos.leverage || "20"),
              marginType: btcPos.marginType,
              liquidationPrice: parseFloat(btcPos.liquidationPrice || "0"),
            }
          : null,
      });
    }

    if (action === "openOrders") {
      const orders = await sendBinanceSignedRequest(
        "GET",
        "/fapi/v1/openOrders",
        { symbol: "BTCUSDT" },
        apiKey,
        apiSecret,
        baseUrl
      );
      return NextResponse.json({ success: true, orders });
    }

    return NextResponse.json({ success: false, error: `Unknown action: ${action}` }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message || String(error), code: error.code },
      { status: error.status || 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  const { apiKey, apiSecret, baseUrl } = getCredentials(req);
  if (!apiKey || !apiSecret) {
    return NextResponse.json(
      { success: false, error: "Binance Testnet API credentials not configured." },
      { status: 400 }
    );
  }

  try {
    const body = await req.json();
    const action = body.action || "ORDER";

    if (action === "ORDER") {
      const symbol = body.symbol || "BTCUSDT";
      const side = body.side; // BUY or SELL
      const type = body.type || "LIMIT";
      const qty = parseFloat(body.quantity);
      const price = body.price ? parseFloat(body.price).toFixed(1) : undefined;
      const timeInForce = body.timeInForce || (type === "LIMIT" ? "GTX" : undefined); // GTX = Post-Only Maker

      if (!side || !qty || qty <= 0) {
        return NextResponse.json({ success: false, error: "Invalid order parameters" }, { status: 400 });
      }

      const params: Record<string, any> = {
        symbol,
        side,
        type,
        quantity: qty.toFixed(3),
      };

      if (type === "LIMIT") {
        if (!price) {
          return NextResponse.json({ success: false, error: "Price required for LIMIT orders" }, { status: 400 });
        }
        params.price = price;
        params.timeInForce = timeInForce;
      }

      if (body.reduceOnly) {
        params.reduceOnly = "true";
      }

      const orderResult = await sendBinanceSignedRequest("POST", "/fapi/v1/order", params, apiKey, apiSecret, baseUrl);
      return NextResponse.json({ success: true, order: orderResult });
    }

    if (action === "CLOSE_POSITION") {
      const symbol = body.symbol || "BTCUSDT";
      const posRisk = await sendBinanceSignedRequest("GET", "/fapi/v2/positionRisk", { symbol }, apiKey, apiSecret, baseUrl);
      const btcPos = Array.isArray(posRisk) && posRisk.length > 0 ? posRisk[0] : null;
      const currentAmt = btcPos ? parseFloat(btcPos.positionAmt || "0") : 0;

      if (currentAmt === 0) {
        return NextResponse.json({ success: true, message: "Position already flat" });
      }

      // Close position with opposite Market order
      const closeSide = currentAmt > 0 ? "SELL" : "BUY";
      const closeQty = Math.abs(currentAmt).toFixed(3);

      const closeOrder = await sendBinanceSignedRequest(
        "POST",
        "/fapi/v1/order",
        {
          symbol,
          side: closeSide,
          type: "MARKET",
          quantity: closeQty,
          reduceOnly: "true",
        },
        apiKey,
        apiSecret,
        baseUrl
      );

      return NextResponse.json({ success: true, closeOrder });
    }

    if (action === "SET_LEVERAGE") {
      const symbol = body.symbol || "BTCUSDT";
      const leverage = parseInt(body.leverage || "50");
      const levResult = await sendBinanceSignedRequest(
        "POST",
        "/fapi/v1/leverage",
        { symbol, leverage },
        apiKey,
        apiSecret,
        baseUrl
      );
      return NextResponse.json({ success: true, leverage: levResult });
    }

    if (action === "CANCEL_ALL") {
      const symbol = body.symbol || "BTCUSDT";
      const cancelResult = await sendBinanceSignedRequest(
        "DELETE",
        "/fapi/v1/allOpenOrders",
        { symbol },
        apiKey,
        apiSecret,
        baseUrl
      );
      return NextResponse.json({ success: true, result: cancelResult });
    }

    return NextResponse.json({ success: false, error: `Unsupported action: ${action}` }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message || String(error), code: error.code },
      { status: error.status || 500 }
    );
  }
}
