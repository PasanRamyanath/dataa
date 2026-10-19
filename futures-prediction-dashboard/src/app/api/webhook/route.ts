import { NextRequest, NextResponse } from "next/server";

// In-memory circular buffer for received webhook events
const receivedWebhooks: any[] = [];
const MAX_WEBHOOK_STORE = 100;

export async function POST(req: NextRequest) {
  try {
    const payload = await req.json();
    const secret = req.headers.get("x-stream-secret") || req.headers.get("authorization");

    const record = {
      receivedAt: new Date().toISOString(),
      payload,
    };

    receivedWebhooks.unshift(record);
    if (receivedWebhooks.length > MAX_WEBHOOK_STORE) {
      receivedWebhooks.pop();
    }

    return NextResponse.json({ status: "ok", count: receivedWebhooks.length });
  } catch (e: any) {
    return NextResponse.json({ error: e.message || "Invalid payload" }, { status: 400 });
  }
}

export async function GET() {
  return NextResponse.json({
    total: receivedWebhooks.length,
    events: receivedWebhooks,
  });
}
