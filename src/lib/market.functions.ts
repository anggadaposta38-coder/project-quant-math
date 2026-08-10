import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import type { Candle, Interval } from "./market.server";

export const UNIVERSE = [
  "BTCUSDT",
  "ETHUSDT",
  "SOLUSDT",
  "BNBUSDT",
  "XRPUSDT",
  "ADAUSDT",
  "DOGEUSDT",
  "AVAXUSDT",
] as const;

export interface MarketPayload {
  interval: Interval;
  limit: number;
  fetchedAt: number;
  sources: string[];
  failed: string[];
  series: { symbol: string; candles: Candle[]; source: string }[];
}

const inputSchema = z.object({
  interval: z.enum(["1h", "4h", "1d"]).default("4h"),
  limit: z.number().int().min(100).max(1000).default(500),
});

export const getMarketData = createServerFn({ method: "POST" })
  .inputValidator((data: unknown) => inputSchema.parse(data))
  .handler(async ({ data }): Promise<MarketPayload> => {
    const { fetchSeries } = await import("./market.server");
    const results = await Promise.all(
      UNIVERSE.map((s) => fetchSeries(s, data.interval, data.limit)),
    );
    const series = results
      .filter((r): r is NonNullable<typeof r> => r !== null)
      .map((r) => ({ symbol: r.symbol, candles: r.candles, source: r.source }));

    return {
      interval: data.interval,
      limit: data.limit,
      fetchedAt: Date.now(),
      sources: Array.from(new Set(series.map((s) => s.source))),
      failed: UNIVERSE.filter((s) => !series.some((x) => x.symbol === s)),
      series,
    };
  });
