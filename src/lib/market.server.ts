/**
 * Pengambilan data pasar. Dieksekusi HANYA di server (Cloudflare Worker),
 * sehingga browser pengguna tidak pernah menghubungi domain bursa secara
 * langsung — aman dari pemblokiran ISP / DNS di Indonesia.
 *
 * Urutan fallback: Binance (data-api mirror -> api utama) -> Bitget.
 */

export interface Candle {
  t: number; // epoch ms (open time)
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

export interface SymbolSeries {
  symbol: string;
  candles: Candle[];
  source: "binance" | "bitget";
}

export type Interval = "1h" | "4h" | "1d";

export const BARS_PER_YEAR: Record<Interval, number> = {
  "1h": 24 * 365,
  "4h": 6 * 365,
  "1d": 365,
};

const BINANCE_HOSTS = [
  "https://data-api.binance.vision",
  "https://api.binance.com",
  "https://api1.binance.com",
];

const BITGET_GRANULARITY: Record<Interval, string> = {
  "1h": "1h",
  "4h": "4h",
  "1d": "1day",
};

async function timedFetch(url: string, ms = 8000): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, {
      signal: ctrl.signal,
      headers: { accept: "application/json", "user-agent": "quant-dashboard/1.0" },
    });
  } finally {
    clearTimeout(timer);
  }
}

async function fetchBinance(
  symbol: string,
  interval: Interval,
  limit: number,
): Promise<Candle[] | null> {
  for (const host of BINANCE_HOSTS) {
    try {
      const res = await timedFetch(
        `${host}/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`,
      );
      if (!res.ok) continue;
      const raw = (await res.json()) as unknown[];
      if (!Array.isArray(raw) || raw.length === 0) continue;
      return raw.map((row) => {
        const r = row as (string | number)[];
        return {
          t: Number(r[0]),
          o: Number(r[1]),
          h: Number(r[2]),
          l: Number(r[3]),
          c: Number(r[4]),
          v: Number(r[5]),
        };
      });
    } catch {
      // coba host berikutnya
    }
  }
  return null;
}

async function fetchBitget(
  symbol: string,
  interval: Interval,
  limit: number,
): Promise<Candle[] | null> {
  try {
    const res = await timedFetch(
      `https://api.bitget.com/api/v2/spot/market/candles?symbol=${symbol}&granularity=${BITGET_GRANULARITY[interval]}&limit=${Math.min(limit, 1000)}`,
    );
    if (!res.ok) return null;
    const json = (await res.json()) as { code?: string; data?: unknown };
    if (!json || !Array.isArray(json.data) || json.data.length === 0) return null;
    // Bitget: [ts, open, high, low, close, baseVol, quoteVol, usdtVol]
    return (json.data as (string | number)[][])
      .map((r) => ({
        t: Number(r[0]),
        o: Number(r[1]),
        h: Number(r[2]),
        l: Number(r[3]),
        c: Number(r[4]),
        v: Number(r[5]),
      }))
      .sort((a, b) => a.t - b.t);
  } catch {
    return null;
  }
}

export async function fetchSeries(
  symbol: string,
  interval: Interval,
  limit: number,
): Promise<SymbolSeries | null> {
  const b = await fetchBinance(symbol, interval, limit);
  if (b && b.length > 30) return { symbol, candles: b, source: "binance" };
  const g = await fetchBitget(symbol, interval, limit);
  if (g && g.length > 30) return { symbol, candles: g, source: "bitget" };
  return null;
}
