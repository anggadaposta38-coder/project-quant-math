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

/** Kalender crypto berjalan 24/7. Gunakan 365 hari/tahun secara konsisten
 * untuk annualization; tidak memakai 252 hari bursa. */
export const DAYS_PER_YEAR = 365;
export const YEAR_MS = DAYS_PER_YEAR * 24 * 60 * 60 * 1000;

export const BARS_PER_YEAR: Record<Interval, number> = {
  "1h": YEAR_MS / (60 * 60 * 1000),
  "4h": YEAR_MS / (4 * 60 * 60 * 1000),
  "1d": YEAR_MS / (24 * 60 * 60 * 1000),
};

const BINANCE_HOSTS = [
  "https://data-api.binance.vision",
  "https://api.binance.com",
  "https://api1.binance.com",
];

const INTERVAL_MS: Record<Interval, number> = {
  "1h": 60 * 60 * 1000,
  "4h": 4 * 60 * 60 * 1000,
  "1d": 24 * 60 * 60 * 1000,
};

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

function sameCandle(a: Candle, b: Candle): boolean {
  return a.t === b.t && a.o === b.o && a.h === b.h && a.l === b.l && a.c === b.c && a.v === b.v;
}

/**
 * Normalisasi exchange candle sebelum data masuk ke model quant.
 *
 * Kontrak:
 * - urutan input boleh arbitrary; output selalu chronological;
 * - duplicate timestamp identik boleh dikolaps; duplicate yang konflik ditolak;
 * - candle yang belum close dibuang;
 * - OHLCV harus finite dan OHLC harus konsisten;
 * - open time harus tepat pada boundary interval UTC;
 * - gap candle TIDAK diinterpolasi/diisi; source ditolak agar model tidak
 *   diam-diam mengubah satu return menjadi multi-period return.
 */
export function normalizeCandlesForInterval(
  candles: Candle[],
  interval: Interval,
  now = Date.now(),
): Candle[] | null {
  const step = INTERVAL_MS[interval];
  if (!Number.isFinite(now) || !Number.isFinite(step)) return null;

  const sorted = [...candles].sort((a, b) => a.t - b.t);
  const deduped: Candle[] = [];
  for (const candle of sorted) {
    const prev = deduped[deduped.length - 1];
    if (prev && candle.t === prev.t) {
      if (!sameCandle(prev, candle)) return null;
      continue;
    }
    deduped.push(candle);
  }

  const closed = deduped.filter((c) => c.t + step <= now);
  if (closed.length < 2) return null;

  for (let i = 0; i < closed.length; i++) {
    const c = closed[i]!;
    if (
      !Number.isFinite(c.t) || !Number.isFinite(c.o) || !Number.isFinite(c.h) ||
      !Number.isFinite(c.l) || !Number.isFinite(c.c) || !Number.isFinite(c.v) ||
      !Number.isInteger(c.t) || c.t < 0 || c.t % step !== 0 ||
      c.o <= 0 || c.h <= 0 || c.l <= 0 || c.c <= 0 || c.v < 0 ||
      c.h < Math.max(c.o, c.c) || c.l > Math.min(c.o, c.c)
    ) return null;
    if (i > 0 && c.t - closed[i - 1]!.t !== step) return null;
  }

  return closed;
}

export async function fetchSeries(
  symbol: string,
  interval: Interval,
  limit: number,
): Promise<SymbolSeries | null> {
  const normalize = (raw: Candle[] | null): Candle[] | null =>
    raw ? normalizeCandlesForInterval(raw, interval) : null;

  const minimumUsableCandles = 60;

  const b = normalize(await fetchBinance(symbol, interval, limit));
  if (b && b.length >= minimumUsableCandles) return { symbol, candles: b, source: "binance" };
  const g = normalize(await fetchBitget(symbol, interval, limit));
  if (g && g.length >= minimumUsableCandles) return { symbol, candles: g, source: "bitget" };
  return null;
}
