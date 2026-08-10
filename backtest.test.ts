import { expect, test } from "vitest";
import { runWalkForwardBacktest, DEFAULT_BACKTEST_CONFIG } from "./src/lib/quant/backtest";
import { mulberry32, randNorm } from "./src/lib/quant/stats";
import type { Candle } from "./src/lib/market.server";

/**
 * Candle sintetis: gabungan GBM (drift+vol berubah per rezim) dengan komponen
 * OU (mean-reverting) di sekitar harga log, supaya kedua strategi (composite
 * & ou-zone) punya kesempatan menghasilkan sinyal/entry yang nontrivial.
 */
function synthCandles(n: number, seed: number): Candle[] {
  const rng = mulberry32(seed);
  const dt = 1 / (6 * 365); // ~4h bar
  let logP = Math.log(100);
  let ouLevel = logP;
  const out: Candle[] = [];
  let regime = 0;
  for (let i = 0; i < n; i++) {
    if (rng() < 0.01) regime = 1 - regime;
    const mu = regime === 0 ? -0.3 : 0.5;
    const sigma = 0.6;
    // Komponen OU ringan menarik logP kembali ke ouLevel yang bergerak lambat.
    const theta = 4;
    const pull = theta * (ouLevel - logP) * dt;
    const diffusion = sigma * Math.sqrt(dt) * randNorm(rng);
    const drift = (mu - 0.5 * sigma * sigma) * dt;
    logP = logP + drift + pull + diffusion;
    ouLevel = ouLevel + 0.05 * sigma * Math.sqrt(dt) * randNorm(rng);
    const close = Math.exp(logP);
    const wig = close * 0.004 * Math.abs(randNorm(rng));
    const open = i === 0 ? close : out[i - 1]!.c;
    const high = Math.max(open, close) + wig;
    const low = Math.min(open, close) - wig;
    out.push({ t: 1700000000000 + i * 4 * 3600 * 1000, o: open, h: high, l: low, c: close, v: 1000 });
  }
  return out;
}

test("throws on insufficient data", () => {
  const candles = synthCandles(50, 1);
  expect(() => runWalkForwardBacktest(candles, "4h")).toThrow();
});

test(
  "produces well-formed equity curves & metrics",
  { timeout: 120000 },
  () => {
    const candles = synthCandles(400, 42);
    const result = runWalkForwardBacktest(candles, "4h");

    for (const res of [result.composite, result.ouZone]) {
      expect(res.equityCurve.length).toBeGreaterThan(10);
      expect(res.equityCurve[0]!.equity).toBeCloseTo(1, 8);
      // waktu monoton naik, tidak ada NaN/Infinity di equity
      for (let i = 1; i < res.equityCurve.length; i++) {
        expect(res.equityCurve[i]!.t).toBeGreaterThan(res.equityCurve[i - 1]!.t);
        expect(Number.isFinite(res.equityCurve[i]!.equity)).toBe(true);
        expect(res.equityCurve[i]!.equity).toBeGreaterThan(0);
      }
      expect(res.metrics.exposurePct).toBeGreaterThanOrEqual(0);
      expect(res.metrics.exposurePct).toBeLessThanOrEqual(1);
      expect(res.metrics.winRate).toBeGreaterThanOrEqual(0);
      expect(res.metrics.winRate).toBeLessThanOrEqual(1);
      expect(res.metrics.numTrades).toBe(res.trades.length);
      expect(Number.isFinite(res.metrics.maxDrawdown)).toBe(true);
      expect(res.metrics.maxDrawdown).toBeLessThanOrEqual(0);

      // tiap trade: exit setelah entry, harga & return masuk akal
      for (const t of res.trades) {
        expect(t.exitIndex).toBeGreaterThan(t.entryIndex);
        expect(t.entryPrice).toBeGreaterThan(0);
        expect(t.exitPrice).toBeGreaterThan(0);
        expect(Number.isFinite(t.retPct)).toBe(true);
      }
    }
  },
);

test(
  "no lookahead: prefix of a longer run matches a run truncated at that prefix",
  { timeout: 180000 },
  () => {
    const full = synthCandles(400, 7);
    const partial = full.slice(0, 260);
    const cfg = { ...DEFAULT_BACKTEST_CONFIG };

    const rFull = runWalkForwardBacktest(full, "4h", cfg);
    const rPartial = runWalkForwardBacktest(partial, "4h", cfg);

    // Pilih titik waktu yang jauh dari batas potong (260) agar posisi apa pun
    // yang terbuka di sekitar batas sudah pasti selesai sebelum titik ini.
    const cutoffTime = full[200]!.t;

    for (const [a, b] of [
      [rFull.composite, rPartial.composite],
      [rFull.ouZone, rPartial.ouZone],
    ] as const) {
      const mapA = new Map(a.equityCurve.map((p) => [p.t, p.equity]));
      const mapB = new Map(b.equityCurve.map((p) => [p.t, p.equity]));
      let compared = 0;
      for (const [t, eqB] of mapB) {
        if (t > cutoffTime) continue;
        const eqA = mapA.get(t);
        expect(eqA).toBeDefined();
        expect(eqA!).toBeCloseTo(eqB, 8);
        compared++;
      }
      expect(compared).toBeGreaterThan(20);

      // trade yang seluruhnya terjadi sebelum cutoff harus identik persis.
      const tradesA = a.trades.filter((t) => t.exitTime <= cutoffTime);
      const tradesB = b.trades.filter((t) => t.exitTime <= cutoffTime);
      expect(tradesB).toEqual(tradesA);
    }
  },
);

test("buy-and-hold benchmark matches raw price change over the tested window", () => {
  const candles = synthCandles(400, 99);
  const result = runWalkForwardBacktest(candles, "4h");
  const closes = candles.map((c) => c.c);
  const expected = closes[closes.length - 1]! / closes[DEFAULT_BACKTEST_CONFIG.warmupBars]! - 1;
  expect(result.composite.metrics.buyHoldReturn).toBeCloseTo(expected, 8);
  expect(result.ouZone.metrics.buyHoldReturn).toBeCloseTo(expected, 8);
});
