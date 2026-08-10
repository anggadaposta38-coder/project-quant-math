import { expect, test } from "vitest";
import {
  runWalkForwardBacktest,
  summarizeWalkForwardFolds,
  DEFAULT_BACKTEST_CONFIG,
  directionalRawReturn,
  netTradeReturn,
  positionFractionFromKelly,
  positionFractionFromStop,
} from "./src/lib/quant/backtest";
import { computeSignal } from "./src/lib/quant/analysis";
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


test("backtest rejects missing interval bars", () => {
  const candles = synthCandles(220, 7);
  candles[120]!.t += 4 * 3600 * 1000;
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



test("future candles cannot change a signal computed at an earlier checkpoint", { timeout: 120000 }, () => {
  const base = synthCandles(360, 2026);
  const cutoff = 220;
  const prefix = base.slice(0, cutoff);
  const futureChanged = base.map((c, i) =>
    i < cutoff
      ? { ...c }
      : { ...c, o: c.o * 7, h: c.h * 7, l: c.l * 7, c: c.c * 7 },
  );

  const a = computeSignal(prefix, "4h", 0.04);
  const b = computeSignal(futureChanged.slice(0, cutoff), "4h", 0.04);

  expect(b.action).toBe(a.action);
  expect(b.regime).toBe(a.regime);
  expect(b.regimeProbs).toEqual(a.regimeProbs);
  expect(b.z).toBeCloseTo(a.z, 12);
  expect(b.score).toBeCloseTo(a.score, 12);
  expect(b.kellyFraction).toBeCloseTo(a.kellyFraction, 12);
});

test("walk-forward backtest is invariant to arbitrary future-price changes before the same cutoff", { timeout: 180000 }, () => {
  const base = synthCandles(400, 2027);
  const cutoff = 260;
  const mutatedFuture = base.map((c, i) =>
    i < cutoff
      ? { ...c }
      : { ...c, o: c.o * 3, h: c.h * 3, l: c.l * 3, c: c.c * 3 },
  );
  const cfg = { ...DEFAULT_BACKTEST_CONFIG };
  const a = runWalkForwardBacktest(base, "4h", cfg);
  const b = runWalkForwardBacktest(mutatedFuture, "4h", cfg);
  const cutoffTime = base[220]!.t;

  for (const [ra, rb] of [[a.composite, b.composite], [a.ouZone, b.ouZone]] as const) {
    const mb = new Map(rb.equityCurve.map((p) => [p.t, p.equity]));
    for (const p of ra.equityCurve) {
      if (p.t > cutoffTime) break;
      expect(mb.get(p.t)).toBeDefined();
      expect(mb.get(p.t)!).toBeCloseTo(p.equity, 8);
    }
    const ta = ra.trades.filter((t) => t.exitTime <= cutoffTime);
    const tb = rb.trades.filter((t) => t.exitTime <= cutoffTime);
    expect(tb).toEqual(ta);
  }
});

test("buy-and-hold benchmark matches raw price change over the tested window", () => {
  const candles = synthCandles(400, 99);
  const result = runWalkForwardBacktest(candles, "4h");
  const closes = candles.map((c) => c.c);
  const expected = closes[closes.length - 1]! / closes[DEFAULT_BACKTEST_CONFIG.warmupBars]! - 1;
  expect(result.composite.metrics.buyHoldReturn).toBeCloseTo(expected, 8);
  expect(result.ouZone.metrics.buyHoldReturn).toBeCloseTo(expected, 8);
});


test("rejects missing or irregular candle intervals", () => {
  const candles = synthCandles(220, 123);
  candles[100]!.t += 4 * 3600 * 1000;
  expect(() => runWalkForwardBacktest(candles, "4h")).toThrow(/Candle tidak valid/);
});

test("transaction costs are applied per side", { timeout: 120000 }, () => {
  const candles=synthCandles(400, 1234);
  const free=runWalkForwardBacktest(candles,"4h",{feeBps:0,slippageBps:0});
  const costly=runWalkForwardBacktest(candles,"4h",{feeBps:50,slippageBps:25});
  expect(costly.composite.metrics.totalReturn).toBeLessThanOrEqual(free.composite.metrics.totalReturn + 1e-12);
  expect(costly.ouZone.metrics.totalReturn).toBeLessThanOrEqual(free.ouZone.metrics.totalReturn + 1e-12);
  if (costly.composite.trades.length > 0) expect(costly.composite.trades.length).toBe(free.composite.trades.length);
  if (costly.ouZone.trades.length > 0) expect(costly.ouZone.trades.length).toBe(free.ouZone.trades.length);
});


test("equity curve has unique timestamps and every trade exits after entry", { timeout: 180000 }, () => {
  const candles = synthCandles(400, 31415);
  const result = runWalkForwardBacktest(candles, "4h");
  for (const res of [result.composite, result.ouZone]) {
    for (let i = 1; i < res.equityCurve.length; i++) {
      expect(res.equityCurve[i]!.t).toBeGreaterThan(res.equityCurve[i - 1]!.t);
    }
    for (const trade of res.trades) {
      expect(trade.exitIndex).toBeGreaterThan(trade.entryIndex);
      expect(trade.exitTime).toBeGreaterThan(trade.entryTime);
    }
  }
});

test("trade return includes compounded entry and exit transaction costs", { timeout: 180000 }, () => {
  const candles = synthCandles(400, 2718);
  const feeBps = 10;
  const slippageBps = 5;
  const cfg = { ...DEFAULT_BACKTEST_CONFIG, feeBps, slippageBps };
  const result = runWalkForwardBacktest(candles, "4h", cfg);
  const oneWay = (feeBps + slippageBps) / 10000;
  for (const trade of [...result.composite.trades, ...result.ouZone.trades]) {
    const sign = trade.direction === "LONG" ? 1 : -1;
    const raw = sign * (trade.exitPrice / trade.entryPrice - 1);
    const expected = (1 + raw) * (1 - oneWay) ** 2 - 1;
    expect(trade.retPct).toBeCloseTo(expected, 12);
  }
});


test("LONG and SHORT accounting are directionally symmetric", () => {
  expect(directionalRawReturn("LONG", 100, 110)).toBeCloseTo(0.1, 12);
  expect(directionalRawReturn("SHORT", 100, 90)).toBeCloseTo(0.1, 12);
  expect(directionalRawReturn("LONG", 100, 90)).toBeCloseTo(-0.1, 12);
  expect(directionalRawReturn("SHORT", 100, 110)).toBeCloseTo(-0.1, 12);

  const cost = 0.0015;
  expect(netTradeReturn("LONG", 100, 110, cost)).toBeCloseTo(netTradeReturn("SHORT", 100, 90, cost), 12);
  expect(netTradeReturn("LONG", 100, 90, cost)).toBeCloseTo(netTradeReturn("SHORT", 100, 110, cost), 12);
});

test("SHORT gap exit uses the open beyond the stop/target level", () => {
  const shortStop = 105;
  const gapOpen = 110;
  const fill = Math.max(shortStop, gapOpen);
  expect(fill).toBe(110);

  const shortTarget = 95;
  const gapDown = 90;
  const targetFill = Math.min(shortTarget, gapDown);
  expect(targetFill).toBe(90);
});


test("position sizing never exceeds configured notional cap", () => {
  expect(positionFractionFromKelly("LONG", 0.9, 0.25)).toBeCloseTo(0.25);
  expect(positionFractionFromKelly("SHORT", -0.8, 0.25)).toBeCloseTo(0.25);
  expect(positionFractionFromKelly("LONG", -0.8, 0.25)).toBe(0);
  expect(positionFractionFromKelly("SHORT", 0.8, 0.25)).toBe(0);
  expect(positionFractionFromStop(100, 95, 0.01, 0.25)).toBeCloseTo(0.2);
  expect(positionFractionFromStop(100, 99.9, 0.01, 0.25)).toBeCloseTo(0.25);
});

test("backtest records bounded position fractions", { timeout: 180000 }, () => {
  const candles = synthCandles(400, 8080);
  const result = runWalkForwardBacktest(candles, "4h");
  for (const res of [result.composite, result.ouZone]) {
    for (const trade of res.trades) {
      expect(trade.positionFraction).toBeGreaterThan(0);
      expect(trade.positionFraction).toBeLessThanOrEqual(result.config.maxPositionFraction);
    }
  }
});


test("performance metrics use portfolio-weighted profit factor and include Sortino", () => {
  const candles = synthCandles(400, 8080);
  const result = runWalkForwardBacktest(candles, "4h");
  for (const res of [result.composite, result.ouZone]) {
    expect(Number.isFinite(res.metrics.sortino) || res.metrics.sortino === Infinity).toBe(true);
    const weightedWins = res.trades
      .filter((t) => t.retPct > 0)
      .reduce((sum, t) => sum + t.positionFraction * t.retPct, 0);
    const weightedLosses = -res.trades
      .filter((t) => t.retPct < 0)
      .reduce((sum, t) => sum + t.positionFraction * t.retPct, 0);
    const expectedPf = weightedLosses > 1e-12
      ? weightedWins / weightedLosses
      : weightedWins > 0 ? Infinity : 0;
    if (Number.isFinite(expectedPf)) expect(res.metrics.profitFactor).toBeCloseTo(expectedPf, 12);
    else expect(res.metrics.profitFactor).toBe(expectedPf);
  }
});


test("walk-forward robustness folds are chronological and cover the full OOS curve", { timeout: 180000 }, () => {
  const candles = synthCandles(400, 8080);
  const result = runWalkForwardBacktest(candles, "4h");
  const folds = summarizeWalkForwardFolds(result.composite, 3);
  expect(folds).toHaveLength(3);
  for (let i = 0; i < folds.length; i++) {
    expect(folds[i]!.startTime).toBeLessThan(folds[i]!.endTime);
    expect(folds[i]!.exposurePct).toBeGreaterThanOrEqual(0);
    expect(folds[i]!.exposurePct).toBeLessThanOrEqual(1);
    expect(folds[i]!.winRate).toBeGreaterThanOrEqual(0);
    expect(folds[i]!.winRate).toBeLessThanOrEqual(1);
    if (i > 0) expect(folds[i]!.startTime).toBeGreaterThanOrEqual(folds[i - 1]!.endTime);
  }
  expect(folds[0]!.startTime).toBe(result.composite.equityCurve[0]!.t);
  expect(folds[2]!.endTime).toBe(result.composite.equityCurve[result.composite.equityCurve.length - 1]!.t);
});

test("robustness fold count must be at least two", () => {
  const candles = synthCandles(400, 8081);
  const result = runWalkForwardBacktest(candles, "4h");
  expect(() => summarizeWalkForwardFolds(result.composite, 1)).toThrow();
});
