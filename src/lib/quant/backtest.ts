/**
 * Backtest walk-forward untuk dua strategi yang dihasilkan `computeSignal`:
 *
 *  1. "composite"  — ikuti action (LONG/SHORT/WAIT) dari skor komposit,
 *                     posisi diganti hanya di tiap checkpoint refit.
 *  2. "ou-zone"    — order limit di entry zone mean-reversion (OU), exit di
 *                     target atau stop-loss (bukan menunggu checkpoint).
 *
 * Prinsip anti-lookahead:
 *   Di checkpoint index i, sinyal dihitung HANYA dari candles[0..i]
 *   (computeSignal internally juga sudah begitu). Sinyal itu lalu dipakai
 *   untuk memutuskan posisi pada bar i+1 dan seterusnya — never the reverse.
 *   OU zone entry diasumsikan terisi persis di harga level (limit order),
 *   dan bila stop & target sama-sama tersentuh pada bar OHLC yang sama,
 *   diasumsikan stop tersentuh lebih dulu (konservatif / worst-case).
 *
 * Karena re-fit HMM (Baum-Welch) mahal, model hanya di-refit tiap
 * `refitInterval` bar (bukan tiap bar) — sesuai praktik walk-forward umum.
 */

import type { Candle, Interval } from "@/lib/market.server";
import { computeSignal, MIN_SIGNAL_BARS, type SignalState } from "./analysis";

export type StrategyId = "composite" | "ou-zone";

export interface BacktestConfig {
  /** Jumlah bar minimum sebelum checkpoint pertama (agar HMM/OU stabil). */
  warmupBars: number;
  /** Jarak antar refit model (bar). */
  refitInterval: number;
  riskFree: number;
  /** Biaya trading per sisi (bps), diterapkan saat entry & exit. */
  feeBps: number;
  /** Slippage per sisi (bps). */
  slippageBps: number;
  /** Risiko maksimum per trade sebagai fraksi equity, digunakan OU-zone. */
  riskPerTradePct: number;
  /** Batas notional sebagai fraksi equity; mencegah leverage tersembunyi. */
  maxPositionFraction: number;
}

export const DEFAULT_BACKTEST_CONFIG: BacktestConfig = {
  warmupBars: 150,
  refitInterval: 20,
  riskFree: 0.04,
  feeBps: 5,
  slippageBps: 2,
  riskPerTradePct: 0.01,
  maxPositionFraction: 0.25,
};

export interface BacktestTrade {
  direction: "LONG" | "SHORT";
  entryIndex: number;
  entryTime: number;
  entryPrice: number;
  exitIndex: number;
  exitTime: number;
  exitPrice: number;
  reason: "signal-flip" | "target" | "stop" | "end-of-data";
  /** Return net biaya, sebagai fraksi (0.01 = +1%) atas notional posisi. */
  retPct: number;
  /** Notional posisi sebagai fraksi equity saat entry (tanpa leverage). */
  positionFraction: number;
}

export interface BacktestMetrics {
  bars: number;
  totalReturn: number;
  cagr: number;
  annualizedVol: number;
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  numTrades: number;
  winRate: number;
  profitFactor: number;
  avgTradeReturn: number;
  exposurePct: number;
  buyHoldReturn: number;
}

export interface BacktestResult {
  strategy: StrategyId;
  equityCurve: { t: number; equity: number }[];
  trades: BacktestTrade[];
  metrics: BacktestMetrics;
}

const oneWayCost = (cfg: BacktestConfig) => (cfg.feeBps + cfg.slippageBps) / 10000;

/**
 * Ukuran posisi dari fractional Kelly, searah dengan signal. Nilai negatif
 * Kelly tidak boleh diam-diam mengubah LONG menjadi SHORT; ia berarti tidak
 * ada alokasi untuk sisi tersebut. Hasil dibatasi maxPositionFraction agar
 * dashboard/backtest tidak pernah mengasumsikan leverage tersembunyi.
 */
export function positionFractionFromKelly(
  direction: "LONG" | "SHORT",
  kellyFraction: number,
  maxPositionFraction: number,
): number {
  if (!Number.isFinite(kellyFraction) || !Number.isFinite(maxPositionFraction) || maxPositionFraction < 0 || maxPositionFraction > 1) {
    throw new Error("Parameter Kelly/position limit tidak valid.");
  }
  const aligned = direction === "LONG" ? kellyFraction : -kellyFraction;
  return Math.min(Math.max(aligned, 0), maxPositionFraction);
}

/**
 * Ukuran posisi berbasis risiko stop: fraction = riskBudget / stopDistance.
 * Selalu dibatasi maxPositionFraction dan tidak pernah menggunakan leverage.
 */
export function positionFractionFromStop(
  entryPrice: number,
  stopPrice: number,
  riskPerTradePct: number,
  maxPositionFraction: number,
): number {
  if (![entryPrice, stopPrice, riskPerTradePct, maxPositionFraction].every(Number.isFinite) || entryPrice <= 0 || stopPrice <= 0) {
    throw new Error("Parameter stop sizing tidak valid.");
  }
  if (riskPerTradePct < 0 || maxPositionFraction < 0 || maxPositionFraction > 1) {
    throw new Error("Risk/position limit tidak valid.");
  }
  const stopDistance = Math.abs(entryPrice - stopPrice) / entryPrice;
  if (stopDistance <= 0 || riskPerTradePct <= 0) return 0;
  return Math.min(riskPerTradePct / stopDistance, maxPositionFraction);
}

/** Return sederhana posisi dari entry ke exit, simetris untuk LONG/SHORT. */
export function directionalRawReturn(
  direction: "LONG" | "SHORT",
  entryPrice: number,
  exitPrice: number,
): number {
  if (!(entryPrice > 0) || !(exitPrice > 0) || !Number.isFinite(entryPrice) || !Number.isFinite(exitPrice)) {
    throw new Error("Harga entry/exit harus finite dan > 0.");
  }
  const sign = direction === "LONG" ? 1 : -1;
  return sign * (exitPrice / entryPrice - 1);
}

/** Return trade setelah biaya satu sisi saat entry dan satu sisi saat exit. */
export function netTradeReturn(
  direction: "LONG" | "SHORT",
  entryPrice: number,
  exitPrice: number,
  oneWayCostRate: number,
): number {
  if (!Number.isFinite(oneWayCostRate) || oneWayCostRate < 0 || oneWayCostRate >= 1) {
    throw new Error("oneWayCostRate harus finite dan berada pada [0, 1).");
  }
  const raw = directionalRawReturn(direction, entryPrice, exitPrice);
  // Both entry and exit costs are charged against the entry notional.
  // The exit notional is entryNotional * (1 + raw), so the net return is:
  // raw - entryCost - exitCost = (1 + raw) * (1 - c) - 1 - c.
  return (1 + raw) * (1 - oneWayCostRate) - 1 - oneWayCostRate;
}

/**
 * Future-data firewall for walk-forward fitting. `checkpoint` is the last
 * candle whose CLOSE is known when the model is fitted. Nothing after that
 * index is passed into any estimator (HMM, GBM, OU, RSI/MACD, etc.).
 */
function signalAtCheckpoint(
  candles: Candle[],
  checkpoint: number,
  interval: Interval,
  riskFree: number,
) {
  if (!Number.isInteger(checkpoint) || checkpoint < 0 || checkpoint >= candles.length) {
    throw new Error("checkpoint di luar rentang candles.");
  }
  return computeSignal(candles.slice(0, checkpoint + 1), interval, riskFree);
}

/**
 * Compute each checkpoint signal once. Composite and OU strategies share the
 * same walk-forward checkpoints, so refitting HMM/GBM/OU twice per checkpoint
 * is pure duplicate work and does not change the model result.
 */
function buildCheckpointSignals(
  candles: Candle[],
  checkpoints: number[],
  interval: Interval,
  riskFree: number,
): CheckpointSignals {
  const signals: CheckpointSignals = new Map();
  for (const checkpoint of checkpoints) {
    signals.set(checkpoint, signalAtCheckpoint(candles, checkpoint, interval, riskFree));
  }
  return signals;
}

type CheckpointSignals = Map<number, SignalState>;

function buildCheckpoints(n: number, cfg: BacktestConfig): number[] {
  const points: number[] = [];
  for (let i = cfg.warmupBars; i < n - 1; i += cfg.refitInterval) points.push(i);
  return points;
}

function maxDrawdown(equity: number[]): number {
  let peak = equity[0] ?? 1;
  let mdd = 0;
  for (const e of equity) {
    peak = Math.max(peak, e);
    mdd = Math.min(mdd, e / peak - 1);
  }
  return mdd;
}

function metricsFromCurve(
  equity: { t: number; equity: number }[],
  trades: BacktestTrade[],
  barsPerYear: number,
  riskFree: number,
  exposedBars: number,
  closes: number[],
): BacktestMetrics {
  const n = equity.length;
  const bars = Math.max(n - 1, 0);
  const barRets: number[] = [];
  for (let i = 1; i < n; i++) {
    const prev = equity[i - 1]!.equity;
    const curr = equity[i]!.equity;
    if (!(prev > 0) || !Number.isFinite(prev) || !Number.isFinite(curr) || curr <= 0) {
      throw new Error("Equity curve tidak valid untuk perhitungan metrik.");
    }
    barRets.push(curr / prev - 1);
  }

  const initialEquity = equity[0]?.equity ?? 1;
  const finalEquity = equity[n - 1]?.equity ?? initialEquity;
  const totalReturn = n > 0 && initialEquity > 0 ? finalEquity / initialEquity - 1 : 0;

  // Gunakan waktu aktual untuk CAGR. Ini tetap sama dengan bar-count pada
  // data reguler, tetapi tidak menyembunyikan kesalahan jika future caller
  // memberi equity curve dengan interval berbeda.
  const elapsedYears = n > 1
    ? Math.max((equity[n - 1]!.t - equity[0]!.t) / (365 * 24 * 60 * 60 * 1000), 0)
    : 0;
  const cagr = elapsedYears > 0 && initialEquity > 0 && finalEquity > 0
    ? Math.pow(finalEquity / initialEquity, 1 / elapsedYears) - 1
    : 0;

  const mean = barRets.length > 0
    ? barRets.reduce((a, b) => a + b, 0) / barRets.length
    : 0;
  const variance = barRets.length > 1
    ? barRets.reduce((a, b) => a + (b - mean) ** 2, 0) / (barRets.length - 1)
    : 0;
  const sdBar = Math.sqrt(Math.max(variance, 0));
  const annualizedVol = sdBar * Math.sqrt(barsPerYear);
  const rfPerBar = Math.pow(1 + riskFree, 1 / barsPerYear) - 1;
  const excessMean = mean - rfPerBar;
  const sharpe = sdBar > 0 ? (excessMean / sdBar) * Math.sqrt(barsPerYear) : 0;

  // Sortino memakai downside deviation terhadap risk-free hurdle per bar.
  // Semua bar ikut denominator; hanya downside excess yang berkontribusi.
  const downsideMeanSquare = barRets.length > 0
    ? barRets.reduce((sum, r) => {
        const downside = Math.min(r - rfPerBar, 0);
        return sum + downside * downside;
      }, 0) / barRets.length
    : 0;
  const downsideDevBar = Math.sqrt(Math.max(downsideMeanSquare, 0));
  const sortino = downsideDevBar > 0
    ? (excessMean / downsideDevBar) * Math.sqrt(barsPerYear)
    : excessMean > 0 ? Infinity : 0;

  // Profit factor seharusnya berbasis PnL portfolio, bukan sekadar menjumlah
  // return atas notional. Position size berbeda antar trade, sehingga trade
  // yang 5% return pada 5% equity tidak boleh diberi bobot sama dengan trade
  // 5% return pada 25% equity.
  const weightedPnls = trades.map((t) => t.positionFraction * t.retPct);
  const grossWin = weightedPnls.filter((r) => r > 0).reduce((a, r) => a + r, 0);
  const grossLoss = -weightedPnls.filter((r) => r < 0).reduce((a, r) => a + r, 0);
  const wins = trades.filter((t) => t.retPct > 0);
  const winRate = trades.length > 0 ? wins.length / trades.length : 0;
  const profitFactor = grossLoss > 1e-12 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0;
  const avgTradeReturn = trades.length > 0
    ? trades.reduce((a, t) => a + t.retPct, 0) / trades.length
    : 0;

  const buyHoldReturn = closes.length > 1
    ? closes[closes.length - 1]! / closes[0]! - 1
    : 0;

  return {
    bars,
    totalReturn,
    cagr,
    annualizedVol,
    sharpe,
    sortino,
    maxDrawdown: maxDrawdown(equity.map((e) => e.equity)),
    numTrades: trades.length,
    winRate,
    profitFactor,
    avgTradeReturn,
    exposurePct: bars > 0 ? Math.min(Math.max(exposedBars / bars, 0), 1) : 0,
    buyHoldReturn,
  };
}

// ---------------------------------------------------------------------------
// Strategi 1: ikuti action dari skor komposit, rebalance tiap checkpoint.
// ---------------------------------------------------------------------------

function runCompositeBacktest(
  candles: Candle[],
  interval: Interval,
  cfg: BacktestConfig,
  checkpointSignals: CheckpointSignals,
): BacktestResult {
  const closes = candles.map((c) => c.c);
  const times = candles.map((c) => c.t);
  const n = candles.length;
  const checkpoints = buildCheckpoints(n, cfg);
  const transactionCost = oneWayCost(cfg);

  const equity: { t: number; equity: number }[] = [{ t: times[cfg.warmupBars]!, equity: 1 }];
  const trades: BacktestTrade[] = [];
  let eq = 1;
  let dir: -1 | 0 | 1 = 0;
  let openTrade: {
    direction: "LONG" | "SHORT";
    entryIndex: number;
    entryPrice: number;
    positionFraction: number;
    entryEquity: number;
    notional: number;
  } | null = null;
  let exposedBars = 0;
  let barsPerYear = 365;
  let lastProcessed = cfg.warmupBars;

  const markOpenTrade = (price: number) => {
    if (!openTrade) return eq;
    const raw = directionalRawReturn(openTrade.direction, openTrade.entryPrice, price);
    // Fixed notional: positionFraction is defined against equity at entry,
    // not continuously rebalanced against current equity.
    return openTrade.entryEquity - openTrade.notional * transactionCost + openTrade.notional * raw;
  };

  const closePosition = (exitIndex: number, reason: BacktestTrade["reason"]) => {
    if (!openTrade) return;
    const exitPrice = closes[exitIndex]!;
    eq = markOpenTrade(exitPrice);
    const exitNotional = openTrade.notional * (1 + directionalRawReturn(openTrade.direction, openTrade.entryPrice, exitPrice));
    eq -= exitNotional * transactionCost;
    trades.push({
      direction: openTrade.direction,
      entryIndex: openTrade.entryIndex,
      entryTime: times[openTrade.entryIndex]!,
      entryPrice: openTrade.entryPrice,
      exitIndex,
      exitTime: times[exitIndex]!,
      exitPrice,
      reason,
      retPct: netTradeReturn(openTrade.direction, openTrade.entryPrice, exitPrice, transactionCost),
      positionFraction: openTrade.positionFraction,
    });
    openTrade = null;
  };

  for (const i of checkpoints) {
    const executionIndex = i + 1;
    if (executionIndex >= n) break;

    const signal = checkpointSignals.get(i);
    if (!signal) throw new Error(`Signal checkpoint ${i} tidak tersedia.`);
    barsPerYear = signal.barsPerYear;
    const newDir: -1 | 0 | 1 = signal.action === "LONG" ? 1 : signal.action === "SHORT" ? -1 : 0;

    // Signal hanya diketahui setelah close[i]. Eksekusi dipindahkan ke close[i+1].
    for (let j = lastProcessed + 1; j <= executionIndex; j++) {
      if (openTrade) {
        eq = markOpenTrade(closes[j]!);
        exposedBars++;
      }
      equity.push({ t: times[j]!, equity: eq });
    }

    if (newDir !== dir) {
      if (openTrade) {
        closePosition(executionIndex, "signal-flip");
      }
      if (newDir !== 0) {
        const tradeDirection = newDir === 1 ? "LONG" : "SHORT";
        const positionFraction = positionFractionFromKelly(tradeDirection, signal.kellyFraction, cfg.maxPositionFraction);
        if (positionFraction > 0) {
          const entryEquity = eq;
          const notional = entryEquity * positionFraction;
          eq = entryEquity - notional * transactionCost;
          openTrade = {
            direction: tradeDirection,
            entryIndex: executionIndex,
            entryPrice: closes[executionIndex]!,
            positionFraction,
            entryEquity,
            notional,
          };
          dir = newDir;
        } else {
          openTrade = null;
          dir = 0;
        }
      } else {
        dir = 0;
      }
      equity[equity.length - 1]!.equity = eq;
    }

    const segEnd = Math.min(i + cfg.refitInterval, n - 1);
    for (let j = executionIndex + 1; j <= segEnd; j++) {
      if (openTrade) {
        eq = markOpenTrade(closes[j]!);
        exposedBars++;
      }
      equity.push({ t: times[j]!, equity: eq });
    }
    lastProcessed = Math.max(lastProcessed, segEnd);
  }

  if (openTrade) {
    const lastIdx = n - 1;
    if (lastProcessed < lastIdx) {
      for (let j = lastProcessed + 1; j <= lastIdx; j++) {
        eq = markOpenTrade(closes[j]!);
        exposedBars++;
        equity.push({ t: times[j]!, equity: eq });
      }
    }
    closePosition(lastIdx, "end-of-data");
    equity[equity.length - 1]!.equity = eq;
  } else if (lastProcessed < n - 1) {
    for (let j = lastProcessed + 1; j < n; j++) equity.push({ t: times[j]!, equity: eq });
  }

  const usedCloses = closes.slice(cfg.warmupBars);
  return {
    strategy: "composite",
    equityCurve: equity,
    trades,
    metrics: metricsFromCurve(equity, trades, barsPerYear, cfg.riskFree, exposedBars, usedCloses),
  };
}

// ---------------------------------------------------------------------------
// Strategi 2: order limit di zona entry OU, exit di target/stop.
// ---------------------------------------------------------------------------

function runOuZoneBacktest(
  candles: Candle[],
  interval: Interval,
  cfg: BacktestConfig,
  checkpointSignals: CheckpointSignals,
): BacktestResult {
  const closes = candles.map((c) => c.c);
  const highs = candles.map((c) => c.h);
  const lows = candles.map((c) => c.l);
  const times = candles.map((c) => c.t);
  const n = candles.length;
  const checkpoints = buildCheckpoints(n, cfg);
  const transactionCost = oneWayCost(cfg);

  const equity: { t: number; equity: number }[] = [{ t: times[cfg.warmupBars]!, equity: 1 }];
  const trades: BacktestTrade[] = [];
  let eq = 1;
  let exposedBars = 0;
  let barsPerYear = 365;
  let cursor = cfg.warmupBars;
  let cpIdx = 0;

  while (cpIdx < checkpoints.length) {
    const i = checkpoints[cpIdx]!;
    if (i < cursor) {
      cpIdx++;
      continue;
    }
    const signal = checkpointSignals.get(i);
    if (!signal) throw new Error(`Signal checkpoint ${i} tidak tersedia.`);
    barsPerYear = signal.barsPerYear;
    const { longZone, shortZone } = signal;

    for (let j = cursor + 1; j <= i; j++) equity.push({ t: times[j]!, equity: eq });
    cursor = i;

    if (!longZone && !shortZone) {
      cpIdx++;
      continue;
    }

    const nextCp = checkpoints[cpIdx + 1] ?? n - 1;
    let entryIndex = -1;
    let direction: "LONG" | "SHORT" | null = null;
    for (let j = i + 1; j <= Math.min(nextCp, n - 1); j++) {
      const hitLong = !!longZone && lows[j]! <= longZone.entry;
      const hitShort = !!shortZone && highs[j]! >= shortZone.entry;
      if (hitLong && hitShort) {
        equity.push({ t: times[j]!, equity: eq });
        cursor = j;
        continue;
      } else if (hitLong) {
        direction = "LONG";
        entryIndex = j;
        break;
      } else if (hitShort) {
        direction = "SHORT";
        entryIndex = j;
        break;
      }
      equity.push({ t: times[j]!, equity: eq });
      cursor = j;
    }

    if (entryIndex < 0 || !direction || entryIndex >= n - 1) {
      cpIdx++;
      continue;
    }

    const zone = direction === "LONG" ? longZone! : shortZone!;
    const entryPrice = zone.entry;
    const positionFraction = positionFractionFromStop(entryPrice, zone.stop, cfg.riskPerTradePct, cfg.maxPositionFraction);
    if (positionFraction <= 0) {
      cpIdx++;
      continue;
    }

    const entryEquity = eq;
    const notional = entryEquity * positionFraction;
    eq = entryEquity - notional * transactionCost;
    equity.push({ t: times[entryIndex]!, equity: eq });
    cursor = entryIndex;

    let exitIndex = -1;
    let exitPrice = entryPrice;
    let reason: BacktestTrade["reason"] = "end-of-data";

    const markAt = (price: number) => {
      const raw = directionalRawReturn(direction!, entryPrice, price);
      return entryEquity - notional * transactionCost + notional * raw;
    };

    for (let j = entryIndex + 1; j < n; j++) {
      const hitStop = direction === "LONG" ? lows[j]! <= zone.stop : highs[j]! >= zone.stop;
      const hitTarget = direction === "LONG" ? highs[j]! >= zone.target : lows[j]! <= zone.target;
      if (hitStop || hitTarget) {
        exitIndex = j;
        const open = candles[j]!.o;
        if (hitStop) {
          exitPrice = direction === "LONG" ? Math.min(zone.stop, open) : Math.max(zone.stop, open);
          reason = "stop";
        } else {
          exitPrice = direction === "LONG" ? Math.max(zone.target, open) : Math.min(zone.target, open);
          reason = "target";
        }
        eq = markAt(exitPrice);
        const exitNotional = notional * (1 + directionalRawReturn(direction, entryPrice, exitPrice));
        eq -= exitNotional * transactionCost;
        exposedBars++;
        equity.push({ t: times[j]!, equity: eq });
        cursor = j;
        break;
      }
      eq = markAt(closes[j]!);
      exposedBars++;
      equity.push({ t: times[j]!, equity: eq });
      cursor = j;
    }

    if (exitIndex < 0) {
      exitIndex = n - 1;
      exitPrice = closes[n - 1]!;
      eq = markAt(exitPrice);
      const exitNotional = notional * (1 + directionalRawReturn(direction, entryPrice, exitPrice));
      eq -= exitNotional * transactionCost;
      equity[equity.length - 1]!.equity = eq;
    }

    trades.push({
      direction,
      entryIndex,
      entryTime: times[entryIndex]!,
      entryPrice,
      exitIndex,
      exitTime: times[exitIndex]!,
      exitPrice,
      reason,
      retPct: netTradeReturn(direction, entryPrice, exitPrice, transactionCost),
      positionFraction,
    });

    while (cpIdx < checkpoints.length && checkpoints[cpIdx]! <= cursor) cpIdx++;
  }

  for (let j = cursor + 1; j < n; j++) equity.push({ t: times[j]!, equity: eq });

  const usedCloses = closes.slice(cfg.warmupBars);
  return {
    strategy: "ou-zone",
    equityCurve: equity,
    trades,
    metrics: metricsFromCurve(equity, trades, barsPerYear, cfg.riskFree, exposedBars, usedCloses),
  };
}

export interface WalkForwardFoldSummary {
  fold: number;
  startTime: number;
  endTime: number;
  totalReturn: number;
  maxDrawdown: number;
  numTrades: number;
  winRate: number;
  exposurePct: number;
}

/**
 * Robustness diagnostic untuk walk-forward result.
 * Membagi periode OOS yang sudah dihasilkan backtest secara kronologis,
 * tanpa refit ulang atau mengintip ke fold berikutnya. Ini BUKAN pengganti
 * validation set independen; tujuannya mendeteksi apakah performa hanya
 * berasal dari satu sub-periode.
 */
export function summarizeWalkForwardFolds(
  result: BacktestResult,
  folds = 3,
): WalkForwardFoldSummary[] {
  if (!Number.isInteger(folds) || folds < 2) {
    throw new Error("folds harus integer >= 2.");
  }
  const curve = result.equityCurve;
  if (curve.length < folds + 1) {
    throw new Error("Equity curve tidak cukup untuk robustness folds.");
  }

  const summaries: WalkForwardFoldSummary[] = [];
  const totalBars = curve.length - 1;
  for (let f = 0; f < folds; f++) {
    const startBar = Math.floor((f * totalBars) / folds);
    const endBar = f === folds - 1 ? totalBars : Math.floor(((f + 1) * totalBars) / folds);
    const start = curve[startBar]!;
    const end = curve[endBar]!;
    const segment = curve.slice(startBar, endBar + 1);
    const segmentTrades = result.trades.filter(
      (t) => t.exitTime > start.t && t.exitTime <= end.t,
    );
    let peak = segment[0]!.equity;
    let mdd = 0;
    let exposedBars = 0;
    for (let i = 1; i < segment.length; i++) {
      const e = segment[i]!.equity;
      peak = Math.max(peak, e);
      mdd = Math.min(mdd, e / peak - 1);
      const barTime = segment[i]!.t;
      if (segmentTrades.some((t) => t.entryTime < barTime && t.exitTime >= barTime)) exposedBars++;
    }
    const bars = Math.max(segment.length - 1, 1);
    const totalReturn = start.equity > 0 ? end.equity / start.equity - 1 : 0;
    const wins = segmentTrades.filter((t) => t.retPct > 0).length;
    summaries.push({
      fold: f + 1,
      startTime: start.t,
      endTime: end.t,
      totalReturn,
      maxDrawdown: mdd,
      numTrades: segmentTrades.length,
      winRate: segmentTrades.length > 0 ? wins / segmentTrades.length : 0,
      exposurePct: Math.min(Math.max(exposedBars / bars, 0), 1),
    });
  }
  return summaries;
}

export interface BacktestComparison {
  composite: BacktestResult;
  ouZone: BacktestResult;
  config: BacktestConfig;
}

export function runWalkForwardBacktest(
  candles: Candle[],
  interval: Interval,
  config: Partial<BacktestConfig> = {},
): BacktestComparison {
  const cfg: BacktestConfig = { ...DEFAULT_BACKTEST_CONFIG, ...config };

  if (!Number.isInteger(cfg.warmupBars) || cfg.warmupBars < MIN_SIGNAL_BARS) {
    throw new Error(`warmupBars harus integer >= ${MIN_SIGNAL_BARS}.`);
  }
  if (!Number.isInteger(cfg.refitInterval) || cfg.refitInterval < 1) {
    throw new Error("refitInterval harus integer >= 1.");
  }
  if (!Number.isFinite(cfg.riskFree) || cfg.riskFree < -1) {
    throw new Error("riskFree harus finite dan > -100%.");
  }
  if (!Number.isFinite(cfg.feeBps) || cfg.feeBps < 0) {
    throw new Error("feeBps harus finite dan >= 0.");
  }
  if (!Number.isFinite(cfg.slippageBps) || cfg.slippageBps < 0) {
    throw new Error("slippageBps harus finite dan >= 0.");
  }
  if (!Number.isFinite(cfg.riskPerTradePct) || cfg.riskPerTradePct < 0 || cfg.riskPerTradePct > 1) {
    throw new Error("riskPerTradePct harus finite dan berada pada [0, 1].");
  }
  if (!Number.isFinite(cfg.maxPositionFraction) || cfg.maxPositionFraction < 0 || cfg.maxPositionFraction > 1) {
    throw new Error("maxPositionFraction harus finite dan berada pada [0, 1].");
  }
  const intervalMs: Record<Interval, number> = {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
  };
  const expectedStep = intervalMs[interval];

  for (let i = 0; i < candles.length; i++) {
    const c = candles[i]!;
    if (
      !Number.isFinite(c.t) ||
      !Number.isFinite(c.o) ||
      !Number.isFinite(c.h) ||
      !Number.isFinite(c.l) ||
      !Number.isFinite(c.c) ||
      c.o <= 0 ||
      c.h <= 0 ||
      c.l <= 0 ||
      c.c <= 0 ||
      c.h < Math.max(c.o, c.c) ||
      c.l > Math.min(c.o, c.c) ||
      (i > 0 && c.t - candles[i - 1]!.t !== expectedStep)
    ) {
      throw new Error(`Candle tidak valid pada index ${i}.`);
    }
  }

  if (candles.length < cfg.warmupBars + cfg.refitInterval + 5) {
    throw new Error(
      `Data tidak cukup untuk backtest: butuh minimal ${cfg.warmupBars + cfg.refitInterval + 5} bar, tersedia ${candles.length}.`,
    );
  }
  const checkpoints = buildCheckpoints(candles.length, cfg);
  const checkpointSignals = buildCheckpointSignals(candles, checkpoints, interval, cfg.riskFree);

  return {
    composite: runCompositeBacktest(candles, interval, cfg, checkpointSignals),
    ouZone: runOuZoneBacktest(candles, interval, cfg, checkpointSignals),
    config: cfg,
  };
}
