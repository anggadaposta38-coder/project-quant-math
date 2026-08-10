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
import { computeSignal } from "./analysis";

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
}

export const DEFAULT_BACKTEST_CONFIG: BacktestConfig = {
  warmupBars: 150,
  refitInterval: 20,
  riskFree: 0.04,
  feeBps: 5,
  slippageBps: 2,
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
  /** Return net biaya, sebagai fraksi (0.01 = +1%). */
  retPct: number;
}

export interface BacktestMetrics {
  bars: number;
  totalReturn: number;
  cagr: number;
  annualizedVol: number;
  sharpe: number;
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

const roundTripCost = (cfg: BacktestConfig) => (cfg.feeBps + cfg.slippageBps) / 10000;

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
  for (let i = 1; i < n; i++) barRets.push(equity[i]!.equity / equity[i - 1]!.equity - 1);

  const totalReturn = n > 0 ? equity[n - 1]!.equity / equity[0]!.equity - 1 : 0;
  const years = bars / barsPerYear;
  const cagr = years > 0 && equity[0]!.equity > 0
    ? Math.pow(equity[n - 1]!.equity / equity[0]!.equity, 1 / years) - 1
    : 0;

  const m = barRets.reduce((a, b) => a + b, 0) / (barRets.length || 1);
  const variance =
    barRets.reduce((a, b) => a + (b - m) ** 2, 0) / (Math.max(barRets.length - 1, 1));
  const sdBar = Math.sqrt(Math.max(variance, 0));
  const annualizedVol = sdBar * Math.sqrt(barsPerYear);
  const rfPerBar = riskFree / barsPerYear;
  const sharpe =
    sdBar > 0 ? ((m - rfPerBar) / sdBar) * Math.sqrt(barsPerYear) : 0;

  const wins = trades.filter((t) => t.retPct > 0);
  const losses = trades.filter((t) => t.retPct <= 0);
  const grossWin = wins.reduce((a, t) => a + t.retPct, 0);
  const grossLoss = -losses.reduce((a, t) => a + t.retPct, 0);
  const winRate = trades.length > 0 ? wins.length / trades.length : 0;
  const profitFactor = grossLoss > 1e-12 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0;
  const avgTradeReturn =
    trades.length > 0 ? trades.reduce((a, t) => a + t.retPct, 0) / trades.length : 0;

  const buyHoldReturn =
    closes.length > 1 ? closes[closes.length - 1]! / closes[0]! - 1 : 0;

  return {
    bars,
    totalReturn,
    cagr,
    annualizedVol,
    sharpe,
    maxDrawdown: maxDrawdown(equity.map((e) => e.equity)),
    numTrades: trades.length,
    winRate,
    profitFactor,
    avgTradeReturn,
    exposurePct: bars > 0 ? exposedBars / bars : 0,
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
): BacktestResult {
  const closes = candles.map((c) => c.c);
  const times = candles.map((c) => c.t);
  const n = candles.length;
  const checkpoints = buildCheckpoints(n, cfg);
  const cost = roundTripCost(cfg);

  const equity: { t: number; equity: number }[] = [{ t: times[cfg.warmupBars]!, equity: 1 }];
  const trades: BacktestTrade[] = [];
  let eq = 1;
  let dir: -1 | 0 | 1 = 0;
  let openTrade: { direction: "LONG" | "SHORT"; entryIndex: number; entryPrice: number } | null =
    null;
  let exposedBars = 0;
  let barsPerYear = 365;

  const closePosition = (exitIndex: number, reason: BacktestTrade["reason"]) => {
    if (!openTrade) return;
    const exitPrice = closes[exitIndex]!;
    const sign = openTrade.direction === "LONG" ? 1 : -1;
    const raw = sign * (exitPrice / openTrade.entryPrice - 1);
    const net = raw - cost; // fee dikenakan sekali per round-trip (entry+exit gabungan)
    trades.push({
      direction: openTrade.direction,
      entryIndex: openTrade.entryIndex,
      entryTime: times[openTrade.entryIndex]!,
      entryPrice: openTrade.entryPrice,
      exitIndex,
      exitTime: times[exitIndex]!,
      exitPrice,
      reason,
      retPct: net,
    });
    openTrade = null;
  };

  for (let k = 0; k < checkpoints.length; k++) {
    const i = checkpoints[k]!;
    const segEnd = Math.min(i + cfg.refitInterval, n - 1);
    const signal = computeSignal(candles.slice(0, i + 1), interval, cfg.riskFree);
    barsPerYear = signal.barsPerYear;
    const newDir: -1 | 0 | 1 = signal.action === "LONG" ? 1 : signal.action === "SHORT" ? -1 : 0;

    if (newDir !== dir) {
      // Tutup posisi lama (jika ada) tepat di harga checkpoint (close[i]) —
      // separuh biaya round-trip untuk leg penutupan.
      if (openTrade) {
        eq *= 1 - cost / 2;
        closePosition(i, "signal-flip");
      }
      if (newDir !== 0) {
        eq *= 1 - cost / 2; // separuh biaya round-trip untuk leg pembukaan
        openTrade = { direction: newDir === 1 ? "LONG" : "SHORT", entryIndex: i, entryPrice: closes[i]! };
      }
      dir = newDir;
    }

    for (let j = i + 1; j <= segEnd; j++) {
      const barLogRet = Math.log(closes[j]! / closes[j - 1]!);
      eq *= Math.exp(dir * barLogRet);
      if (dir !== 0) exposedBars++;
      equity.push({ t: times[j]!, equity: eq });
    }
  }

  if (openTrade) {
    const lastIdx = n - 1;
    eq *= 1 - cost / 2; // separuh biaya round-trip saat menutup di akhir data
    closePosition(lastIdx, "end-of-data");
    if (equity[equity.length - 1]!.t === times[lastIdx]) {
      equity[equity.length - 1]!.equity = eq;
    } else {
      equity.push({ t: times[lastIdx]!, equity: eq });
    }
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
): BacktestResult {
  const closes = candles.map((c) => c.c);
  const highs = candles.map((c) => c.h);
  const lows = candles.map((c) => c.l);
  const times = candles.map((c) => c.t);
  const n = candles.length;
  const checkpoints = buildCheckpoints(n, cfg);
  const cost = roundTripCost(cfg);

  const equity: { t: number; equity: number }[] = [{ t: times[cfg.warmupBars]!, equity: 1 }];
  const trades: BacktestTrade[] = [];
  let eq = 1;
  let exposedBars = 0;
  let barsPerYear = 365;
  let cursor = cfg.warmupBars; // bar terakhir yang sudah diproses ke equity curve
  let cpIdx = 0;

  while (cpIdx < checkpoints.length) {
    const i = checkpoints[cpIdx]!;
    if (i < cursor) {
      cpIdx++;
      continue;
    }
    const signal = computeSignal(candles.slice(0, i + 1), interval, cfg.riskFree);
    barsPerYear = signal.barsPerYear;
    const { longZone, shortZone } = signal;

    // Bawa equity curve datar (posisi flat) sampai bar i (checkpoint saat ini).
    for (let j = cursor + 1; j <= i; j++) equity.push({ t: times[j]!, equity: eq });
    cursor = i;

    if (!longZone && !shortZone) {
      cpIdx++;
      continue;
    }

    // Cari entry pertama yang tersentuh setelah checkpoint ini, sebelum checkpoint berikutnya.
    const nextCp = checkpoints[cpIdx + 1] ?? n - 1;
    let entryIndex = -1;
    let direction: "LONG" | "SHORT" | null = null;
    for (let j = i + 1; j <= Math.min(nextCp, n - 1); j++) {
      const hitLong = longZone && lows[j]! <= longZone.entry;
      const hitShort = shortZone && highs[j]! >= shortZone.entry;
      if (hitLong && hitShort) {
        // Kedua sisi tersentuh di bar sama — ambil yang levelnya lebih dekat ke close checkpoint (lebih mungkin duluan).
        const distLong = Math.abs(closes[i]! - longZone.entry);
        const distShort = Math.abs(shortZone.entry - closes[i]!);
        direction = distLong <= distShort ? "LONG" : "SHORT";
        entryIndex = j;
        break;
      } else if (hitLong) {
        direction = "LONG";
        entryIndex = j;
        break;
      } else if (hitShort) {
        direction = "SHORT";
        entryIndex = j;
        break;
      }
      // bar netral: tetap flat, equity tidak bergerak sampai entry tersentuh
      equity.push({ t: times[j]!, equity: eq });
      cursor = j;
    }

    if (entryIndex < 0 || !direction || entryIndex >= n - 1) {
      // Tidak ada entry, atau entry jatuh tepat di bar terakhir (tak ada bar
      // tersisa untuk mark-to-market/exit) — lewati kesempatan ini.
      cpIdx++;
      continue;
    }

    const zone = direction === "LONG" ? longZone! : shortZone!;
    const entryPrice = zone.entry;
    const sign = direction === "LONG" ? 1 : -1;
    eq *= 1 - cost / 2;
    equity.push({ t: times[entryIndex]!, equity: eq });
    cursor = entryIndex;

    // Pantau bar berikutnya sampai target atau stop tersentuh (boleh melewati
    // checkpoint berikutnya). `markPrice` = harga terakhir yang sudah
    // ter-refleksi di `eq`, dimulai dari harga fill entry (zone.entry), agar
    // tidak ada selisih ganda antara harga fill dan close bar entry.
    let exitIndex = -1;
    let exitPrice = entryPrice;
    let reason: BacktestTrade["reason"] = "end-of-data";
    let markPrice = entryPrice;
    for (let j = entryIndex + 1; j < n; j++) {
      const hitStop = direction === "LONG" ? lows[j]! <= zone.stop : highs[j]! >= zone.stop;
      const hitTarget = direction === "LONG" ? highs[j]! >= zone.target : lows[j]! <= zone.target;
      if (hitStop || hitTarget) {
        exitIndex = j;
        exitPrice = hitStop ? zone.stop : zone.target;
        reason = hitStop ? "stop" : "target";
        eq *= Math.exp(sign * Math.log(exitPrice / markPrice));
        eq *= 1 - cost / 2; // biaya keluar
        exposedBars++;
        equity.push({ t: times[j]!, equity: eq });
        cursor = j;
        break;
      }
      eq *= Math.exp(sign * Math.log(closes[j]! / markPrice));
      markPrice = closes[j]!;
      exposedBars++;
      equity.push({ t: times[j]!, equity: eq });
      cursor = j;
    }

    if (exitIndex < 0) {
      // Posisi belum exit sampai akhir data: tutup paksa di harga terakhir
      // (yang sudah ter-refleksi di eq/markPrice dari loop di atas) — hanya
      // kenakan biaya keluar, jangan hitung ulang pergerakan harga bar itu.
      exitIndex = n - 1;
      exitPrice = markPrice;
      eq *= 1 - cost / 2;
      equity[equity.length - 1]!.equity = eq;
    }

    const raw = sign * (exitPrice / entryPrice - 1);
    trades.push({
      direction,
      entryIndex,
      entryTime: times[entryIndex]!,
      entryPrice,
      exitIndex,
      exitTime: times[exitIndex]!,
      exitPrice,
      reason,
      retPct: raw - cost,
    });

    // Lanjut ke checkpoint pertama setelah posisi ini ditutup.
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
  if (candles.length < cfg.warmupBars + cfg.refitInterval + 5) {
    throw new Error(
      `Data tidak cukup untuk backtest: butuh minimal ${cfg.warmupBars + cfg.refitInterval + 5} bar, tersedia ${candles.length}.`,
    );
  }
  return {
    composite: runCompositeBacktest(candles, interval, cfg),
    ouZone: runOuZoneBacktest(candles, interval, cfg),
    config: cfg,
  };
}
