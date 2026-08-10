/** Pipeline analisa: HMM → Stochastic → Aljabar Linear → Timing entry. */

import type { Candle, Interval } from "@/lib/market.server";
import { BARS_PER_YEAR } from "@/lib/market.server";
import { fitHmm, regimeLabel, type HmmFit } from "./hmm";
import {
  computeEntryZone,
  fitGbm,
  fitOu,
  monteCarloGbm,
  optimalEntryThreshold,
  volatilitySurface,
  type EntryZone,
  type GbmParams,
  type MonteCarloResult,
  type OuParams,
} from "./stochastic";
import { ema, lastFinite, macd, rollingZScore, rsi } from "./indicators";
import { kurtosis, logReturns, mean, skewness, stdev } from "./stats";

export const HMM_STATES = 3;

/**
 * Semua yang dibutuhkan untuk MEMUTUSKAN posisi (regime, mean-reversion,
 * timing entry, skor komposit). Sengaja TIDAK termasuk Monte Carlo cone /
 * volatility surface — keduanya untuk visualisasi risiko, bukan input
 * keputusan LONG/SHORT/WAIT, dan mahal untuk dihitung berulang kali (dipakai
 * juga oleh backtest walk-forward yang me-refit di banyak checkpoint).
 */
export interface SignalState {
  price: number;
  changePct: number;
  dt: number;
  barsPerYear: number;
  logRet: number[];
  closes: number[];
  times: number[];
  gbm: GbmParams;
  sigmaShort: number;
  sigmaLong: number;
  skew: number;
  excessKurtosis: number;
  hmm: HmmFit;
  regimeProbs: number[];
  regime: string;
  /** true bila probabilitas bertahan di regime saat ini (A_ii) ≥ ambang stabil. */
  regimeStable: boolean;
  regimePersistence: number;
  expectedDurationBars: number;
  ou: OuParams;
  z: number;
  zSeries: number[];
  entryThreshold: number;
  longZone: EntryZone | null;
  shortZone: EntryZone | null;
  rsi: number;
  macdHist: number;
  macdLine: number;
  score: number;
  action: "LONG" | "SHORT" | "WAIT";
  kellyFraction: number;
  statePoints: { ret: number; vol: number; state: number }[];
}

export interface SymbolAnalysis extends SignalState {
  symbol: string;
  mc: MonteCarloResult;
  surface: { grid: number[][]; moneyness: number[]; maturities: number[] };
}

const clamp = (v: number, lo = -1, hi = 1) => Math.min(hi, Math.max(lo, v));

/** Volatilitas bergulir (annualized) untuk sumbu Y ruang state HMM. */
function rollingVol(r: number[], window: number, dt: number): number[] {
  const out = new Array<number>(r.length).fill(NaN);
  for (let i = window - 1; i < r.length; i++) {
    out[i] = stdev(r.slice(i - window + 1, i + 1)) / Math.sqrt(dt);
  }
  return out;
}

/**
 * Hitung state sinyal (regime, mean-reversion, entry zone, skor komposit)
 * murni dari candle yang diberikan — tanpa lookahead: hanya memakai bar
 * candles[0..n-1]. Dipakai baik oleh dashboard live (lewat analyzeSymbol)
 * maupun backtest walk-forward (lewat backtest.ts), sehingga logika
 * matematika sinyal hanya ada di satu tempat.
 */
export function computeSignal(
  candles: Candle[],
  interval: Interval,
  riskFree = 0.04,
): SignalState {
  const barsPerYear = BARS_PER_YEAR[interval];
  const dt = 1 / barsPerYear;
  const closes = candles.map((c) => c.c);
  const times = candles.map((c) => c.t);
  const logRet = logReturns(closes);
  const price = closes[closes.length - 1] ?? 0;
  const prev = closes[closes.length - 2] ?? price;

  const gbm = fitGbm(logRet, dt);
  const shortWin = Math.min(30, Math.max(10, Math.floor(logRet.length * 0.1)));
  const sigmaShort = stdev(logRet.slice(-shortWin)) / Math.sqrt(dt);
  const sigmaLong = gbm.sigma;
  const skew = skewness(logRet);
  const excessKurtosis = kurtosis(logRet);

  const hmm = fitHmm(logRet, HMM_STATES);
  const regimeProbs = hmm.gamma[hmm.gamma.length - 1] ?? new Array(HMM_STATES).fill(1 / HMM_STATES);
  const currentState = regimeProbs.indexOf(Math.max(...regimeProbs));
  const persistence = hmm.params.A[currentState]?.[currentState] ?? 0;
  const regime = regimeLabel(currentState, HMM_STATES);
  // Durasi ekspektasi state (geometric): E[d] = 1/(1 − A_ii)
  const expectedDurationBars = persistence < 1 ? 1 / (1 - persistence) : Infinity;
  // Regime dianggap "stabil" bila probabilitas bertahan (A_ii) di atas ambang.
  // Di bawah ambang ini, HMM baru saja (atau sedang) berpindah state —
  // klasifikasi regime pada titik transisi secara empiris paling noisy,
  // jadi entry mean-reversion (yang menganggap proses OU berlaku terhadap
  // level ekuilibrium regime saat ini) paling berisiko salah di sini.
  const REGIME_STABLE_MIN = 0.7;
  const regimeStable = persistence >= REGIME_STABLE_MIN;

  const logClose = closes.map((c) => Math.log(c));
  const ou = fitOu(logClose, dt);
  const zWindow = Math.min(60, Math.max(20, Math.floor(closes.length / 8)));
  const zSeries = rollingZScore(logClose, zWindow);
  const z = lastFinite(zSeries);
  const sigmaZ = stdev(logClose.slice(-zWindow));
  const entryThreshold = optimalEntryThreshold(ou, sigmaZ, riskFree);
  // Rujukan level (log-price) yang sama dipakai rollingZScore untuk z saat ini.
  const zMeanLog = mean(logClose.slice(-zWindow));
  const ouValid = Number.isFinite(ou.theta) && ou.theta > 0;
  // Regime-gate (#1): LONG (bertaruh harga naik kembali ke mean) hanya masuk
  // akal bila regime saat ini bukan Bear yang sedang berlanjut — sebaliknya
  // untuk SHORT vs Bull. Trading melawan tren dominan yang masih kuat adalah
  // pola loss utama yang terlihat di backtest (entry LONG lalu kena stop
  // karena downtrend belum selesai).
  // Persistence-gate (#2): kedua sisi mati total bila regime belum stabil
  // (persistence < REGIME_STABLE_MIN), terlepas arahnya, karena estimasi
  // OU/level-ekuilibrium sendiri jadi tidak reliable saat regime baru bergeser.
  const canLong = ouValid && regimeStable && regime !== "Bear";
  const canShort = ouValid && regimeStable && regime !== "Bull";
  const longZone = canLong ? computeEntryZone("LONG", zMeanLog, sigmaZ, entryThreshold) : null;
  const shortZone = canShort ? computeEntryZone("SHORT", zMeanLog, sigmaZ, entryThreshold) : null;

  const rsiSeries = rsi(closes, 14);
  const rsiNow = lastFinite(rsiSeries);
  const macdRes = macd(closes, 12, 26, 9);
  const macdHist = lastFinite(macdRes.histogram);
  const macdLine = lastFinite(macdRes.macd);
  const histScale = stdev(macdRes.histogram.filter(Number.isFinite)) || 1;

  const pBull = regimeProbs[HMM_STATES - 1] ?? 0;
  const pBear = regimeProbs[0] ?? 0;
  const score = clamp(
    0.4 * (pBull - pBear) +
      0.25 * clamp(-z / 2) +
      0.15 * clamp((50 - rsiNow) / 30) +
      0.2 * clamp(macdHist / (2 * histScale)),
  );

  // Kelly fraksional: f* = (μ − r_f)/σ², dipakai 25% (quarter Kelly).
  const kelly = gbm.sigma > 0 ? (gbm.mu - riskFree) / (gbm.sigma * gbm.sigma) : 0;
  const kellyFraction = clamp(kelly * 0.25, -1, 1);

  const action: SignalState["action"] =
    score > 0.22 ? "LONG" : score < -0.22 ? "SHORT" : "WAIT";

  const vols = rollingVol(logRet, Math.min(20, Math.max(5, Math.floor(logRet.length / 20))), dt);
  const statePoints = logRet
    .map((r, i) => ({ ret: r, vol: vols[i]!, state: hmm.viterbi[i] ?? 0 }))
    .filter((p) => Number.isFinite(p.vol));

  return {
    price,
    changePct: prev > 0 ? (price / prev - 1) * 100 : 0,
    dt,
    barsPerYear,
    logRet,
    closes,
    times,
    gbm,
    sigmaShort,
    sigmaLong,
    skew,
    excessKurtosis,
    hmm,
    regimeProbs,
    regime,
    regimeStable,
    regimePersistence: persistence,
    expectedDurationBars,
    ou,
    z,
    zSeries,
    entryThreshold,
    longZone,
    shortZone,
    rsi: rsiNow,
    macdHist,
    macdLine,
    score,
    action,
    kellyFraction,
    statePoints,
  };
}

export function analyzeSymbol(
  symbol: string,
  candles: Candle[],
  interval: Interval,
  horizonBars = 72,
  riskFree = 0.04,
): SymbolAnalysis {
  const signal = computeSignal(candles, interval, riskFree);
  const mc = monteCarloGbm(signal.price, signal.gbm, horizonBars, 600);
  const surface = volatilitySurface({
    sigmaShort: signal.sigmaShort,
    sigmaLong: signal.sigmaLong,
    skew: signal.skew,
    excessKurtosis: signal.excessKurtosis,
    moneyness: Array.from({ length: 17 }, (_, i) => 0.6 + i * 0.05),
    maturities: Array.from({ length: 12 }, (_, i) => (7 + i * 14) / 365),
  });

  return { symbol, ...signal, mc, surface };
}

/** Matriks return T×N yang disejajarkan (panjang minimum lintas aset). */
export function alignedReturns(series: { symbol: string; candles: Candle[] }[]) {
  const rets = series.map((s) => logReturns(s.candles.map((c) => c.c)));
  const T = Math.min(...rets.map((r) => r.length));
  const R = Array.from({ length: T }, (_, t) =>
    rets.map((r) => r[r.length - T + t]!),
  );
  return { R, labels: series.map((s) => s.symbol.replace("USDT", "")) };
}

export const fmtPct = (v: number, d = 1) =>
  `${v >= 0 ? "+" : ""}${(v * 100).toFixed(d)}%`;

export const fmtPrice = (v: number) =>
  v >= 1000
    ? v.toLocaleString("en-US", { maximumFractionDigits: 0 })
    : v >= 1
      ? v.toFixed(2)
      : v.toFixed(5);

export { mean, ema };
