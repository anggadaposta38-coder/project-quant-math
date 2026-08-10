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
import { annualizedVolatility, kurtosis, logReturns, mean, skewness, stdev, stableSeed } from "./stats";

export const HMM_STATES = 3;

/**
 * Minimum closed candles required for a fully-warmed signal.
 *
 * The slowest canonical timing indicator is MACD(12,26,9), whose signal
 * becomes valid only after 26 + 9 - 1 = 34 closes. We deliberately require
 * a larger fixed history so adaptive windows never silently shrink on short
 * samples and produce a signal with different statistical meaning.
 */
export const MIN_SIGNAL_BARS = 60;
export const RSI_PERIOD = 14;
export const MACD_FAST = 12;
export const MACD_SLOW = 26;
export const MACD_SIGNAL = 9;
export const Z_WINDOW = 60;
export const VOL_WINDOW = 30;

/** Shared signal-policy constants: keep tuning knobs named and auditable. */
export const SIGNAL_WEIGHTS = {
  regime: 0.4,
  z: 0.25,
  rsi: 0.15,
  macd: 0.2,
} as const;
export const SIGNAL_ACTION_THRESHOLD = 0.22;
export const QUARTER_KELLY = 0.25;
export const REGIME_STABLE_MIN = 0.7;

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
  /** Komponen score yang sudah dinormalisasi ke [-1, 1]; score tetap heuristic, bukan probabilitas. */
  scoreComponents: { regime: number; z: number; rsi: number; macd: number };
  action: "LONG" | "SHORT" | "WAIT";
  /** Signed quarter-Kelly yang sudah disejajarkan dengan action; 0 bila WAIT atau Kelly berlawanan arah. */
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
    out[i] = annualizedVolatility(r.slice(i - window + 1, i + 1), 1 / dt);
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
  if (candles.length < MIN_SIGNAL_BARS) {
    throw new Error(
      `Signal membutuhkan minimal ${MIN_SIGNAL_BARS} candle yang sudah closed; tersedia ${candles.length}.`,
    );
  }
  if (candles.some((c) => !Number.isFinite(c.c) || c.c <= 0)) {
    throw new Error("Close price harus finite dan > 0.");
  }

  const barsPerYear = BARS_PER_YEAR[interval];
  const dt = 1 / barsPerYear;
  const closes = candles.map((c) => c.c);
  const times = candles.map((c) => c.t);
  const logRet = logReturns(closes);
  const price = closes[closes.length - 1] ?? 0;
  const prev = closes[closes.length - 2] ?? price;

  const gbm = fitGbm(logRet, dt);
  const shortWin = VOL_WINDOW;
  const sigmaShort = annualizedVolatility(logRet.slice(-shortWin), barsPerYear);
  const sigmaLong = gbm.sigma;
  const skew = skewness(logRet);
  const excessKurtosis = kurtosis(logRet);

  const hmm = fitHmm(logRet, HMM_STATES);
  const regimeProbs = hmm.gamma[hmm.gamma.length - 1] ?? new Array(HMM_STATES).fill(1 / HMM_STATES);
  const currentState = regimeProbs.indexOf(Math.max(...regimeProbs));
  const persistence = hmm.params.A[currentState]?.[currentState] ?? 0;
  const regime = regimeLabel(currentState, HMM_STATES);
  const pBear = regimeProbs[0] ?? 0;
  const pBull = regimeProbs[HMM_STATES - 1] ?? 0;
  // A state whose emission mean is almost indistinguishable from its neighbor
  // is not a reliable semantic Bear/Sideways/Bull label. Treat such fits as
  // ambiguous rather than letting tiny EM noise flip the regime between
  // walk-forward refits. The threshold is deliberately conservative: 0.25
  // pooled standard deviations between adjacent state means.
  const adjacentSeparations = hmm.params.mu.slice(0, -1).map((m, i) => {
    const next = hmm.params.mu[i + 1]!;
    const pooled = Math.sqrt((hmm.params.sigma[i]! ** 2 + hmm.params.sigma[i + 1]! ** 2) / 2);
    return pooled > 0 ? Math.abs(next - m) / pooled : Infinity;
  });
  const regimeLabelsSeparated = adjacentSeparations.every((d) => d >= 0.25);
  // Durasi ekspektasi state (geometric): E[d] = 1/(1 − A_ii)
  const expectedDurationBars = persistence < 1 ? 1 / (1 - persistence) : Infinity;
  // Regime dianggap "stabil" bila probabilitas bertahan (A_ii) di atas ambang.
  // Di bawah ambang ini, HMM baru saja (atau sedang) berpindah state —
  // klasifikasi regime pada titik transisi secara empiris paling noisy,
  // jadi entry mean-reversion (yang menganggap proses OU berlaku terhadap
  // level ekuilibrium regime saat ini) paling berisiko salah di sini.
  const regimeStable = hmm.converged && persistence >= REGIME_STABLE_MIN && regimeLabelsSeparated;

  const logClose = closes.map((c) => Math.log(c));
  const ou = fitOu(logClose, dt);
  const zWindow = Z_WINDOW;
  const zSeries = rollingZScore(logClose, zWindow);
  const z = lastFinite(zSeries);
  const sigmaZ = stdev(logClose.slice(-zWindow));
  const entryThreshold = optimalEntryThreshold(ou, sigmaZ, riskFree);
  // Rujukan level (log-price) yang sama dipakai rollingZScore untuk z saat ini.
  const zMeanLog = mean(logClose.slice(-zWindow));
  const ouValid =
    ou.reliable &&
    ou.stationary &&
    Number.isFinite(ou.theta) &&
    ou.theta > 0 &&
    Number.isFinite(ou.halfLifeBars) &&
    ou.halfLifeBars <= logClose.length * 0.5 &&
    Number.isFinite(sigmaZ) &&
    sigmaZ > 0;
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

  const rsiSeries = rsi(closes, RSI_PERIOD);
  const rsiNow = lastFinite(rsiSeries);
  const macdRes = macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL);
  const macdHist = lastFinite(macdRes.histogram);
  const macdLine = lastFinite(macdRes.macd);
  const histScale = stdev(macdRes.histogram.filter(Number.isFinite)) || 1;
  if (![rsiNow, macdHist, macdLine, histScale, z, sigmaZ].every(Number.isFinite)) {
    throw new Error("Indikator menghasilkan NaN/Infinity setelah warm-up.");
  }

  // Semua komponen dipetakan ke [-1, 1]. Bobot di bawah adalah heuristic
  // (bukan hasil kalibrasi probabilistik), sehingga `score` tidak boleh
  // diinterpretasikan sebagai P(profit). Komponen disimpan terpisah agar
  // kontribusi signal dapat diaudit dan kelak dikalibrasi dari walk-forward.
  const scoreComponents = {
    regime: clamp(pBull - pBear),
    z: clamp(-z / 2),
    rsi: clamp((50 - rsiNow) / 30),
    macd: clamp(macdHist / (2 * histScale)),
  };
  const score = clamp(
    SIGNAL_WEIGHTS.regime * scoreComponents.regime +
      SIGNAL_WEIGHTS.z * scoreComponents.z +
      SIGNAL_WEIGHTS.rsi * scoreComponents.rsi +
      SIGNAL_WEIGHTS.macd * scoreComponents.macd,
  );

  const action: SignalState["action"] =
    score > SIGNAL_ACTION_THRESHOLD
      ? "LONG"
      : score < -SIGNAL_ACTION_THRESHOLD
        ? "SHORT"
        : "WAIT";

  // Kelly fraksional: f* = (μ − r_f)/σ², dipakai 25% (quarter Kelly).
  // IMPORTANT: GBM Kelly berasal dari return model, sedangkan action berasal
  // dari composite signal. Jangan membiarkan tanda Kelly yang berlawanan
  // menghasilkan posisi untuk arah yang salah. Signed fraction hanya positif
  // jika estimasi Kelly mendukung action; WAIT selalu 0.
  const rawKelly = gbm.sigma > 0 ? (gbm.mu - riskFree) / (gbm.sigma * gbm.sigma) : 0;
  const directionalKelly = action === "LONG" ? rawKelly : action === "SHORT" ? -rawKelly : 0;
  const kellyFraction = Number.isFinite(directionalKelly)
    ? clamp(directionalKelly * QUARTER_KELLY, 0, 1) * (action === "SHORT" ? -1 : action === "LONG" ? 1 : 0)
    : 0;

  const vols = rollingVol(logRet, VOL_WINDOW, dt);
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
    scoreComponents,
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
  // Reproducible, tetapi setiap symbol mendapat stream RNG yang berbeda.
  // Ini menghindari semua symbol memakai urutan shock yang identik secara
  // tidak sengaja saat beberapa cone ditampilkan berdampingan.
  const mcSeed = stableSeed(`${symbol}|${interval}|${horizonBars}`);
  const mc = monteCarloGbm(signal.price, signal.gbm, horizonBars, 600, mcSeed);
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

/**
 * Matriks return T×N yang disejajarkan berdasarkan timestamp.
 *
 * Hanya return yang benar-benar satu bar sesuai `interval` yang boleh masuk.
 * Ini mencegah missing candle pada satu aset berubah menjadi return 2×/3×
 * periode lalu disejajarkan dengan return 1× periode aset lain.
 */
export function alignedReturns(
  series: { symbol: string; candles: Candle[] }[],
  interval?: Interval,
) {
  if (series.length === 0) return { R: [] as number[][], labels: [] as string[] };

  const intervalMs: number | undefined = interval ? {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
  }[interval] : undefined;

  const modeDelta = (candles: Candle[]) => {
    const counts = new Map<number, number>();
    for (let i = 1; i < candles.length; i++) {
      const d = candles[i]!.t - candles[i - 1]!.t;
      if (Number.isFinite(d) && d > 0) counts.set(d, (counts.get(d) ?? 0) + 1);
    }
    let best = 0;
    let bestCount = 0;
    for (const [d, count] of counts) {
      if (count > bestCount || (count === bestCount && d < best)) {
        best = d;
        bestCount = count;
      }
    }
    return best;
  };

  const expectedDeltas = series.map((s) => intervalMs ?? modeDelta(s.candles));
  if (expectedDeltas.some((d) => !(d > 0))) {
    throw new Error("Tidak dapat menentukan interval return portfolio.");
  }
  if (intervalMs === undefined && expectedDeltas.some((d) => d !== expectedDeltas[0])) {
    throw new Error("Series portfolio memiliki interval candle yang berbeda.");
  }

  const byTime = series.map((s, seriesIndex) => {
    const out = new Map<number, number>();
    for (let i = 1; i < s.candles.length; i++) {
      const prev = s.candles[i - 1]!;
      const cur = s.candles[i]!;
      if (cur.t <= prev.t || cur.t - prev.t !== expectedDeltas[seriesIndex] || prev.c <= 0 || cur.c <= 0) continue;
      const r = Math.log(cur.c / prev.c);
      if (Number.isFinite(r)) out.set(cur.t, r);
    }
    return out;
  });

  const commonTimes = [...byTime[0]!.keys()]
    .filter((t) => byTime.every((m) => m.has(t)))
    .sort((a, b) => a - b);
  const R = commonTimes.map((t) => byTime.map((m) => m.get(t)!));
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
