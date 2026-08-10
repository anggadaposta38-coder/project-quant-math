/** Indikator timing entry: EMA, MACD, RSI (Wilder), rolling Z-score. */

import { mean, stdev } from "./stats";

/**
 * EMA dengan α = 2/(n+1). Seed = SMA dari n data pertama (konvensi standar).
 * Output sejajar input; index < n-1 bernilai NaN.
 */
export function ema(x: number[], period: number): number[] {
  const out = new Array<number>(x.length).fill(NaN);
  if (!Number.isInteger(period) || period < 1) throw new Error("EMA period harus integer >= 1.");
  if (x.some((v) => !Number.isFinite(v))) throw new Error("EMA input harus finite.");
  if (x.length < period) return out;
  const alpha = 2 / (period + 1);
  let prev = mean(x.slice(0, period));
  out[period - 1] = prev;
  for (let i = period; i < x.length; i++) {
    prev = alpha * x[i]! + (1 - alpha) * prev;
    out[i] = prev;
  }
  return out;
}

export interface MacdResult {
  macd: number[];
  signal: number[];
  histogram: number[];
}

/** MACD = EMA_fast − EMA_slow; signal = EMA_signal(MACD). */
export function macd(
  x: number[],
  fast = 12,
  slow = 26,
  signalPeriod = 9,
): MacdResult {
  if (!Number.isInteger(fast) || !Number.isInteger(slow) || !Number.isInteger(signalPeriod) || fast < 1 || slow < 1 || signalPeriod < 1 || fast >= slow) {
    throw new Error("MACD periods harus integer positif dengan fast < slow.");
  }
  const ef = ema(x, fast);
  const es = ema(x, slow);
  const line = x.map((_, i) =>
    Number.isNaN(ef[i]!) || Number.isNaN(es[i]!) ? NaN : ef[i]! - es[i]!,
  );

  // EMA sinyal hanya dihitung atas bagian MACD yang valid.
  const firstValid = line.findIndex((v) => !Number.isNaN(v));
  const signal = new Array<number>(x.length).fill(NaN);
  if (firstValid >= 0) {
    const valid = line.slice(firstValid);
    const sig = ema(valid, signalPeriod);
    for (let i = 0; i < sig.length; i++) signal[firstValid + i] = sig[i]!;
  }
  const histogram = line.map((v, i) =>
    Number.isNaN(v) || Number.isNaN(signal[i]!) ? NaN : v - signal[i]!,
  );
  return { macd: line, signal, histogram };
}

/**
 * RSI Wilder: RSI = 100 − 100/(1+RS), RS = avgGain/avgLoss
 * dengan smoothing Wilder (α = 1/period).
 */
export function rsi(prices: number[], period = 14): number[] {
  const out = new Array<number>(prices.length).fill(NaN);
  if (!Number.isInteger(period) || period < 1) throw new Error("RSI period harus integer >= 1.");
  if (prices.some((v) => !Number.isFinite(v))) throw new Error("RSI input harus finite.");
  if (prices.length <= period) return out;

  let gain = 0;
  let loss = 0;
  for (let i = 1; i <= period; i++) {
    const d = prices[i]! - prices[i - 1]!;
    if (d >= 0) gain += d;
    else loss -= d;
  }
  let avgGain = gain / period;
  let avgLoss = loss / period;
  out[period] = avgLoss === 0 ? (avgGain === 0 ? 50 : 100) : 100 - 100 / (1 + avgGain / avgLoss);

  for (let i = period + 1; i < prices.length; i++) {
    const d = prices[i]! - prices[i - 1]!;
    const g = d > 0 ? d : 0;
    const l = d < 0 ? -d : 0;
    avgGain = (avgGain * (period - 1) + g) / period;
    avgLoss = (avgLoss * (period - 1) + l) / period;
    out[i] = avgLoss === 0 ? (avgGain === 0 ? 50 : 100) : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return out;
}

/** Z-score bergulir: Z_t = (P_t − μ_roll) / σ_roll */
export function rollingZScore(x: number[], window = 50): number[] {
  const out = new Array<number>(x.length).fill(NaN);
  if (!Number.isInteger(window) || window < 2) throw new Error("Z-score window harus integer >= 2.");
  if (x.some((v) => !Number.isFinite(v))) throw new Error("Z-score input harus finite.");
  for (let i = window - 1; i < x.length; i++) {
    const w = x.slice(i - window + 1, i + 1);
    const s = stdev(w);
    out[i] = s === 0 ? 0 : (x[i]! - mean(w)) / s;
  }
  return out;
}

export function lastFinite(x: number[]): number {
  for (let i = x.length - 1; i >= 0; i--) if (Number.isFinite(x[i]!)) return x[i]!;
  return NaN;
}
