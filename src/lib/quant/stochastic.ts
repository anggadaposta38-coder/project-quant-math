/**
 * Stochastic calculus: GBM, Monte Carlo, Black-Scholes, Ornstein-Uhlenbeck.
 *
 * Konvensi: dt = panjang 1 bar dalam satuan tahun.
 * Dari log-return r_t ~ N((μ − σ²/2)dt, σ²dt):
 *   σ = sd(r)/√dt
 *   μ = mean(r)/dt + σ²/2      (koreksi Itô)
 */

import { mean, mulberry32, normCdf, quantile, randNorm, stdev } from "./stats";

export interface GbmParams {
  /** drift aritmetik tahunan μ */
  mu: number;
  /** volatilitas tahunan σ */
  sigma: number;
  /** drift log tahunan (μ − σ²/2) */
  logDrift: number;
  dt: number;
}

export function fitGbm(logRet: number[], dt: number): GbmParams {
  const sigma = stdev(logRet) / Math.sqrt(dt);
  const logDrift = mean(logRet) / dt;
  return { mu: logDrift + 0.5 * sigma * sigma, sigma, logDrift, dt };
}

export interface MonteCarloResult {
  /** paths[i][t] harga simulasi, t = 0..steps */
  paths: number[][];
  /** kuantil lintas simulasi per langkah waktu */
  bands: { p05: number[]; p25: number[]; p50: number[]; p75: number[]; p95: number[] };
  terminal: number[];
  expectedTerminal: number;
  var95: number;
  cvar95: number;
  probUp: number;
  horizonYears: number;
}

/**
 * Simulasi GBM eksak (bukan Euler):
 *   S_{t+Δ} = S_t · exp((μ − σ²/2)Δ + σ√Δ · Z)
 */
export function monteCarloGbm(
  s0: number,
  p: GbmParams,
  steps: number,
  nPaths: number,
  seed = 20260810,
): MonteCarloResult {
  const rng = mulberry32(seed);
  const drift = (p.mu - 0.5 * p.sigma * p.sigma) * p.dt;
  const vol = p.sigma * Math.sqrt(p.dt);
  const paths: number[][] = [];

  for (let i = 0; i < nPaths; i++) {
    const path = new Array<number>(steps + 1);
    path[0] = s0;
    let s = s0;
    for (let t = 1; t <= steps; t++) {
      s = s * Math.exp(drift + vol * randNorm(rng));
      path[t] = s;
    }
    paths.push(path);
  }

  const q = (t: number, prob: number) => {
    const col = paths.map((pt) => pt[t]!).sort((a, b) => a - b);
    return quantile(col, prob);
  };
  const bands = { p05: [] as number[], p25: [] as number[], p50: [] as number[], p75: [] as number[], p95: [] as number[] };
  for (let t = 0; t <= steps; t++) {
    bands.p05.push(q(t, 0.05));
    bands.p25.push(q(t, 0.25));
    bands.p50.push(q(t, 0.5));
    bands.p75.push(q(t, 0.75));
    bands.p95.push(q(t, 0.95));
  }

  const terminal = paths.map((pt) => pt[steps]!);
  const sortedT = [...terminal].sort((a, b) => a - b);
  const var95Price = quantile(sortedT, 0.05);
  const tailCount = Math.max(1, Math.floor(sortedT.length * 0.05));
  const tailMean = mean(sortedT.slice(0, tailCount));

  return {
    paths,
    bands,
    terminal,
    expectedTerminal: mean(terminal),
    var95: var95Price / s0 - 1,
    cvar95: tailMean / s0 - 1,
    probUp: terminal.filter((v) => v > s0).length / terminal.length,
    horizonYears: steps * p.dt,
  };
}

// ---------------------------------------------------------------------------
// Black-Scholes
// ---------------------------------------------------------------------------

export interface Greeks {
  price: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
}

/**
 * Black-Scholes Eropa.
 * d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T),  d2 = d1 − σ√T
 * C = S·Φ(d1) − K·e^{−rT}·Φ(d2)
 * P = K·e^{−rT}·Φ(−d2) − S·Φ(−d1)
 */
export function blackScholes(
  S: number,
  K: number,
  T: number,
  r: number,
  sigma: number,
  type: "call" | "put" = "call",
): Greeks {
  const sq = Math.sqrt(Math.max(T, 1e-12));
  const vol = Math.max(sigma, 1e-12);
  const d1 = (Math.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * sq);
  const d2 = d1 - vol * sq;
  const disc = Math.exp(-r * T);
  const pdf1 = Math.exp(-0.5 * d1 * d1) / Math.sqrt(2 * Math.PI);

  if (type === "call") {
    return {
      price: S * normCdf(d1) - K * disc * normCdf(d2),
      delta: normCdf(d1),
      gamma: pdf1 / (S * vol * sq),
      vega: S * pdf1 * sq,
      theta: (-(S * pdf1 * vol) / (2 * sq) - r * K * disc * normCdf(d2)),
      rho: K * T * disc * normCdf(d2),
    };
  }
  return {
    price: K * disc * normCdf(-d2) - S * normCdf(-d1),
    delta: normCdf(d1) - 1,
    gamma: pdf1 / (S * vol * sq),
    vega: S * pdf1 * sq,
    theta: (-(S * pdf1 * vol) / (2 * sq) + r * K * disc * normCdf(-d2)),
    rho: -K * T * disc * normCdf(-d2),
  };
}

/**
 * Permukaan volatilitas model (bukan quote pasar).
 * Skew/smile diturunkan dari ekspansi Corrado-Su terhadap moment realized
 * (skewness s dan excess kurtosis k) pada log-moneyness x = ln(K/S):
 *   σ(x,T) = σ_ATM(T)·[1 − (s/6)·(x/(σ√T)) + (k/24)·((x/(σ√T))² − 1)]
 * Term structure memakai mean-reversion vol ke level jangka panjang.
 */
export function volatilitySurface(opts: {
  sigmaShort: number;
  sigmaLong: number;
  skew: number;
  excessKurtosis: number;
  moneyness: number[];
  maturities: number[];
}): { grid: number[][]; moneyness: number[]; maturities: number[] } {
  const { sigmaShort, sigmaLong, skew, excessKurtosis, moneyness, maturities } = opts;
  const kappa = 3; // kecepatan mean-reversion varians (per tahun)
  const grid: number[][] = [];

  for (const T of maturities) {
    // Varians rata-rata terintegrasi pada model mean-reverting:
    // v̄(T) = v∞ + (v0 − v∞)·(1 − e^{−κT})/(κT)
    const v0 = sigmaShort * sigmaShort;
    const vInf = sigmaLong * sigmaLong;
    const decay = T > 1e-9 ? (1 - Math.exp(-kappa * T)) / (kappa * T) : 1;
    const atm = Math.sqrt(Math.max(vInf + (v0 - vInf) * decay, 1e-8));
    const row: number[] = [];
    for (const m of moneyness) {
      const x = Math.log(m); // m = K/S
      const z = x / (atm * Math.sqrt(Math.max(T, 1e-6)));
      const adj = 1 - (skew / 6) * z + (excessKurtosis / 24) * (z * z - 1);
      row.push(Math.max(atm * Math.min(Math.max(adj, 0.25), 3), 1e-4));
    }
    grid.push(row);
  }
  return { grid, moneyness, maturities };
}

// ---------------------------------------------------------------------------
// Ornstein-Uhlenbeck
// ---------------------------------------------------------------------------

export interface OuParams {
  /** kecepatan mean-reversion (per tahun) */
  theta: number;
  /** level ekuilibrium */
  mu: number;
  /** volatilitas difusi (per √tahun) */
  sigma: number;
  /** waktu paruh dalam jumlah bar */
  halfLifeBars: number;
  /** koefisien AR(1) */
  b: number;
}

/**
 * Estimasi OU dari diskretisasi eksak AR(1):
 *   X_{t+Δ} = μ(1−e^{−θΔ}) + e^{−θΔ}X_t + ε,  Var(ε) = σ²(1−e^{−2θΔ})/(2θ)
 * Sehingga: b = e^{−θΔ} ⇒ θ = −ln(b)/Δ, μ = a/(1−b),
 *   σ = sd(ε)·√(2θ/(1−b²))
 */
export function fitOu(x: number[], dt: number): OuParams {
  const n = x.length;
  if (n < 3) return { theta: 0, mu: mean(x), sigma: 0, halfLifeBars: Infinity, b: 1 };
  const xs = x.slice(0, n - 1);
  const ys = x.slice(1);
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    sxy += (xs[i]! - mx) * (ys[i]! - my);
    sxx += (xs[i]! - mx) ** 2;
  }
  const b = sxx === 0 ? 1 : sxy / sxx;
  const a = my - b * mx;
  const resid = xs.map((v, i) => ys[i]! - (a + b * v));
  const sdEps = stdev(resid);

  if (b <= 0 || b >= 1) {
    return { theta: 0, mu: mean(x), sigma: sdEps / Math.sqrt(dt), halfLifeBars: Infinity, b };
  }
  const theta = -Math.log(b) / dt;
  const mu = a / (1 - b);
  const sigma = sdEps * Math.sqrt((2 * theta) / (1 - b * b));
  return { theta, mu, sigma, halfLifeBars: Math.log(2) / (-Math.log(b)), b };
}

/**
 * Optimal stopping (entry) sederhana untuk proses mean-reverting:
 * ambang entry pada z* yang memaksimalkan E[keuntungan diskonto]
 *   V(z) = (μ − X)·e^{−r·τ(z)},  τ(z) = waktu ekspektasi kembali ke mean.
 * Dihitung numerik pada grid z.
 */
export function optimalEntryThreshold(ou: OuParams, sigmaZ: number, r = 0.05): number {
  if (!Number.isFinite(ou.theta) || ou.theta <= 0) return -2;
  let best = -2;
  let bestVal = -Infinity;
  for (let z = -3.5; z <= -0.1; z += 0.05) {
    const dist = -z * sigmaZ; // jarak ke mean dalam satuan harga log
    const tau = Math.log(Math.max(Math.abs(z), 1.0001)) / ou.theta; // waktu ekspektasi (tahun)
    const val = dist * Math.exp(-r * tau);
    if (val > bestVal) {
      bestVal = val;
      best = z;
    }
  }
  return best;
}
