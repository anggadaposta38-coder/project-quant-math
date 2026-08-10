/**
 * Stochastic calculus: GBM, Monte Carlo, Black-Scholes, Ornstein-Uhlenbeck.
 *
 * Konvensi: dt = panjang 1 bar dalam satuan tahun.
 * Dari log-return r_t ~ N((μ − σ²/2)dt, σ²dt):
 *   σ = sd(r)/√dt
 *   μ = mean(r)/dt + σ²/2      (koreksi Itô)
 */

import { annualizedVolatility, mean, mulberry32, normCdf, quantile, randNorm, shrinkageWeight, stdev } from "./stats";

export interface GbmParams {
  /** drift aritmetik tahunan μ setelah shrinkage regularisasi. */
  mu: number;
  /** volatilitas tahunan σ. */
  sigma: number;
  /** drift log tahunan μ_log = μ − σ²/2 setelah shrinkage. */
  logDrift: number;
  /** MLE log-drift sebelum shrinkage; disimpan untuk audit model. */
  rawLogDrift: number;
  /** Bobot shrinkage yang diterapkan ke rawLogDrift. */
  driftShrinkageWeight: number;
  dt: number;
}

export function fitGbm(logRet: number[], dt: number): GbmParams {
  if (!Number.isFinite(dt) || dt <= 0) throw new Error("dt harus finite dan > 0.");
  if (logRet.length < 2) throw new Error("GBM membutuhkan minimal 2 log-return.");
  if (logRet.some((r) => !Number.isFinite(r))) throw new Error("log-return harus finite.");
  const sigma = annualizedVolatility(logRet, 1 / dt);
  // σ diestimasi cukup stabil bahkan pada sampel pendek; μ (mean return)
  // jauh lebih noisy dan meledak saat dianualisasi dari data beberapa bulan.
  // Shrink drift mentah ke 0 sesuai jumlah tahun data yang tersedia.
  const years = logRet.length * dt;
  const rawLogDrift = mean(logRet) / dt;
  const driftShrinkageWeight = shrinkageWeight(years);
  const logDrift = rawLogDrift * driftShrinkageWeight;
  const mu = logDrift + 0.5 * sigma * sigma;
  if (![mu, sigma, logDrift, rawLogDrift, driftShrinkageWeight].every(Number.isFinite)) {
    throw new Error("Parameter GBM menghasilkan NaN/Infinity.");
  }
  return { mu, sigma, logDrift, rawLogDrift, driftShrinkageWeight, dt };
}

export interface MonteCarloResult {
  /** paths[i][t] harga simulasi, t = 0..steps */
  paths: number[][];
  /** predictive quantiles lintas simulasi per langkah waktu (bukan confidence interval statistik) */
  bands: { p05: number[]; p25: number[]; p50: number[]; p75: number[]; p95: number[] };
  terminal: number[];
  expectedTerminal: number;
  /** VaR 95% sebagai besar kerugian positif pada kuantil 5% terminal return. */
  var95: number;
  /** CVaR 95% sebagai besar kerugian positif rata-rata 5% tail terburuk. */
  cvar95: number;
  probUp: number;
  horizonYears: number;
  /** Seed yang dipakai; membuat hasil dapat direproduksi dan diaudit. */
  seed: number;
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
  if (!Number.isFinite(s0) || s0 <= 0) throw new Error("s0 harus finite dan > 0.");
  if (!Number.isFinite(p.mu) || !Number.isFinite(p.sigma) || p.sigma < 0 || !Number.isFinite(p.dt) || p.dt <= 0) {
    throw new Error("Parameter GBM tidak valid.");
  }
  if (!Number.isInteger(steps) || steps < 1) throw new Error("steps harus integer >= 1.");
  if (!Number.isInteger(nPaths) || nPaths < 1) throw new Error("nPaths harus integer >= 1.");
  if (!Number.isFinite(seed)) throw new Error("seed harus finite.");

  const normalizedSeed = Math.trunc(seed) >>> 0;
  const rng = mulberry32(normalizedSeed);

  // Simulate in log-price space. Multiplying the price at every step can
  // overflow to Infinity when a fitted crypto volatility is temporarily
  // extreme, even though all GBM parameters are finite. The log-space form
  // preserves the exact GBM update while keeping the intermediate exponent
  // inside JavaScript's representable positive-number range.
  const drift = (p.mu - 0.5 * p.sigma * p.sigma) * p.dt;
  const vol = p.sigma * Math.sqrt(p.dt);
  if (![drift, vol].every(Number.isFinite)) throw new Error("Parameter Monte Carlo menghasilkan NaN/Infinity.");

  const logS0 = Math.log(s0);
  const MAX_LOG_PRICE = Math.log(Number.MAX_VALUE);
  const MIN_LOG_PRICE = Math.log(Number.MIN_VALUE);
  if (!Number.isFinite(logS0)) throw new Error("Harga awal Monte Carlo menghasilkan log yang tidak valid.");

  const paths: number[][] = [];

  for (let i = 0; i < nPaths; i++) {
    const path = new Array<number>(steps + 1);
    path[0] = s0;
    let logS = logS0;
    for (let t = 1; t <= steps; t++) {
      const increment = drift + vol * randNorm(rng);
      if (!Number.isFinite(increment)) {
        throw new Error("Monte Carlo shock menghasilkan NaN/Infinity.");
      }

      logS += increment;
      // Clamp only at the floating-point representable boundaries. This is
      // not a model-parameter cap: it prevents JS exp() overflow/underflow
      // from poisoning all 600 paths with Infinity/0.
      logS = Math.min(MAX_LOG_PRICE, Math.max(MIN_LOG_PRICE, logS));
      const s = Math.exp(logS);
      if (!Number.isFinite(s) || s <= 0) {
        throw new Error("Monte Carlo menghasilkan harga non-finite.");
      }
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

  if ([...terminal, ...bands.p05, ...bands.p25, ...bands.p50, ...bands.p75, ...bands.p95].some((v) => !Number.isFinite(v))) {
    throw new Error("Monte Carlo menghasilkan NaN/Infinity.");
  }

  return {
    paths,
    bands,
    terminal,
    expectedTerminal: mean(terminal),
    var95: 1 - var95Price / s0,
    cvar95: 1 - tailMean / s0,
    probUp: terminal.filter((v) => v > s0).length / terminal.length,
    horizonYears: steps * p.dt,
    seed: normalizedSeed,
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
  if (![S, K, T, r, sigma].every(Number.isFinite) || S <= 0 || K <= 0 || T < 0 || sigma < 0) {
    throw new Error("Parameter Black-Scholes tidak valid.");
  }

  // At expiry the d1/d2 formulas are singular. Return the well-defined
  // intrinsic value and the conventional ATM delta instead of fabricating
  // enormous gamma/theta values through an epsilon T.
  if (T === 0) {
    const intrinsic = type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
    const delta = S > K ? (type === "call" ? 1 : 0) : S < K ? (type === "call" ? 0 : -1) : type === "call" ? 0.5 : -0.5;
    return { price: intrinsic, delta, gamma: 0, vega: 0, theta: 0, rho: 0 };
  }

  // Zero volatility is deterministic. Handle its continuous-price limit
  // explicitly; replacing sigma by 1e-12 creates meaningless Greeks.
  if (sigma === 0) {
    const disc = Math.exp(-r * T);
    const forwardIntrinsic = type === "call" ? S - K * disc : K * disc - S;
    const price = Math.max(forwardIntrinsic, 0);
    const boundary = K * disc;
    const delta = type === "call" ? (S > boundary ? 1 : S < boundary ? 0 : 0.5) : (S > boundary ? 0 : S < boundary ? -1 : -0.5);
    const rho = type === "call" ? K * T * disc * (forwardIntrinsic > 0 ? 1 : 0) : -K * T * disc * (forwardIntrinsic > 0 ? 1 : 0);
    const theta = type === "call" ? r * K * disc * (forwardIntrinsic > 0 ? 1 : 0) : -r * K * disc * (forwardIntrinsic > 0 ? 1 : 0);
    return { price, delta, gamma: 0, vega: 0, theta, rho };
  }

  const sq = Math.sqrt(T);
  const vol = sigma;
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
 * Permukaan volatilitas model (bukan implied-volatility quote pasar).
 * Formula ini adalah moment-based heuristic; tidak menjamin static-arbitrage
 * (monotonicity/convexity terhadap strike atau calendar arbitrage).
 * Skew/smile diturunkan dari ekspansi Corrado-Su terhadap moment realized
 * (skewness s dan excess kurtosis k) pada log-moneyness x = ln(K/S):
 *   σ(x,T) = σ_ATM(T)·[1 + (s/6)·(x/(σ√T)) + (k/24)·((x/(σ√T))² − 1)]
 * (ekuivalen Backus et al. dengan d = −x/(σ√T): skew negatif ⇒ IV lebih tinggi
 * pada strike rendah / OTM put.)
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
  if (![sigmaShort, sigmaLong, skew, excessKurtosis].every(Number.isFinite) || sigmaShort < 0 || sigmaLong < 0) {
    throw new Error("Parameter volatility surface tidak valid.");
  }
  if (moneyness.some((m) => !Number.isFinite(m) || m <= 0) || maturities.some((T) => !Number.isFinite(T) || T <= 0)) {
    throw new Error("Grid volatility surface mengandung nilai tidak valid.");
  }
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
      const adj = 1 + (skew / 6) * z + (excessKurtosis / 24) * (z * z - 1);
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
  /** R² regresi AR(1), untuk audit kecocokan model. */
  rSquared: number;
  /** Statistik Dickey-Fuller sederhana (tanpa lagged differences). */
  dfStatistic: number;
  /** Kandidat stasioner berdasarkan simple Dickey-Fuller, bukan ADF penuh. */
  stationary: boolean;
  /** Fit cukup informatif untuk dipakai sebagai signal OU. */
  reliable: boolean;
  /** Simpangan baku inovasi AR(1) dengan estimator conditional MLE. */
  innovationStd: number;
}

/**
 * Estimasi OU dari diskretisasi eksak AR(1):
 *   X_{t+Δ} = μ(1−e^{−θΔ}) + e^{−θΔ}X_t + ε,  Var(ε) = σ²(1−e^{−2θΔ})/(2θ)
 * Sehingga: b = e^{−θΔ} ⇒ θ = −ln(b)/Δ, μ = a/(1−b),
 *   σ = sd(ε)·√(2θ/(1−b²))
 */
export const OU_MIN_RELIABLE_OBS = 60;
export const OU_MAX_HALF_LIFE_FRACTION = 0.5;

export function fitOu(x: number[], dt: number): OuParams {
  if (!Number.isFinite(dt) || dt <= 0) throw new Error("dt harus finite dan > 0.");
  if (x.some((v) => !Number.isFinite(v))) throw new Error("Observasi OU harus finite.");
  const n = x.length;
  if (n < 3) {
    return { theta: 0, mu: mean(x), sigma: 0, halfLifeBars: Infinity, b: 1, rSquared: 0, dfStatistic: 0, stationary: false, reliable: false, innovationStd: 0 };
  }
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
  const sse = resid.reduce((sum, e) => sum + e * e, 0);
  // Conditional Gaussian MLE for innovation variance uses SSE / N_resid,
  // where N_resid = n - 1. `stdev(resid)` uses n - 2 and is unbiased for
  // regression residual variance, but it is not the OU transition MLE.
  const innovationVariance = resid.length > 0 ? sse / resid.length : 0;
  const innovationStd = Math.sqrt(Math.max(innovationVariance, 0));
  const tss = ys.reduce((sum, y) => sum + (y - my) ** 2, 0);
  const rSquared = tss > 0 ? Math.max(0, Math.min(1, 1 - sse / tss)) : 0;

  // Dickey-Fuller regression: ΔX_t = c + γ X_{t-1} + ε_t, γ = b - 1.
  // This is intentionally labelled a simple DF statistic, not a full ADF test.
  const gamma = b - 1;
  const mse = xs.length > 2 ? sse / (xs.length - 2) : Infinity;
  const seGamma = sxx > 0 && mse >= 0 ? Math.sqrt(mse / sxx) : Infinity;
  const dfStatistic = seGamma > 0 && Number.isFinite(seGamma) ? gamma / seGamma : 0;

  if (b <= 0 || b >= 1) {
    return { theta: 0, mu: mean(x), sigma: Number.isFinite(innovationStd) ? innovationStd / Math.sqrt(dt) : 0, halfLifeBars: Infinity, b, rSquared, dfStatistic, stationary: false, reliable: false, innovationStd };
  }

  const theta = -Math.log(b) / dt;
  const mu = a / (1 - b);
  const sigma = innovationStd * Math.sqrt((2 * theta) / (1 - b * b));
  const halfLifeBars = Math.log(2) / (-Math.log(b));
  const stationary = dfStatistic <= -2.86;
  const reliable =
    n >= OU_MIN_RELIABLE_OBS &&
    stationary &&
    halfLifeBars <= n * OU_MAX_HALF_LIFE_FRACTION &&
    rSquared >= 0 &&
    [theta, mu, sigma, halfLifeBars, rSquared, dfStatistic].every(Number.isFinite);

  if (![theta, mu, sigma, halfLifeBars, rSquared, dfStatistic, innovationStd].every(Number.isFinite)) {
    return { theta: 0, mu: mean(x), sigma: 0, halfLifeBars: Infinity, b, rSquared, dfStatistic, stationary: false, reliable: false, innovationStd: Number.isFinite(innovationStd) ? innovationStd : 0 };
  }
  return { theta, mu, sigma, halfLifeBars, b, rSquared, dfStatistic, stationary, reliable, innovationStd };
}

/**
 * Ambang entry optimal untuk proses mean-reverting (pendekatan optimal stopping
 * diskrit atas grid z). Untuk entry di z < 0 dan exit di z_exit = −0.25:
 *   - keuntungan log ≈ (|z| − |z_exit|)·σ_z
 *   - waktu tempuh ekspektasi (jalur rerata OU: E[X_t|X_0] = μ + (X_0−μ)e^{−θt}):
 *     τ(z) = ln(|z| / |z_exit|) / θ
 *   - peluang kesempatan itu muncul (densitas stasioner): Φ(z)
 * Objektif: V(z) = Φ(z) · (|z| − |z_exit|)·σ_z · e^{−r·τ(z)} → dimaksimalkan.
 */
export const OU_Z_EXIT = 0.25;
/**
 * Target rasio reward:risk minimum untuk zona stop-loss.
 *
 * Sebelumnya stop buffer memakai konstanta tetap (OU_STOP_BUFFER_Z = 1.0σ_z)
 * yang sama sekali independen dari hasil optimalEntryThreshold. Untuk rentang
 * θ yang realistis, optimalEntryThreshold konvergen ke |z| ≈ 0.93 (karena
 * untuk θ tidak terlalu kecil, faktor diskon e^{-r·τ(z)} ≈ 1 dan objektif
 * V(z) ≈ Φ(z)·(|z|−z_exit) sudah tidak lagi bergantung θ) — sehingga reward
 * tipikal hanya ≈ |z| − z_exit ≈ 0.68σ_z. Dengan risk tetap 1.0σ_z, R:R ≈ 0.68
 * dan breakeven win rate ≈ risk/(risk+reward) ≈ 59.5% — jauh di atas win rate
 * empiris strategi OU-zone (~41% pada backtest). Ini penyebab struktural
 * profit factor rendah meski sebagian besar trade "benar arah".
 *
 * Perbaikan: turunkan stop buffer dari jarak reward itu sendiri
 * (rewardZ / OU_TARGET_RR), bukan konstanta lepas — supaya R:R konsisten
 * berapa pun nilai entryThreshold yang dihasilkan (mis. bila r/z_exit di-tune
 * di masa depan).
 */
export const OU_TARGET_RR = 1.5;
/** Batas bawah buffer stop (dalam σ_z) agar tidak terlalu sempit saat reward kecil. */
export const OU_STOP_BUFFER_MIN_Z = 0.35;

export function optimalEntryThreshold(ou: OuParams, sigmaZ: number, r = 0.05): number {
  if (!Number.isFinite(ou.theta) || ou.theta <= 0 || !Number.isFinite(sigmaZ) || sigmaZ <= 0 || !Number.isFinite(r)) return -2;
  const zExit = OU_Z_EXIT;
  let best = -2;
  let bestVal = -Infinity;
  for (let z = -3.5; z <= -0.3; z += 0.01) {
    const az = Math.abs(z);
    const gain = (az - zExit) * sigmaZ;
    if (gain <= 0) continue;
    const tau = Math.log(az / zExit) / ou.theta;
    const val = normCdf(z) * gain * Math.exp(-r * tau);
    if (val > bestVal) {
      bestVal = val;
      best = z;
    }
  }
  return best;
}

export interface EntryZone {
  direction: "LONG" | "SHORT";
  /** harga masuk (level oversold/overbought optimal, dari optimalEntryThreshold) */
  entry: number;
  /** harga stop-loss (tesis mean-reversion dianggap gagal jika tertembus) */
  stop: number;
  /** harga target (exit di z_exit, harga sudah pulih sebagian ke arah mean) */
  target: number;
  /** rasio |target − entry| / |entry − stop| */
  riskReward: number;
}

/**
 * Konversi ambang z-score (dari optimalEntryThreshold, OU) menjadi zona harga.
 * P(z) = exp(refLog + z·σ_z), dengan refLog = rata-rata bergulir log-price
 * (rujukan yang sama dipakai rollingZScore untuk menghitung z saat ini).
 *
 * SHORT adalah cerminan LONG (proses OU simetris terhadap μ): entry di
 * +|entryThreshold|, exit di +z_exit. Stop diletakkan lebih jauh dari entry,
 * ke arah yang melawan posisi, dengan jarak = rewardZ / OU_TARGET_RR (bukan
 * konstanta tetap) supaya R:R konsisten ≥ OU_TARGET_RR (lihat komentar di atas
 * OU_TARGET_RR untuk alasan perubahan ini).
 *
 * Asumsi: refLog & σ_z dianggap tetap dari sekarang sampai harga menyentuh
 * zona — estimasi snapshot, bukan proyeksi harga masa depan.
 */
export function computeEntryZone(
  direction: "LONG" | "SHORT",
  refLog: number,
  sigmaZ: number,
  entryThresholdNeg: number,
): EntryZone | null {
  if (!Number.isFinite(entryThresholdNeg) || !Number.isFinite(sigmaZ) || sigmaZ <= 0) {
    return null;
  }
  const sign = direction === "LONG" ? 1 : -1;
  const zEntry = sign * entryThresholdNeg; // LONG: negatif (oversold), SHORT: positif (overbought)
  const zTarget = -sign * OU_Z_EXIT; // LONG: -0.25, SHORT: +0.25
  const rewardZ = Math.max(Math.abs(entryThresholdNeg) - OU_Z_EXIT, 1e-6);
  const stopBufferZ = Math.max(OU_STOP_BUFFER_MIN_Z, rewardZ / OU_TARGET_RR);
  const zStop = zEntry - sign * stopBufferZ; // lebih jauh ke arah melawan posisi

  const toPrice = (z: number) => Math.exp(refLog + z * sigmaZ);
  const entry = toPrice(zEntry);
  const stop = toPrice(zStop);
  const target = toPrice(zTarget);

  const risk = Math.abs(entry - stop);
  const reward = Math.abs(target - entry);
  const riskReward = risk > 0 ? reward / risk : 0;

  return { direction, entry, stop, target, riskReward };
}

