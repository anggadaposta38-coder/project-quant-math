/**
 * Portfolio & risk: Markowitz mean-variance dan PCA.
 *
 * Solusi analitik Markowitz (dua kendala, tanpa batas short):
 *   min wᵀΣw  s.t. wᵀμ = r_target, 1ᵀw = 1
 * Definisikan A = 1ᵀΣ⁻¹1, B = 1ᵀΣ⁻¹μ, C = μᵀΣ⁻¹μ, D = AC − B²
 *   w*(r) = Σ⁻¹(λ·1 + γ·μ),  λ = (C − B·r)/D,  γ = (A·r − B)/D
 *   σ²(r) = (A·r² − 2B·r + C)/D
 * Portofolio varians minimum global: w_gmv = Σ⁻¹1 / A, r_gmv = B/A, σ²_gmv = 1/A
 */

import {
  correlationFromCov,
  covarianceMatrix,
  dot,
  inverse,
  jacobiEigen,
  matVec,
  quadForm,
  ridge,
  nearestPsd,
  shrinkageWeight,
  type Matrix,
} from "./stats";

export interface FrontierPoint {
  ret: number;
  risk: number;
  weights: number[];
}

export interface PortfolioAnalysis {
  cov: Matrix;
  corr: Matrix;
  /** μ tahunan per aset */
  mu: number[];
  frontier: FrontierPoint[];
  gmv: FrontierPoint;
  tangency: FrontierPoint | null;
  maxSharpe: number;
  /** optimum tanpa short (w ≥ 0, Σw = 1) */
  longOnly: FrontierPoint & { sharpe: number };

  riskFree: number;
}

function scalars(Sinv: Matrix, mu: number[]) {
  const ones = new Array<number>(mu.length).fill(1);
  const SinvOne = matVec(Sinv, ones);
  const SinvMu = matVec(Sinv, mu);
  const A = dot(ones, SinvOne);
  const B = dot(ones, SinvMu);
  const C = dot(mu, SinvMu);
  return { A, B, C, D: A * C - B * B, SinvOne, SinvMu };
}

export function efficientWeights(Sinv: Matrix, mu: number[], target: number): number[] {
  const { A, B, C, D, SinvOne, SinvMu } = scalars(Sinv, mu);
  if (Math.abs(D) < 1e-18) return SinvOne.map((v) => v / (A || 1));
  const lambda = (C - B * target) / D;
  const gamma = (A * target - B) / D;
  return SinvOne.map((v, i) => lambda * v + gamma * SinvMu[i]!);
}

/**
 * @param R matriks return T×N (log-return per bar)
 * @param barsPerYear faktor anualisasi
 */
export function analyzePortfolio(
  R: Matrix,
  barsPerYear: number,
  riskFree = 0.04,
  points = 60,
): PortfolioAnalysis {
  const T = R.length;
  const N = T > 0 ? R[0]!.length : 0;
  if (N === 0) throw new Error("Portfolio membutuhkan minimal satu aset.");
  if (!Number.isFinite(barsPerYear) || barsPerYear <= 0) {
    throw new Error("barsPerYear harus finite dan > 0.");
  }
  if (!Number.isFinite(riskFree)) throw new Error("riskFree harus finite.");
  if (!Number.isInteger(points) || points < 2) throw new Error("points harus integer >= 2.");
  if (T < 2 || R.some((row) => row.length !== N || row.some((v) => !Number.isFinite(v)))) {
    throw new Error("Return matrix tidak valid atau terlalu pendek.");
  }
  // Markowitz memakai return aritmetik (simple return), bukan log-return.
  // Input tetap berupa log-return sesuai kontrak API, tetapi mean dan kovarians
  // dikonversi ke simple return sebelum annualization agar risk/return dan
  // risk-free rate berada dalam unit yang sama. Menggunakan m + σ²/2 langsung
  // di sini mencampur drift GBM kontinu dengan expected simple return.
  const simpleR: Matrix = R.map((row) => row.map((r) => Math.expm1(r)));
  const covBar = covarianceMatrix(simpleR);
  const rawCov = covBar.map((row) => row.map((v) => v * barsPerYear));
  const cov = nearestPsd(rawCov, 1e-10);
  const corr = correlationFromCov(cov);
  if (cov.some((row, i) => row[i]! <= 0 || row.some((v) => !Number.isFinite(v)))) {
    throw new Error("Kovarians portfolio harus positive-definite setelah regularisasi.");
  }

  const muBar = new Array<number>(N).fill(0);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < N; j++) muBar[j]! += simpleR[t]![j]! / (T || 1);
  }
  // Shrink expected simple return toward 0 berdasarkan panjang sampel agar
  // estimasi mean tidak mendorong bobot ekstrem pada sampel pendek.
  const years = (T || 0) / barsPerYear;
  const w = shrinkageWeight(years);
  const mu = muBar.map((m) => m * barsPerYear * w);

  // PSD projection menghilangkan eigenvalue negatif; ridge kemudian memberi
  // margin numerik tambahan agar inverse tidak sensitif terhadap eigenvalue kecil.
  const Sinv = inverse(ridge(cov, 1e-4)) ?? inverse(ridge(cov, 1e-2));
  if (!Sinv) throw new Error("Matriks kovarians tidak dapat diinversi.");
  if (Sinv.some((row) => row.some((v) => !Number.isFinite(v)))) throw new Error("Inverse covariance menghasilkan NaN/Infinity.");
  const { A, B, C, D } = scalars(Sinv, mu);
  if (![A, B, C, D].every(Number.isFinite) || A <= 0 || D <= 0) {
    throw new Error("Kovarians terlalu degeneratif untuk efficient frontier.");
  }
  const ones = new Array<number>(N).fill(1);
  const wGmv = matVec(Sinv, ones).map((v) => v / A);
  const gmv: FrontierPoint = {
    weights: wGmv,
    ret: dot(wGmv, mu),
    risk: Math.sqrt(Math.max(quadForm(cov, wGmv), 0)),
  };

  const rGmv = B / A;
  const rSpan = Math.max(Math.abs(rGmv), Math.max(...mu.map(Math.abs)), 0.2);
  const rMin = rGmv - 0.6 * rSpan;
  const rMax = rGmv + 0.6 * rSpan;
  const frontier: FrontierPoint[] = [];
  for (let i = 0; i < points; i++) {
    const target = rMin + ((rMax - rMin) * i) / (points - 1);
    const w = efficientWeights(Sinv, mu, target);
    frontier.push({
      ret: target,
      risk: Math.sqrt(Math.max(quadForm(cov, w), 0)),
      weights: w,
    });
  }

  // Portofolio tangency: w ∝ Σ⁻¹(μ − r_f·1).
  // Valid hanya bila 1ᵀΣ⁻¹(μ − r_f·1) > 0; jika ≤ 0 tangency tidak eksis
  // (tidak ada portofolio dengan excess return positif yang bisa dinormalisasi).
  const excess = mu.map((m) => m - riskFree);
  const raw = matVec(Sinv, excess);
  const s = raw.reduce((a, b) => a + b, 0);
  let tangency: FrontierPoint | null = null;
  let maxSharpe = 0;
  if (s > 1e-12) {
    const w = raw.map((v) => v / s);
    const ret = dot(w, mu);
    const risk = Math.sqrt(Math.max(quadForm(cov, w), 0));
    tangency = { weights: w, ret, risk };
    maxSharpe = risk > 0 ? (ret - riskFree) / risk : 0;
  }

  const longOnly = maxSharpeLongOnly(cov, mu, riskFree);

  const result = { cov, corr, mu, frontier, gmv, tangency, maxSharpe, longOnly, riskFree };
  const flat = [
    ...cov.flat(), ...corr.flat(), ...mu,
    ...frontier.flatMap((p) => [p.ret, p.risk, ...p.weights]),
    ...gmv.weights, gmv.ret, gmv.risk,
    ...(tangency ? [...tangency.weights, tangency.ret, tangency.risk] : []),
    ...longOnly.weights, longOnly.ret, longOnly.risk, longOnly.sharpe, maxSharpe,
  ];
  if (flat.some((v) => !Number.isFinite(v))) throw new Error("Portfolio menghasilkan NaN/Infinity.");
  return result;
}

/** Proyeksi Euclidean ke simpleks {w ≥ 0, Σw = 1} (algoritma Duchi et al.). */
export function projectSimplex(v: number[]): number[] {
  const n = v.length;
  const u = [...v].sort((a, b) => b - a);
  let cssv = 0;
  let rho = 0;
  let theta = 0;
  for (let i = 0; i < n; i++) {
    cssv += u[i]!;
    const t = (cssv - 1) / (i + 1);
    if (u[i]! - t > 0) {
      rho = i + 1;
      theta = t;
    }
  }
  if (rho === 0) return new Array<number>(n).fill(1 / n);
  return v.map((x) => Math.max(x - theta, 0));
}

/**
 * Max-Sharpe long-only (tanpa short, fully invested) via projected gradient ascent:
 *   maksimalkan f(w) = wᵀ(μ − r_f) / √(wᵀΣw) pada simpleks.
 *   ∇f = (μ − r_f)/σ − (wᵀ(μ − r_f))·Σw/σ³
 */
export function maxSharpeLongOnly(
  cov: Matrix,
  mu: number[],
  riskFree: number,
  iters = 800,
): FrontierPoint & { sharpe: number } {
  const n = mu.length;
  if (n === 0 || cov.length !== n || cov.some((r) => r.length !== n)) {
    throw new Error("Dimensi covariance/return portfolio tidak cocok.");
  }
  if (![riskFree, ...mu, ...cov.flat()].every(Number.isFinite)) {
    throw new Error("Portfolio long-only menerima nilai non-finite.");
  }

  const excess = mu.map((m) => m - riskFree);
  const sharpeOf = (x: number[]) => {
    const sd2 = quadForm(cov, x);
    if (!(sd2 > 0) || !Number.isFinite(sd2)) return -Infinity;
    return dot(x, excess) / Math.sqrt(sd2);
  };

  // Untuk jumlah aset kecil, enumerasi semua support memberi solusi global:
  // pada setiap face simpleks, optimum Sharpe interior memenuhi
  // w ∝ Σ⁻¹(μ-r_f 1). Memeriksa semua face menangani optimum di boundary
  // tanpa bergantung pada titik awal gradient ascent.
  const exactLimit = 12;
  let bestW = new Array<number>(n).fill(1 / n);
  let best = sharpeOf(bestW);

  if (n <= exactLimit) {
    const totalMasks = 1 << n;
    for (let mask = 1; mask < totalMasks; mask++) {
      const idx: number[] = [];
      for (let i = 0; i < n; i++) if (mask & (1 << i)) idx.push(i);

      const sub = idx.map((i) => idx.map((j) => cov[i]![j]!));
      const subExcess = idx.map((i) => excess[i]!);
      const Sinv = inverse(sub);
      if (!Sinv) continue;
      const raw = matVec(Sinv, subExcess);
      const denom = raw.reduce((a, b) => a + b, 0);
      if (!(denom > 1e-14) || !Number.isFinite(denom)) continue;

      const wSub = raw.map((v) => v / denom);
      if (wSub.some((v) => v < -1e-9 || !Number.isFinite(v))) continue;
      const w = new Array<number>(n).fill(0);
      idx.forEach((asset, k) => { w[asset] = Math.max(wSub[k]!, 0); });
      const sum = w.reduce((a, b) => a + b, 0);
      if (!(sum > 0)) continue;
      for (let i = 0; i < n; i++) w[i] = w[i]! / sum;
      const val = sharpeOf(w);
      if (val > best) {
        best = val;
        bestW = w;
      }
    }
  } else {
    // Fallback untuk jumlah aset besar: multi-start projected gradient.
    // Beberapa deterministic starts mengurangi ketergantungan pada titik awal.
    const starts: number[][] = [
      new Array<number>(n).fill(1 / n),
      projectSimplex(excess.map((e) => Math.max(e, 0))),
      ...Array.from({ length: Math.min(8, n) }, (_, k) => {
        const w = new Array<number>(n).fill(0);
        w[k] = 1;
        return w;
      }),
    ];

    for (const initial of starts) {
      let w = projectSimplex(initial);
      let step = 0.05;
      let localBest = sharpeOf(w);
      for (let k = 0; k < iters; k++) {
        const Sw = matVec(cov, w);
        const sd2 = quadForm(cov, w);
        if (!(sd2 > 0)) break;
        const sd = Math.sqrt(sd2);
        const ex = dot(w, excess);
        const grad = excess.map((e, i) => e / sd - (ex * Sw[i]!) / (sd2 * sd));
        const cand = projectSimplex(w.map((v, i) => v + step * grad[i]!));
        const val = sharpeOf(cand);
        if (val > localBest + 1e-12) {
          localBest = val;
          w = cand;
        } else {
          step *= 0.75;
          if (step < 1e-9) break;
        }
      }
      if (localBest > best) {
        best = localBest;
        bestW = w;
      }
    }
  }

  return {
    weights: bestW,
    ret: dot(bestW, mu),
    risk: Math.sqrt(Math.max(quadForm(cov, bestW), 0)),
    sharpe: best,
  };
}

/**
 * Principal Component Analysis (PCA) pada matriks return T×N.
 *
 * Return:
 * - scores3d: proyeksi setiap observasi ke PC1/PC2/PC3.
 * - explained: proporsi variance yang dijelaskan tiap PC.
 * - eigenvalues: eigenvalue PCA, terurut menurun.
 * - loadings: eigenvector utama (aset × PC).
 *
 * PCA menggunakan return yang distandarisasi sehingga aset dengan skala
 * volatilitas berbeda tidak mendominasi ruang eigen hanya karena unitnya.
 */
export function pca(
  R: Matrix,
  components = 3,
): {
  scores3d: { x: number; y: number; z: number }[];
  scores: Matrix;
  loadings: Matrix;
  eigenvalues: number[];
  explained: number[];
} {
  const T = R.length;
  const N = T > 0 ? R[0]!.length : 0;
  if (T < 2 || N === 0) throw new Error("PCA membutuhkan minimal 2 observasi dan 1 aset.");
  if (!Number.isInteger(components) || components < 1) {
    throw new Error("Jumlah komponen PCA harus integer >= 1.");
  }
  if (R.some((row) => row.length !== N || row.some((v) => !Number.isFinite(v)))) {
    throw new Error("Matriks return PCA tidak valid atau mengandung NaN/Infinity.");
  }

  const means = new Array<number>(N).fill(0);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < N; j++) means[j]! += R[t]![j]! / T;
  }

  const stds = new Array<number>(N).fill(0);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < N; j++) {
      const d = R[t]![j]! - means[j]!;
      stds[j]! += d * d;
    }
  }
  for (let j = 0; j < N; j++) {
    stds[j] = Math.sqrt(stds[j]! / Math.max(T - 1, 1));
    if (!(stds[j]! > 0) || !Number.isFinite(stds[j]!)) {
      throw new Error(`PCA tidak dapat dilakukan: variance aset ke-${j + 1} nol/degeneratif.`);
    }
  }

  const Z: Matrix = R.map((row) =>
    row.map((v, j) => (v - means[j]!) / stds[j]!),
  );

  const corr = covarianceMatrix(Z);
  const eig = jacobiEigen(corr);
  const total = eig.values.reduce((sum, v) => sum + Math.max(v, 0), 0);
  if (!(total > 0) || !Number.isFinite(total)) {
    throw new Error("PCA menghasilkan total eigenvalue yang tidak valid.");
  }

  const k = Math.min(components, N);
  const eigenvalues = eig.values.slice(0, k).map((v) => Math.max(v, 0));
  const explained = eigenvalues.map((v) => v / total);
  const loadings: Matrix = Array.from({ length: N }, (_, i) =>
    Array.from({ length: k }, (_, c) => eig.vectors[i]![c]!),
  );

  const scores: Matrix = Z.map((row) =>
    Array.from({ length: k }, (_, c) => {
      let s = 0;
      for (let j = 0; j < N; j++) s += row[j]! * loadings[j]![c]!;
      return s;
    }),
  );

  if ([...eigenvalues, ...explained, ...loadings.flat(), ...scores.flat()].some((v) => !Number.isFinite(v))) {
    throw new Error("PCA menghasilkan NaN/Infinity.");
  }

  const scores3d = scores.map((row) => ({
    x: row[0] ?? 0,
    y: row[1] ?? 0,
    z: row[2] ?? 0,
  }));

  return { scores3d, scores, loadings, eigenvalues, explained };
}
