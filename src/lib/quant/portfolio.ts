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
  const covBar = covarianceMatrix(R);
  // Anualisasi: Σ_annual = Σ_bar · barsPerYear ; μ_annual = μ_bar · barsPerYear
  const cov = covBar.map((row) => row.map((v) => v * barsPerYear));
  const corr = correlationFromCov(cov);

  const muBar = new Array<number>(N).fill(0);
  for (let t = 0; t < T; t++) for (let j = 0; j < N; j++) muBar[j]! += R[t]![j]! / (T || 1);
  // μ aritmetik tahunan dari log-return: μ = m·bpy + σ²/2
  const mu = muBar.map((m, j) => m * barsPerYear + 0.5 * cov[j]![j]!);

  const Sinv = inverse(ridge(cov, 1e-4)) ?? inverse(ridge(cov, 1e-2))!;
  const { A, B } = scalars(Sinv, mu);
  const ones = new Array<number>(N).fill(1);
  const wGmv = matVec(Sinv, ones).map((v) => v / (A || 1));
  const gmv: FrontierPoint = {
    weights: wGmv,
    ret: dot(wGmv, mu),
    risk: Math.sqrt(Math.max(quadForm(cov, wGmv), 0)),
  };

  const rMin = B / A - 0.6 * Math.max(Math.abs(B / A), 0.2);
  const rMax = Math.max(...mu, B / A) * 1.15;
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

  // Portofolio tangency: w ∝ Σ⁻¹(μ − r_f·1)
  const excess = mu.map((m) => m - riskFree);
  const raw = matVec(Sinv, excess);
  const s = raw.reduce((a, b) => a + b, 0);
  let tangency: FrontierPoint | null = null;
  let maxSharpe = 0;
  if (Math.abs(s) > 1e-12) {
    const w = raw.map((v) => v / s);
    const ret = dot(w, mu);
    const risk = Math.sqrt(Math.max(quadForm(cov, w), 0));
    tangency = { weights: w, ret, risk };
    maxSharpe = risk > 0 ? (ret - riskFree) / risk : 0;
  }

  return { cov, corr, mu, frontier, gmv, tangency, maxSharpe, riskFree };
}

export interface PcaResult {
  /** varians tiap principal component (eigenvalue) */
  eigenvalues: number[];
  /** proporsi varians dijelaskan */
  explained: number[];
  /** loading: kolom j = eigenvector ke-j */
  loadings: Matrix;
  /** skor observasi pada 3 PC pertama */
  scores3d: { x: number; y: number; z: number }[];
}

/**
 * PCA atas matriks korelasi (return distandarisasi) — Σv = λv.
 */
export function pca(R: Matrix): PcaResult {
  const T = R.length;
  const N = T > 0 ? R[0]!.length : 0;
  const cov = covarianceMatrix(R);
  const corr = correlationFromCov(cov);
  const { values, vectors } = jacobiEigen(corr);
  const total = values.reduce((a, b) => a + Math.max(b, 0), 0) || 1;

  const mu = new Array<number>(N).fill(0);
  for (let t = 0; t < T; t++) for (let j = 0; j < N; j++) mu[j]! += R[t]![j]! / (T || 1);
  const sd = Array.from({ length: N }, (_, j) => Math.sqrt(Math.max(cov[j]![j]!, 1e-18)));

  const scores3d = R.map((row) => {
    const z = row.map((v, j) => (v - mu[j]!) / sd[j]!);
    const proj = (k: number) =>
      k < N ? z.reduce((s, v, i) => s + v * vectors[i]![k]!, 0) : 0;
    return { x: proj(0), y: proj(1), z: proj(2) };
  });

  return {
    eigenvalues: values,
    explained: values.map((v) => Math.max(v, 0) / total),
    loadings: vectors,
    scores3d,
  };
}
