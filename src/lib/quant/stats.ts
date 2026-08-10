/**
 * Primitif statistik & aljabar linear.
 * Semua fungsi murni (pure) dan dapat diuji.
 */

export function mean(x: number[]): number {
  if (x.length === 0) return 0;
  let s = 0;
  for (const v of x) s += v;
  return s / x.length;
}

/** Varians sampel (unbiased, pembagi n-1). */
export function variance(x: number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  const m = mean(x);
  let s = 0;
  for (const v of x) s += (v - m) * (v - m);
  return s / (n - 1);
}

export function stdev(x: number[]): number {
  return Math.sqrt(variance(x));
}

export function skewness(x: number[]): number {
  const n = x.length;
  if (n < 3) return 0;
  const m = mean(x);
  const s = stdev(x);
  if (s === 0) return 0;
  let acc = 0;
  for (const v of x) acc += ((v - m) / s) ** 3;
  // Adjusted Fisher-Pearson standardized moment coefficient G1
  return (n / ((n - 1) * (n - 2))) * acc;
}

/** Excess kurtosis (G2, unbiased). Normal => 0. */
export function kurtosis(x: number[]): number {
  const n = x.length;
  if (n < 4) return 0;
  const m = mean(x);
  const s = stdev(x);
  if (s === 0) return 0;
  let acc = 0;
  for (const v of x) acc += ((v - m) / s) ** 4;
  const g2 = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * acc;
  return g2 - (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
}

export function quantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0]!;
  const h = (sorted.length - 1) * p;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  return sorted[lo]! + (h - lo) * (sorted[hi]! - sorted[lo]!);
}

/**
 * Bobot credibility shrinkage James-Stein-lite untuk estimasi mean pada sampel
 * pendek: w = years / (years + tau). tau = "tahun ekuivalen" yang dibutuhkan
 * untuk mendekati bobot penuh (default 2 tahun). Dipakai untuk meredam drift
 * (μ) tahunan yang meledak saat data historis hanya berupa beberapa bulan lalu
 * dianualisasi — bukan mengubah σ (yang jauh lebih stabil diestimasi).
 */
export function shrinkageWeight(years: number, tau = 2): number {
  if (!Number.isFinite(years) || years <= 0) return 0;
  return years / (years + tau);
}

/** Log-return: r_t = ln(P_t / P_{t-1}) */
export function logReturns(prices: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    const a = prices[i - 1]!;
    const b = prices[i]!;
    if (a > 0 && b > 0) out.push(Math.log(b / a));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Distribusi normal
// ---------------------------------------------------------------------------

/** erf via Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7). */
export function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

/** CDF normal standar Φ(x). */
export function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/** PDF normal standar φ(x). */
export function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/** PDF normal umum N(mu, sigma^2). */
export function gaussPdf(x: number, mu: number, sigma: number): number {
  const s = Math.max(sigma, 1e-12);
  const z = (x - mu) / s;
  return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
}

// ---------------------------------------------------------------------------
// RNG deterministik (agar render Monte Carlo stabil antara SSR & klien)
// ---------------------------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller: 1 sampel normal standar dari 2 uniform. */
export function randNorm(rng: () => number): number {
  let u = rng();
  if (u < 1e-12) u = 1e-12;
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------------------------------------------------------------------------
// Matriks
// ---------------------------------------------------------------------------

export type Matrix = number[][];

export function zeros(n: number, m: number): Matrix {
  return Array.from({ length: n }, () => new Array<number>(m).fill(0));
}

export function identity(n: number): Matrix {
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i]![i] = 1;
  return I;
}

export function matVec(A: Matrix, v: number[]): number[] {
  return A.map((row) => row.reduce((s, a, j) => s + a * v[j]!, 0));
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/** Bentuk kuadratik vᵀ A v */
export function quadForm(A: Matrix, v: number[]): number {
  return dot(v, matVec(A, v));
}

/**
 * Matriks kovarians sampel: Σ = 1/(T-1) · (R - R̄)ᵀ (R - R̄)
 * @param R matriks T×N (baris = waktu, kolom = aset)
 */
export function covarianceMatrix(R: Matrix): Matrix {
  const T = R.length;
  const N = T > 0 ? R[0]!.length : 0;
  const S = zeros(N, N);
  if (T < 2) return S;
  const mu = new Array<number>(N).fill(0);
  for (let t = 0; t < T; t++) for (let j = 0; j < N; j++) mu[j]! += R[t]![j]! / T;
  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += (R[t]![i]! - mu[i]!) * (R[t]![j]! - mu[j]!);
      const v = s / (T - 1);
      S[i]![j] = v;
      S[j]![i] = v;
    }
  }
  return S;
}

export function correlationFromCov(S: Matrix): Matrix {
  const n = S.length;
  const C = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const d = Math.sqrt(Math.max(S[i]![i]!, 1e-18) * Math.max(S[j]![j]!, 1e-18));
      C[i]![j] = S[i]![j]! / d;
    }
  }
  return C;
}

/** Inverse via Gauss-Jordan dengan partial pivoting. null jika singular. */
export function inverse(A: Matrix): Matrix | null {
  const n = A.length;
  const M = A.map((r, i) => [...r, ...identity(n)[i]!]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r]![col]!) > Math.abs(M[piv]![col]!)) piv = r;
    }
    if (Math.abs(M[piv]![col]!) < 1e-14) return null;
    if (piv !== col) {
      const tmp = M[piv]!;
      M[piv] = M[col]!;
      M[col] = tmp;
    }
    const p = M[col]![col]!;
    for (let c = 0; c < 2 * n; c++) M[col]![c]! /= p;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r]![col]!;
      if (f === 0) continue;
      for (let c = 0; c < 2 * n; c++) M[r]![c]! -= f * M[col]![c]!;
    }
  }
  return M.map((r) => r.slice(n));
}

/** Ridge shrinkage: Σ + λ·diag rata-rata. Menjamin invertibilitas. */
export function ridge(S: Matrix, lambda = 1e-6): Matrix {
  const n = S.length;
  let tr = 0;
  for (let i = 0; i < n; i++) tr += S[i]![i]!;
  const avg = n > 0 ? tr / n : 1;
  const out = S.map((r) => [...r]);
  for (let i = 0; i < n; i++) out[i]![i]! += lambda * (avg || 1);
  return out;
}

/**
 * Eigen-decomposition matriks simetrik dengan rotasi Jacobi siklik.
 * Mengembalikan eigenvalue terurut menurun + eigenvector kolom-terkait.
 */
export function jacobiEigen(
  input: Matrix,
  maxSweeps = 100,
): { values: number[]; vectors: Matrix } {
  const n = input.length;
  const A = input.map((r) => [...r]);
  let V = identity(n);

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) off += A[i]![j]! * A[i]![j]!;
    if (off < 1e-20) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p]![q]!;
        if (Math.abs(apq) < 1e-18) continue;
        const theta = (A[q]![q]! - A[p]![p]!) / (2 * apq);
        const t =
          Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k]![p]!;
          const akq = A[k]![q]!;
          A[k]![p] = c * akp - s * akq;
          A[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p]![k]!;
          const aqk = A[q]![k]!;
          A[p]![k] = c * apk - s * aqk;
          A[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k]![p]!;
          const vkq = V[k]![q]!;
          V[k]![p] = c * vkp - s * vkq;
          V[k]![q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const pairs = Array.from({ length: n }, (_, i) => ({
    value: A[i]![i]!,
    vector: V.map((row) => row[i]!),
  })).sort((a, b) => b.value - a.value);

  V = zeros(n, n);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) V[i]![j] = pairs[j]!.vector[i]!;

  return { values: pairs.map((p) => p.value), vectors: V };
}

/** Regresi linear sederhana y = a + b·x (OLS). */
export function ols(x: number[], y: number[]): { a: number; b: number; resid: number[] } {
  const n = Math.min(x.length, y.length);
  if (n < 2) return { a: 0, b: 0, resid: [] };
  const mx = mean(x.slice(0, n));
  const my = mean(y.slice(0, n));
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i]! - mx) * (y[i]! - my);
    sxx += (x[i]! - mx) * (x[i]! - mx);
  }
  const b = sxx === 0 ? 0 : sxy / sxx;
  const a = my - b * mx;
  const resid: number[] = [];
  for (let i = 0; i < n; i++) resid.push(y[i]! - (a + b * x[i]!));
  return { a, b, resid };
}
