/**
 * Hidden Markov Model dengan emisi Gaussian univariat.
 * Estimasi parameter: Baum-Welch (EM) memakai forward-backward berskala.
 * Dekode state: Viterbi (log-space).
 */

import { gaussPdf, mean, stdev } from "./stats";

export interface HmmParams {
  /** π: probabilitas awal, panjang K */
  pi: number[];
  /** A: matriks transisi K×K, A[i][j] = P(state j pada t | state i pada t-1) */
  A: number[][];
  /** μ emisi per state */
  mu: number[];
  /** σ emisi per state */
  sigma: number[];
}

export interface HmmFit {
  params: HmmParams;
  /** γ_t(j) = P(state j | seluruh observasi) */
  gamma: number[][];
  /** jalur state paling mungkin (Viterbi) */
  viterbi: number[];
  logLikelihood: number;
  iterations: number;
  /** true jika kriteria konvergensi tercapai sebelum maxIter. */
  converged: boolean;
  /** log-likelihood pada setiap E-step, untuk audit monotonicity EM. */
  logLikelihoodHistory: number[];
}

/**
 * Canonical state ordering: lowest emission mean first (Bear → Bull).
 *
 * HMM state IDs are intrinsically arbitrary: a valid EM solution can call
 * the same latent state 0, 1, or 2.  Never expose raw EM indices as regime
 * identities.  We therefore canonicalize by emission mean after every fit.
 * Sigma is only a deterministic tie-breaker when means are indistinguishable.
 */
export function canonicalizeHmmParams(p: HmmParams): HmmParams {
  const K = p.pi.length;
  const order = p.mu
    .map((mu, i) => ({ mu, sigma: p.sigma[i] ?? Infinity, i }))
    .sort((a, b) => a.mu - b.mu || a.sigma - b.sigma || a.i - b.i)
    .map((x) => x.i);

  return {
    pi: order.map((i) => p.pi[i]!),
    A: order.map((i) => order.map((j) => p.A[i]![j]!)),
    mu: order.map((i) => p.mu[i]!),
    sigma: order.map((i) => p.sigma[i]!),
  };
}

/** Inisialisasi state dengan memotong data pada kuantil (deterministik). */
function initParams(obs: number[], K: number): HmmParams {
  const sorted = [...obs].sort((a, b) => a - b);
  const mu: number[] = [];
  const sigma: number[] = [];
  const chunk = Math.max(1, Math.floor(sorted.length / K));
  const globalSd = Math.max(stdev(obs), 1e-8);
  for (let k = 0; k < K; k++) {
    const part = sorted.slice(k * chunk, k === K - 1 ? sorted.length : (k + 1) * chunk);
    mu.push(part.length ? mean(part) : 0);
    sigma.push(Math.max(part.length > 1 ? stdev(part) : globalSd, globalSd * 0.25));
  }
  const A = Array.from({ length: K }, (_, i) =>
    Array.from({ length: K }, (_, j) => (i === j ? 0.9 : 0.1 / (K - 1))),
  );
  return { pi: new Array<number>(K).fill(1 / K), A, mu, sigma };
}

function emission(params: HmmParams, o: number, j: number): number {
  return Math.max(gaussPdf(o, params.mu[j]!, params.sigma[j]!), 1e-300);
}

/**
 * Forward algorithm berskala.
 * α̂_t(j) ∝ [Σ_i α̂_{t-1}(i) A_ij] · B_j(o_t), dengan c_t faktor normalisasi.
 * logL = −Σ_t ln c_t.
 */
function forward(obs: number[], p: HmmParams) {
  const T = obs.length;
  const K = p.pi.length;
  const alpha = Array.from({ length: T }, () => new Array<number>(K).fill(0));
  const c = new Array<number>(T).fill(1);

  let s0 = 0;
  for (let j = 0; j < K; j++) {
    alpha[0]![j] = p.pi[j]! * emission(p, obs[0]!, j);
    s0 += alpha[0]![j]!;
  }
  c[0] = s0 > 0 ? 1 / s0 : 1;
  for (let j = 0; j < K; j++) alpha[0]![j]! *= c[0]!;

  for (let t = 1; t < T; t++) {
    let st = 0;
    for (let j = 0; j < K; j++) {
      let acc = 0;
      for (let i = 0; i < K; i++) acc += alpha[t - 1]![i]! * p.A[i]![j]!;
      alpha[t]![j] = acc * emission(p, obs[t]!, j);
      st += alpha[t]![j]!;
    }
    c[t] = st > 0 ? 1 / st : 1;
    for (let j = 0; j < K; j++) alpha[t]![j]! *= c[t]!;
  }

  let logL = 0;
  for (let t = 0; t < T; t++) logL -= Math.log(c[t]!);
  return { alpha, c, logL };
}

/** Backward berskala memakai faktor c_t yang sama dengan forward. */
function backward(obs: number[], p: HmmParams, c: number[]) {
  const T = obs.length;
  const K = p.pi.length;
  const beta = Array.from({ length: T }, () => new Array<number>(K).fill(0));
  for (let j = 0; j < K; j++) beta[T - 1]![j] = c[T - 1]!;
  for (let t = T - 2; t >= 0; t--) {
    for (let i = 0; i < K; i++) {
      let acc = 0;
      for (let j = 0; j < K; j++)
        acc += p.A[i]![j]! * emission(p, obs[t + 1]!, j) * beta[t + 1]![j]!;
      beta[t]![i] = acc * c[t]!;
    }
  }
  return beta;
}

/** Viterbi di ruang log untuk stabilitas numerik. */
export function viterbi(obs: number[], p: HmmParams): number[] {
  const T = obs.length;
  const K = p.pi.length;
  const LOG0 = -1e300;
  const ln = (v: number) => (v > 0 ? Math.log(v) : LOG0);

  const delta = Array.from({ length: T }, () => new Array<number>(K).fill(LOG0));
  const psi = Array.from({ length: T }, () => new Array<number>(K).fill(0));

  for (let j = 0; j < K; j++) delta[0]![j] = ln(p.pi[j]!) + ln(emission(p, obs[0]!, j));

  for (let t = 1; t < T; t++) {
    for (let j = 0; j < K; j++) {
      let best = -Infinity;
      let arg = 0;
      for (let i = 0; i < K; i++) {
        const v = delta[t - 1]![i]! + ln(p.A[i]![j]!);
        if (v > best) {
          best = v;
          arg = i;
        }
      }
      delta[t]![j] = best + ln(emission(p, obs[t]!, j));
      psi[t]![j] = arg;
    }
  }

  const path = new Array<number>(T).fill(0);
  let best = -Infinity;
  for (let j = 0; j < K; j++) {
    if (delta[T - 1]![j]! > best) {
      best = delta[T - 1]![j]!;
      path[T - 1] = j;
    }
  }
  for (let t = T - 2; t >= 0; t--) path[t] = psi[t + 1]![path[t + 1]!]!;
  return path;
}

/** Baum-Welch. K = jumlah regime tersembunyi. */
export function fitHmm(obs: number[], K = 3, maxIter = 120, tol = 1e-7): HmmFit {
  if (!Number.isInteger(K) || K < 2) throw new Error("HMM membutuhkan K integer >= 2.");
  if (!Number.isInteger(maxIter) || maxIter < 1) throw new Error("maxIter harus integer >= 1.");
  if (!Number.isFinite(tol) || tol <= 0) throw new Error("tol harus finite dan > 0.");
  if (obs.some((v) => !Number.isFinite(v))) throw new Error("Observasi HMM harus finite.");
  const T = obs.length;
  if (T < K * 5) {
    const p = initParams(obs.length ? obs : [0], K);
    return {
      params: p,
      gamma: Array.from({ length: T }, () => new Array<number>(K).fill(1 / K)),
      viterbi: new Array<number>(T).fill(0),
      logLikelihood: 0,
      iterations: 0,
      converged: false,
      logLikelihoodHistory: [],
    };
  }

  let p = initParams(obs, K);
  let prevLogL = -Infinity;
  let gamma: number[][] = [];
  let iterations = 0;
  let logL = -Infinity;
  let converged = false;
  const logLikelihoodHistory: number[] = [];
  const llDecreaseTol = Math.max(tol * 10, 1e-10);

  for (let iter = 0; iter < maxIter; iter++) {
    iterations = iter + 1;
    const { alpha, c, logL: ll } = forward(obs, p);
    const beta = backward(obs, p, c);
    logL = ll;
    if (!Number.isFinite(logL)) throw new Error("HMM menghasilkan log-likelihood non-finite.");
    logLikelihoodHistory.push(logL);
    if (Number.isFinite(prevLogL) && logL < prevLogL - llDecreaseTol * Math.max(1, Math.abs(prevLogL))) {
      throw new Error("HMM EM tidak monoton: log-likelihood turun di luar toleransi numerik.");
    }

    // E-step: γ_t(j) ∝ α̂_t(j)·β̂_t(j)
    gamma = Array.from({ length: T }, () => new Array<number>(K).fill(0));
    for (let t = 0; t < T; t++) {
      let s = 0;
      for (let j = 0; j < K; j++) {
        gamma[t]![j] = alpha[t]![j]! * beta[t]![j]!;
        s += gamma[t]![j]!;
      }
      if (s > 0) for (let j = 0; j < K; j++) gamma[t]![j]! /= s;
      else for (let j = 0; j < K; j++) gamma[t]![j] = 1 / K;
    }

    // ξ terakumulasi: ξ_t(i,j) ∝ α̂_t(i)·A_ij·B_j(o_{t+1})·β̂_{t+1}(j)
    const xiSum = Array.from({ length: K }, () => new Array<number>(K).fill(0));
    for (let t = 0; t < T - 1; t++) {
      let s = 0;
      const tmp = Array.from({ length: K }, () => new Array<number>(K).fill(0));
      for (let i = 0; i < K; i++) {
        for (let j = 0; j < K; j++) {
          const v =
            alpha[t]![i]! * p.A[i]![j]! * emission(p, obs[t + 1]!, j) * beta[t + 1]![j]!;
          tmp[i]![j] = v;
          s += v;
        }
      }
      if (s > 0)
        for (let i = 0; i < K; i++)
          for (let j = 0; j < K; j++) xiSum[i]![j]! += tmp[i]![j]! / s;
    }

    // M-step
    const pi = gamma[0]!.slice();
    const A = Array.from({ length: K }, () => new Array<number>(K).fill(0));
    for (let i = 0; i < K; i++) {
      let denom = 0;
      for (let t = 0; t < T - 1; t++) denom += gamma[t]![i]!;
      for (let j = 0; j < K; j++) {
        A[i]![j] = denom > 0 ? xiSum[i]![j]! / denom : (i === j ? 1 : 0);
      }
      const rowSum = A[i]!.reduce((a, b) => a + b, 0);
      if (rowSum > 0) for (let j = 0; j < K; j++) A[i]![j]! /= rowSum;
    }

    const mu = new Array<number>(K).fill(0);
    const sigma = new Array<number>(K).fill(0);
    const floor = Math.max(stdev(obs) * 0.05, 1e-8);
    for (let j = 0; j < K; j++) {
      let w = 0;
      let num = 0;
      for (let t = 0; t < T; t++) {
        w += gamma[t]![j]!;
        num += gamma[t]![j]! * obs[t]!;
      }
      mu[j] = w > 0 ? num / w : p.mu[j]!;
      let v = 0;
      for (let t = 0; t < T; t++) v += gamma[t]![j]! * (obs[t]! - mu[j]!) ** 2;
      sigma[j] = Math.max(w > 0 ? Math.sqrt(v / w) : p.sigma[j]!, floor);
    }

    p = { pi, A, mu, sigma };
    if (Number.isFinite(prevLogL) && Math.abs(logL - prevLogL) < tol * Math.max(1, Math.abs(prevLogL))) {
      converged = true;
      prevLogL = logL;
      break;
    }
    prevLogL = logL;
  }

  // Canonicalize state IDs because HMM labels are arbitrary.
  // This prevents numerical/EM label switching from changing Bear/Sideways/Bull
  // merely because the optimizer happened to name equivalent states differently.
  const sorted = canonicalizeHmmParams(p);
  // Validate the final EM solution before exposing it. EM can collapse a
  // state or produce a numerically degenerate transition row on pathological
  // samples; fail closed instead of returning a plausible-looking regime.
  const rowSumsOk = sorted.A.every((row) => {
    const sum = row.reduce((a, b) => a + b, 0);
    return row.every((v) => Number.isFinite(v) && v >= 0) && Math.abs(sum - 1) < 1e-8;
  });
  const piSum = sorted.pi.reduce((a, b) => a + b, 0);
  if (!rowSumsOk || sorted.pi.some((v) => !Number.isFinite(v) || v < 0) || Math.abs(piSum - 1) >= 1e-8 ||
      sorted.mu.some((v) => !Number.isFinite(v)) || sorted.sigma.some((v) => !Number.isFinite(v) || v <= 0)) {
    throw new Error("HMM menghasilkan parameter akhir yang tidak valid.");
  }
  // Recompute posterior probabilities after the final M-step and state sort.
  // Sebelumnya `gamma` masih berasal dari parameter sebelum M-step terakhir,
  // sehingga regime probabilities yang dikonsumsi UI/backtest tidak persis
  // konsisten dengan `params` dan Viterbi final.
  const finalForward = forward(obs, sorted);
  const finalBackward = backward(obs, sorted, finalForward.c);
  const sortedGamma = Array.from({ length: T }, () => new Array<number>(K).fill(0));
  for (let t = 0; t < T; t++) {
    let s = 0;
    for (let j = 0; j < K; j++) {
      sortedGamma[t]![j] = finalForward.alpha[t]![j]! * finalBackward[t]![j]!;
      s += sortedGamma[t]![j]!;
    }
    if (s > 0) {
      for (let j = 0; j < K; j++) sortedGamma[t]![j]! /= s;
    } else {
      for (let j = 0; j < K; j++) sortedGamma[t]![j] = 1 / K;
    }
  }
  const path = viterbi(obs, sorted);

  return {
    params: sorted,
    gamma: sortedGamma,
    viterbi: path,
    logLikelihood: finalForward.logL,
    iterations,
    converged,
    logLikelihoodHistory,
  };
}

export function regimeLabel(k: number, K: number): "Bear" | "Sideways" | "Bull" {
  if (K <= 2) return k === 0 ? "Bear" : "Bull";
  if (k === 0) return "Bear";
  if (k === K - 1) return "Bull";
  return "Sideways";
}
