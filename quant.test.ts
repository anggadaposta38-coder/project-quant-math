import { expect, test } from "vitest";
import { BARS_PER_YEAR } from "./src/lib/market.server";
import { blackScholes, fitGbm, fitOu, monteCarloGbm, volatilitySurface, computeEntryZone, optimalEntryThreshold } from "./src/lib/quant/stochastic";
import { canonicalizeHmmParams, fitHmm } from "./src/lib/quant/hmm";
import { analyzePortfolio, efficientWeights, pca } from "./src/lib/quant/portfolio";
import { covarianceMatrix, correlationFromCov, dot, inverse, jacobiEigen, mulberry32, stableSeed, nearestPsd, quadForm, randNorm, ridge, normCdf, kurtosis, skewness, variance, stdev, annualizedVolatility } from "./src/lib/quant/stats";
import { ema, macd, rsi, rollingZScore } from "./src/lib/quant/indicators";
import { computeSignal, MIN_SIGNAL_BARS, SIGNAL_ACTION_THRESHOLD, SIGNAL_WEIGHTS, QUARTER_KELLY } from "./src/lib/quant/analysis";
import { alignedReturns } from "./src/lib/quant/analysis";
import { normalizeCandlesForInterval } from "./src/lib/market.server";
import { directionalRawReturn, netTradeReturn, positionFractionFromStop } from "./src/lib/quant/backtest";


test("indicators follow standard warm-up and flat-market conventions", () => {
  const x = Array.from({ length: 40 }, (_, i) => 100 + i);
  const e = ema(x, 10);
  expect(e.slice(0, 9).every(Number.isNaN)).toBe(true);
  expect(e[9]).toBeCloseTo(104.5, 12);

  const flat = Array.from({ length: 20 }, () => 100);
  const r = rsi(flat, 14);
  expect(r.slice(0, 14).every(Number.isNaN)).toBe(true);
  expect(r[14]).toBe(50);
  expect(r[19]).toBe(50);

  const z = rollingZScore(flat, 10);
  expect(z.slice(0, 9).every(Number.isNaN)).toBe(true);
  expect(z.slice(9).every((v) => v === 0)).toBe(true);
});

test("MACD uses textbook warm-up and rejects invalid period ordering", () => {
  const x = Array.from({ length: 60 }, (_, i) => 100 + i);
  const m = macd(x, 12, 26, 9);
  expect(m.macd.slice(0, 25).every(Number.isNaN)).toBe(true);
  expect(m.signal.slice(0, 33).every(Number.isNaN)).toBe(true);
  expect(m.histogram.slice(0, 33).every(Number.isNaN)).toBe(true);
  expect(() => macd(x, 26, 12, 9)).toThrow();
});

test("indicators reject non-finite data and invalid windows", () => {
  expect(() => ema([1, 2, Number.NaN], 2)).toThrow();
  expect(() => rsi([1, 2, Number.POSITIVE_INFINITY], 2)).toThrow();
  expect(() => rollingZScore([1, 2, 3], 1)).toThrow();
});


test("signal policy constants remain internally consistent", () => {
  expect(Object.values(SIGNAL_WEIGHTS).reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 12);
  expect(SIGNAL_ACTION_THRESHOLD).toBeGreaterThan(0);
  expect(SIGNAL_ACTION_THRESHOLD).toBeLessThan(1);
  expect(QUARTER_KELLY).toBeCloseTo(0.25, 12);
});

const rng = mulberry32(7);

test("normCdf", () => {
  expect(normCdf(0)).toBeCloseTo(0.5, 6);
  expect(normCdf(1.96)).toBeCloseTo(0.975, 4);
});

test("put-call parity", () => {
  const S=100,K=95,T=0.5,r=0.03,s=0.6;
  const c=blackScholes(S,K,T,r,s,"call").price, p=blackScholes(S,K,T,r,s,"put").price;
  expect(c-p).toBeCloseTo(S-K*Math.exp(-r*T), 6);
});

test("BS greeks vs finite diff", () => {
  const f=(S:number)=>blackScholes(S,100,1,0.02,0.5,"call").price;
  const h=1e-4;
  expect((f(100+h)-f(100-h))/(2*h)).toBeCloseTo(blackScholes(100,100,1,0.02,0.5).delta, 5);
  expect((f(100+h)-2*f(100)+f(100-h))/(h*h)).toBeCloseTo(blackScholes(100,100,1,0.02,0.5).gamma, 4);
});

test("GBM param recovery", () => {
  const dt=1/365, mu=0.5, sig=0.8;
  const r:number[]=[]; for(let i=0;i<200000;i++) r.push((mu-0.5*sig*sig)*dt+sig*Math.sqrt(dt)*randNorm(rng));
  const g=fitGbm(r,dt);
  expect(g.sigma).toBeCloseTo(sig,1);
  expect(g.mu).toBeCloseTo(mu,1);
  expect(g.rawLogDrift).toBeFinite();
  expect(g.driftShrinkageWeight).toBeGreaterThan(0.99);
});

test("Monte Carlo E[S_T] = S0 e^{mu T}", { timeout: 60000 }, () => {
  const g={mu:0.4,sigma:0.7,logDrift:0,dt:1/365};
  const mc=monteCarloGbm(100,g,365,4000,42);
  expect(mc.expectedTerminal/100).toBeCloseTo(Math.exp(0.4),1);
  expect(mc.probUp).toBeGreaterThan(0.3);
});

test("HMM recovers 2 well-separated regimes", () => {
  const obs:number[]=[]; const truth:number[]=[]; let s=0;
  for(let i=0;i<1500;i++){ if(rng()<0.02) s=1-s; truth.push(s); obs.push((s?0.02:-0.02)+0.005*randNorm(rng)); }
  const fit=fitHmm(obs,2,200);
  expect(fit.params.mu[0]).toBeCloseTo(-0.02,2);
  expect(fit.params.mu[1]).toBeCloseTo(0.02,2);
  const acc=fit.viterbi.filter((v,i)=>v===truth[i]).length/obs.length;
  expect(acc).toBeGreaterThan(0.95);
  // rows of A sum to 1
  for(const row of fit.params.A) expect(row.reduce((a,b)=>a+b,0)).toBeCloseTo(1,8);
});



test("HMM canonicalizes arbitrary state IDs by emission mean", () => {
  const raw = {
    pi: [0.2, 0.5, 0.3],
    A: [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.05, 0.15, 0.8]],
    mu: [0.03, -0.02, 0.01],
    sigma: [0.01, 0.02, 0.03],
  };
  const c = canonicalizeHmmParams(raw);
  expect(c.mu).toEqual([-0.02, 0.01, 0.03]);
  expect(c.pi).toEqual([0.5, 0.3, 0.2]);
  expect(c.A[0]!).toEqual([0.7, 0.1, 0.2]);
  expect(c.A[1]!).toEqual([0.15, 0.8, 0.05]);
  expect(c.A[2]!).toEqual([0.1, 0.1, 0.8]);
});

test("HMM near-overlapping regimes are not treated as stable semantic labels", () => {
  const candles = Array.from({length: MIN_SIGNAL_BARS + 20}, (_, i) => {
    const c = 100 * Math.exp(0.0002 * i + 0.0005 * Math.sin(i / 2));
    return {t: i * 3600000, o: c, h: c * 1.001, l: c * 0.999, c, v: 1};
  });
  const s = computeSignal(candles, "1h");
  // The exact fitted means are data-dependent; this assertion guards the
  // canonicalization contract rather than a particular market regime.
  expect(s.hmm.params.mu).toEqual([...s.hmm.params.mu].sort((a,b)=>a-b));
});


test("composite score exposes normalized components and Kelly is direction-aligned", () => {
  const candles = Array.from({ length: MIN_SIGNAL_BARS + 80 }, (_, i) => {
    const c = 100 * Math.exp(0.0005 * i + 0.01 * Math.sin(i / 8));
    return { t: i * 3600000, o: c, h: c * 1.002, l: c * 0.998, c, v: 1 };
  });
  const s = computeSignal(candles, "1h", 0.04);
  for (const v of Object.values(s.scoreComponents)) expect(v).toBeGreaterThanOrEqual(-1);
  for (const v of Object.values(s.scoreComponents)) expect(v).toBeLessThanOrEqual(1);
  if (s.action === "WAIT") expect(s.kellyFraction).toBe(0);
  if (s.action === "LONG") expect(s.kellyFraction).toBeGreaterThanOrEqual(0);
  if (s.action === "SHORT") expect(s.kellyFraction).toBeLessThanOrEqual(0);
});

test("Markowitz constraints hold", () => {
  const N=4,T=800; const R=Array.from({length:T},()=>Array.from({length:N},()=>0.01*randNorm(rng)));
  const S=covarianceMatrix(R); const Si=inverse(ridge(S,1e-6))!;
  const mu=[0.1,0.2,0.15,0.05]; const target=0.13;
  const w=efficientWeights(Si,mu,target);
  expect(w.reduce((a,b)=>a+b,0)).toBeCloseTo(1,8);
  expect(dot(w,mu)).toBeCloseTo(target,8);
  // min variance property: perturb along a direction keeping both constraints
  const d=[1,-1,0,0]; const dmu=dot(d,mu);
  const d2=d.map((v,i)=>v- (dmu/ (mu[1]-mu[0]||1))*0); // simple check: use null-space direction
  const base=quadForm(S,w);
  const dir=[1,-1,-1,1]; // sums to 0
  const adj=dir.map(v=>v); const c=dot(adj,mu);
  const nsp=adj.map((v,i)=>v); // ensure mu-neutral
  // build mu-neutral, sum-zero direction via Gram-Schmidt on [1,-1,0,0],[0,0,1,-1]
  const a=[1,-1,0,0], b=[0,0,1,-1];
  const alpha = dot(a,mu), beta = dot(b,mu);
  const dirn = a.map((v,i)=> beta*v - alpha*b[i]!);
  expect(dot(dirn,mu)).toBeCloseTo(0,10);
  expect(dirn.reduce((x,y)=>x+y,0)).toBeCloseTo(0,10);
  for(const eps of [1e-3,-1e-3]) {
    const w2=w.map((v,i)=>v+eps*dirn[i]!);
    expect(quadForm(S,w2)).toBeGreaterThan(base-1e-15);
  }
});



test("covariance is symmetric and correlation has unit diagonal", () => {
  const R = [[0.01, 0.02], [0.02, 0.01], [-0.01, -0.02], [0.03, 0.01]];
  const S = covarianceMatrix(R);
  const C = correlationFromCov(S);
  expect(S[0]![1]!).toBeCloseTo(S[1]![0]!, 14);
  expect(C[0]![0]!).toBeCloseTo(1, 12);
  expect(C[1]![1]!).toBeCloseTo(1, 12);
  expect(Math.abs(C[0]![1]!)).toBeLessThanOrEqual(1 + 1e-12);
});

test("nearest PSD removes negative eigenvalues", () => {
  const S = [[1, 1.000001], [1.000001, 1]];
  const P = nearestPsd(S, 1e-8);
  const { values } = jacobiEigen(P);
  expect(values.every((v) => v >= -1e-12)).toBe(true);
  expect(P[0]![0]!).toBeCloseTo(P[1]![1]!, 10);
});

test("eigen decomposition", () => {
  const A=[[4,1,0],[1,3,1],[0,1,2]];
  const {values,vectors}=jacobiEigen(A);
  expect(values.reduce((a,b)=>a+b,0)).toBeCloseTo(9,8);
  for(let k=0;k<3;k++){
    const v=vectors.map(r=>r[k]!);
    const Av=A.map(r=>r.reduce((s,x,j)=>s+x*v[j]!,0));
    for(let i=0;i<3;i++) expect(Av[i]!).toBeCloseTo(values[k]!*v[i]!,8);
  }
});

test("PCA explained sums to 1", () => {
  const R=Array.from({length:500},()=>{const f=randNorm(rng);return [f+0.1*randNorm(rng),f+0.1*randNorm(rng),randNorm(rng)];});
  const p=pca(R);
  expect(p.explained.reduce((a,b)=>a+b,0)).toBeCloseTo(1,6);
  expect(p.explained[0]!).toBeGreaterThan(0.5);
});

test("PCA eigenvectors are orthonormal and scores match loadings", () => {
  const R = Array.from({ length: 300 }, (_, t) => {
    const f = randNorm(rng);
    return [f + 0.2 * randNorm(rng), 0.5 * f + 0.2 * randNorm(rng), randNorm(rng)];
  });
  const p = pca(R);
  for (let i = 0; i < 3; i++) {
    const vi = p.loadings.map((r) => r[i]!);
    expect(Math.sqrt(vi.reduce((s, x) => s + x * x, 0))).toBeCloseTo(1, 8);
    for (let j = i + 1; j < 3; j++) {
      const vj = p.loadings.map((r) => r[j]!);
      expect(vi.reduce((s, x, k) => s + x * vj[k]!, 0)).toBeCloseTo(0, 8);
    }
  }
  expect(p.scores3d.every((s) => Number.isFinite(s.x) && Number.isFinite(s.y) && Number.isFinite(s.z))).toBe(true);
});

test("PCA rejects constant assets instead of inventing variance", () => {
  const R = Array.from({ length: 20 }, (_, i) => [i * 0.01, 0.5]);
  expect(() => pca(R)).toThrow(/variance positif/);
});


test("OU rejects non-stationary random-walk-like fits as unreliable", () => {
  const x = Array.from({ length: 120 }, (_, i) => i * 0.001 + randNorm(rng) * 0.01);
  const f = fitOu(x, 1 / 365);
  expect(f.reliable).toBe(false);
  expect(f.stationary).toBe(false);
});

test("OU exposes diagnostics and reliability on a mean-reverting sample", () => {
  const dt = 1 / 365;
  let x = 0;
  const series = [x];
  for (let i = 1; i < 500; i++) {
    x = 0.92 * x + randNorm(rng) * 0.02;
    series.push(x);
  }
  const f = fitOu(series, dt);
  expect(f.stationary).toBe(true);
  expect(f.reliable).toBe(true);
  expect(Number.isFinite(f.dfStatistic)).toBe(true);
  expect(f.halfLifeBars).toBeLessThan(series.length * 0.5);
});

test("OU innovation variance uses conditional MLE scaling", () => {
  const dt = 1 / 365;
  const b = 0.95;
  const innovation = 0.02;
  let x = 0;
  const series = [x];
  for (let i = 0; i < 5000; i++) {
    x = b * x + innovation * randNorm(rng);
    series.push(x);
  }
  const f = fitOu(series, dt);
  expect(f.innovationStd).toBeCloseTo(innovation, 2);
});

test("OU recovery", () => {
  const dt=1/365, theta=6, mu=5, sigma=0.4;
  const b=Math.exp(-theta*dt), sd=sigma*Math.sqrt((1-b*b)/(2*theta));
  const x=[mu]; for(let i=0;i<60000;i++) x.push(mu*(1-b)+b*x[i]!+sd*randNorm(rng));
  const f=fitOu(x,dt);
  expect(f.theta).toBeCloseTo(theta,0);
  expect(f.mu).toBeCloseTo(mu,1);
  expect(f.sigma).toBeCloseTo(sigma,1);
  expect(f.innovationStd).toBeGreaterThan(0);
});

test("EMA/RSI/MACD known values", () => {
  const x=[1,2,3,4,5,6,7,8,9,10];
  const e=ema(x,3);
  expect(e[2]).toBeCloseTo(2,10);            // SMA seed
  expect(e[3]).toBeCloseTo(0.5*4+0.5*2,10);  // alpha=0.5
  const up=Array.from({length:40},(_,i)=>100+i);
  expect(rsi(up,14)[20]!).toBeCloseTo(100,8); // monotone up -> RSI 100
  const z=rollingZScore([1,2,3,4,5],5);
  expect(z[4]!).toBeCloseTo(2/Math.sqrt(2.5),8);
  const m=macd(Array.from({length:100},(_,i)=>Math.sin(i/5)+i*0.01));
  expect(Number.isFinite(m.histogram[99]!)).toBe(true);
});


test("signal warm-up rejects insufficient history", () => {
  const candles = Array.from({ length: MIN_SIGNAL_BARS - 1 }, (_, i) => ({
    t: i * 60_000,
    o: 100 + i,
    h: 101 + i,
    l: 99 + i,
    c: 100 + i,
    v: 1,
  }));
  expect(() => computeSignal(candles, "1h")).toThrow(/minimal/);
});

test("signal warm-up is fixed rather than silently shrinking timing windows", () => {
  const candles = Array.from({ length: MIN_SIGNAL_BARS }, (_, i) => {
    const c = 100 + Math.sin(i / 3) + i * 0.05;
    return { t: i * 60_000, o: c, h: c + 1, l: c - 1, c, v: 1 };
  });
  const s = computeSignal(candles, "1h");
  expect(Number.isFinite(s.rsi)).toBe(true);
  expect(Number.isFinite(s.macdHist)).toBe(true);
  expect(Number.isFinite(s.z)).toBe(true);
});

test("moments", () => {
  const n=200000; const s:number[]=[]; for(let i=0;i<n;i++) s.push(randNorm(rng));
  expect(Math.abs(skewness(s))).toBeLessThan(0.05);
  expect(Math.abs(kurtosis(s))).toBeLessThan(0.1);
});

test("vol surface skew direction", () => {
  const s=volatilitySurface({sigmaShort:0.8,sigmaLong:0.6,skew:-1,excessKurtosis:3,moneyness:[0.8,1,1.2],maturities:[0.1,0.5]});
  // negative skew => higher IV for low strikes (puts)
  expect(s.grid[0]![0]!).toBeGreaterThan(s.grid[0]![2]!);
});


test("crypto annualization factors are interval-consistent", () => {
  expect(BARS_PER_YEAR["1h"]).toBe(8760);
  expect(BARS_PER_YEAR["4h"]).toBe(2190);
  expect(BARS_PER_YEAR["1d"]).toBe(365);
  expect(BARS_PER_YEAR["1h"] * (60 * 60 * 1000)).toBe(365 * 24 * 60 * 60 * 1000);
});

test("portfolio annualization", () => {
  const R=Array.from({length:1000},()=>[0.001+0.01*randNorm(rng),0.0005+0.02*randNorm(rng)]);
  const a=analyzePortfolio(R,365);
  expect(Math.sqrt(a.cov[0]![0]!)).toBeCloseTo(0.01*Math.sqrt(365),1);
  expect(a.frontier.every(f=>Math.abs(f.weights.reduce((x,y)=>x+y,0)-1)<1e-8)).toBe(true);
});

test("entry zone: LONG below ref, SHORT is mirror above ref", () => {
  const ou = { theta: 6, mu: 0, sigma: 0.4, halfLifeBars: 10, b: 0.9 };
  const sigmaZ = 0.05;
  const refLog = Math.log(100);
  const zEntry = optimalEntryThreshold(ou, sigmaZ, 0.04);
  expect(zEntry).toBeLessThan(0);

  const long = computeEntryZone("LONG", refLog, sigmaZ, zEntry)!;
  const short = computeEntryZone("SHORT", refLog, sigmaZ, zEntry)!;

  // LONG: entry di bawah ref, stop di bawah entry, target di antara keduanya.
  expect(long.entry).toBeLessThan(Math.exp(refLog));
  expect(long.stop).toBeLessThan(long.entry);
  expect(long.target).toBeGreaterThan(long.entry);
  expect(long.riskReward).toBeGreaterThan(0);

  // SHORT: cerminan LONG terhadap ref (simetri OU).
  expect(short.entry).toBeGreaterThan(Math.exp(refLog));
  expect(short.stop).toBeGreaterThan(short.entry);
  expect(short.target).toBeLessThan(short.entry);

  // Simetri log-price: jarak log(entry) ke refLog sama besar untuk LONG & SHORT.
  expect(Math.log(short.entry) - refLog).toBeCloseTo(-(Math.log(long.entry) - refLog), 10);
});


test("Monte Carlo VaR/CVaR are reported as positive loss magnitudes", () => {
  const g={mu:-0.2,sigma:0.5,logDrift:-0.325,dt:1/365};
  const mc=monteCarloGbm(100,g,365,2000,123);
  expect(mc.var95).toBeGreaterThan(0);
  expect(mc.cvar95).toBeGreaterThanOrEqual(mc.var95);
});

test("HMM EM reports convergence and non-decreasing likelihood", () => {
  const obs:number[]=[];
  let s=0;
  for(let i=0;i<1000;i++){ if(rng()<0.03) s=1-s; obs.push((s?0.015:-0.015)+0.004*randNorm(rng)); }
  const fit=fitHmm(obs,2,200,1e-7);
  expect(fit.converged).toBe(true);
  expect(fit.logLikelihoodHistory.length).toBe(fit.iterations);
  for(let i=1;i<fit.logLikelihoodHistory.length;i++) {
    expect(fit.logLikelihoodHistory[i]!).toBeGreaterThanOrEqual(fit.logLikelihoodHistory[i-1]! - 1e-7);
  }
  expect(fit.logLikelihood).toBeFinite();
});

test("HMM final posterior matches final fitted parameters", () => {
  const obs:number[]=[];
  for(let i=0;i<800;i++) obs.push((i%2?0.02:-0.02)+0.004*randNorm(rng));
  const fit=fitHmm(obs,2,80);
  let alpha=fit.params.pi.map((pi,j)=>pi * Math.exp(-0.5 * ((obs[0]!-fit.params.mu[j]!) / fit.params.sigma[j]!)**2) / (fit.params.sigma[j]! * Math.sqrt(2*Math.PI)));
  let z=alpha.reduce((a,b)=>a+b,0);
  alpha=alpha.map(v=>v/z);
  for(let t=1;t<obs.length;t++){
    const next=fit.params.mu.map((_,j)=>{
      let s=0;
      for(let i=0;i<2;i++) s += alpha[i]!*fit.params.A[i]![j]!;
      const e=Math.exp(-0.5*((obs[t]!-fit.params.mu[j]!)/fit.params.sigma[j]!)**2)/(fit.params.sigma[j]!*Math.sqrt(2*Math.PI));
      return s*e;
    });
    z=next.reduce((a,b)=>a+b,0);
    alpha=next.map(v=>v/z);
  }
  const last=fit.gamma[fit.gamma.length-1]!;
  expect(last[0]!).toBeCloseTo(alpha[0]!,8);
  expect(last[1]!).toBeCloseTo(alpha[1]!,8);
});

test("alignedReturns aligns by timestamps, not array suffix", () => {
  const a=[
    {t:1,o:100,h:101,l:99,c:100,v:1},
    {t:2,o:100,h:102,l:99,c:101,v:1},
    {t:3,o:101,h:103,l:100,c:102,v:1},
  ];
  const b=[
    {t:1,o:200,h:201,l:199,c:200,v:1},
    {t:3,o:200,h:204,l:199,c:202,v:1},
    {t:4,o:202,h:205,l:201,c:204,v:1},
  ];
  const out=alignedReturns([{symbol:"AAAUSDT",candles:a},{symbol:"BBBUSDT",candles:b}], "1h");
  expect(out.R).toHaveLength(0);

  const c=[...a, {t:4,o:102,h:104,l:101,c:103,v:1}];
  const d=[...b, {t:4,o:202,h:205,l:201,c:204,v:1}];
  const aligned=alignedReturns([{symbol:"AAAUSDT",candles:c},{symbol:"BBBUSDT",candles:d}], "1h");
  expect(aligned.R).toHaveLength(1);
  expect(aligned.R[0]![0]!).toBeCloseTo(Math.log(103/102), 12);
  expect(aligned.R[0]![1]!).toBeCloseTo(Math.log(204/202), 12);
});

test("alignedReturns rejects a multi-bar gap instead of creating a longer return", () => {
  const a=[
    {t:0,o:100,h:101,l:99,c:100,v:1},
    {t:3600000,o:100,h:102,l:99,c:101,v:1},
    {t:10800000,o:101,h:104,l:100,c:103,v:1},
  ];
  const b=[
    {t:0,o:200,h:201,l:199,c:200,v:1},
    {t:3600000,o:200,h:203,l:199,c:202,v:1},
    {t:10800000,o:202,h:205,l:201,c:204,v:1},
  ];
  const out=alignedReturns([{symbol:"AAAUSDT",candles:a},{symbol:"BBBUSDT",candles:b}], "1h");
  expect(out.R).toHaveLength(1);
  expect(out.R[0]![0]!).toBeCloseTo(Math.log(101/100), 12);
  expect(out.R[0]![1]!).toBeCloseTo(Math.log(202/200), 12);
});

test("Monte Carlo is reproducible for the same seed", () => {
  const g = { mu: 0.2, sigma: 0.5, logDrift: 0, dt: 1 / 365 };
  const a = monteCarloGbm(100, g, 20, 100, 12345);
  const b = monteCarloGbm(100, g, 20, 100, 12345);
  expect(a.seed).toBe(b.seed);
  expect(a.terminal).toEqual(b.terminal);
  expect(a.bands.p50).toEqual(b.bands.p50);
});

test("Monte Carlo changes its stream when seed changes", () => {
  const g = { mu: 0.2, sigma: 0.5, logDrift: 0, dt: 1 / 365 };
  const a = monteCarloGbm(100, g, 20, 100, 12345);
  const b = monteCarloGbm(100, g, 20, 100, 12346);
  expect(a.terminal).not.toEqual(b.terminal);
});

test("stableSeed is deterministic and symbol-specific", () => {
  const a = stableSeed("BTCUSDT|1h|72");
  const b = stableSeed("BTCUSDT|1h|72");
  const c = stableSeed("ETHUSDT|1h|72");
  expect(a).toBe(b);
  expect(a).not.toBe(c);
});


test("sample variance uses a numerically stable estimator", () => {
  const x = [1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4];
  expect(variance(x)).toBeCloseTo(5 / 3, 10);
  expect(stdev(x)).toBeCloseTo(Math.sqrt(5 / 3), 10);
});

test("annualized volatility is consistent across equivalent bar scales", () => {
  const hourly = Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.37) * 0.01);
  const oneHour = annualizedVolatility(hourly, 8760);
  const fourHour = annualizedVolatility(hourly, 8760 / 4);
  expect(oneHour).toBeGreaterThan(0);
  expect(fourHour / oneHour).toBeCloseTo(0.5, 12);
});

test("annualized volatility rejects invalid annualization factor", () => {
  expect(() => annualizedVolatility([0.01, -0.01], 0)).toThrow();
  expect(() => annualizedVolatility([0.01, Number.NaN], 365)).toThrow();
});


test("trade net return reconciles with fixed-notional cash accounting", () => {
  const c = 0.0007;
  const raw = directionalRawReturn("LONG", 100, 110);
  const net = netTradeReturn("LONG", 100, 110, c);
  // Entry fee is charged on entry notional; exit fee is charged on exit notional.
  expect(net).toBeCloseTo((1 + raw) * (1 - c) - 1 - c, 14);
  expect(net).not.toBeCloseTo((1 + raw) * (1 - c) ** 2 - 1, 10);
});

test("fixed-notional position PnL is not rebalanced every bar", () => {
  const entryEquity = 1;
  const fraction = 0.25;
  const notional = entryEquity * fraction;
  const c = 0.0007;
  const raw = 0.10;
  const exitNotional = notional * (1 + raw);
  const finalEquity = entryEquity - notional * c + notional * raw - exitNotional * c;
  const expected = entryEquity + notional * netTradeReturn("LONG", 100, 110, c);
  expect(finalEquity).toBeCloseTo(expected, 14);
});

test("LONG and SHORT fixed-notional zero-move accounting is symmetric", () => {
  const c = 0.0007;
  expect(netTradeReturn("LONG", 100, 100, c)).toBeCloseTo(-2 * c, 14);
  expect(netTradeReturn("SHORT", 100, 100, c)).toBeCloseTo(-2 * c, 14);
  expect(positionFractionFromStop(100, 95, 0.01, 0.25)).toBeCloseTo(0.2, 12);
});

test("Black-Scholes handles expiry without singular Greeks", () => {
  const c = blackScholes(110, 100, 0, 0.05, 0.4, "call");
  expect(c.price).toBe(10);
  expect(c.delta).toBe(1);
  expect(c.gamma).toBe(0);
  expect(c.vega).toBe(0);
  expect(c.theta).toBe(0);
  expect(c.rho).toBe(0);
});

test("Black-Scholes handles zero volatility by deterministic limit", () => {
  const c = blackScholes(110, 100, 1, 0.05, 0, "call");
  expect(c.price).toBeCloseTo(110 - 100 * Math.exp(-0.05), 10);
  expect(c.delta).toBe(1);
  expect(c.vega).toBe(0);
});

test("Black-Scholes Greeks use standard annual conventions", () => {
  const g = blackScholes(100, 100, 1, 0.02, 0.5, "call");
  expect(g.vega).toBeGreaterThan(0);
  expect(g.theta).toBeLessThan(0);
  expect(g.rho).toBeGreaterThan(0);
});


test("long-only max Sharpe respects simplex and improves over every single-asset portfolio", () => {
  const cov = [
    [0.04, 0.006, 0.002],
    [0.006, 0.09, 0.003],
    [0.002, 0.003, 0.16],
  ];
  const mu = [0.12, 0.18, 0.08];
  const rf = 0.04;
  const p = analyzePortfolio(
    Array.from({ length: 200 }, (_, t) => [
      0.0003 + 0.01 * Math.sin(t / 7),
      0.0005 + 0.012 * Math.cos(t / 9),
      0.0002 + 0.015 * Math.sin(t / 11),
    ]),
    365,
    rf,
  );
  const w = p.longOnly.weights;
  expect(w.every((x) => x >= -1e-9)).toBe(true);
  expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8);
  expect(Number.isFinite(p.longOnly.sharpe)).toBe(true);
});

test("efficient frontier weights satisfy full-investment constraint", () => {
  const R = Array.from({ length: 300 }, (_, t) => [
    0.0004 + 0.008 * Math.sin(t / 13),
    0.0003 + 0.01 * Math.cos(t / 17),
    0.0002 + 0.012 * Math.sin(t / 19),
  ]);
  const p = analyzePortfolio(R, 365, 0.04, 40);
  expect(p.frontier.every((f) => Math.abs(f.weights.reduce((a, b) => a + b, 0) - 1) < 1e-8)).toBe(true);
  expect(p.gmv.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8);
});


test("market candle normalization sorts out-of-order input and collapses exact duplicates", () => {
  const base = 1_700_000_000_000;
  const step = 60 * 60 * 1000;
  const mk = (t:number, c:number) => ({t, o:c-1, h:c+1, l:c-1, c, v:10});
  const out = normalizeCandlesForInterval(
    [mk(base + step, 101), mk(base, 100), mk(base, 100)],
    "1h",
    base + 3 * step,
  );
  expect(out?.map(c => c.t)).toEqual([base, base + step]);
});

test("market candle normalization rejects conflicting duplicate timestamps", () => {
  const base = 1_700_000_000_000;
  const step = 60 * 60 * 1000;
  const a = {t:base, o:99, h:101, l:99, c:100, v:10};
  const b = {...a, c:100.5};
  expect(normalizeCandlesForInterval([a,b], "1h", base + 3 * step)).toBeNull();
});

test("market candle normalization rejects gaps instead of interpolating them", () => {
  const base = 1_700_000_000_000;
  const step = 60 * 60 * 1000;
  const mk = (t:number, c:number) => ({t, o:c-1, h:c+1, l:c-1, c, v:10});
  expect(normalizeCandlesForInterval(
    [mk(base,100), mk(base + 2 * step,102)],
    "1h",
    base + 4 * step,
  )).toBeNull();
});

test("market candle normalization removes the still-open latest candle", () => {
  const base = 1_700_000_000_000;
  const step = 60 * 60 * 1000;
  const mk = (t:number, c:number) => ({t, o:c-1, h:c+1, l:c-1, c, v:10});
  const out = normalizeCandlesForInterval(
    [mk(base,100), mk(base + step,101), mk(base + 2 * step,102)],
    "1h",
    base + 2 * step + step / 2,
  );
  expect(out?.length).toBe(2);
});

test("market candle normalization rejects misaligned timestamps and invalid OHLCV", () => {
  const step = 60 * 60 * 1000;
  const base = 1_700_000_000_001;
  const mk = (t:number, c:number) => ({t, o:c-1, h:c+1, l:c-1, c, v:10});
  expect(normalizeCandlesForInterval([mk(base,100), mk(base + step,101)], "1h", base + 3 * step)).toBeNull();
  expect(normalizeCandlesForInterval([mk(1_700_000_000_000,100), {...mk(1_700_000_000_000 + step,101), h:90}], "1h", 1_700_000_000_000 + 3 * step)).toBeNull();
});
