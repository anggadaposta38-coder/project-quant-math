import { expect, test } from "vitest";
import { blackScholes, fitGbm, fitOu, monteCarloGbm, volatilitySurface } from "./src/lib/quant/stochastic";
import { fitHmm } from "./src/lib/quant/hmm";
import { analyzePortfolio, efficientWeights, pca } from "./src/lib/quant/portfolio";
import { covarianceMatrix, dot, inverse, jacobiEigen, mulberry32, quadForm, randNorm, ridge, normCdf, kurtosis, skewness } from "./src/lib/quant/stats";
import { ema, macd, rsi, rollingZScore } from "./src/lib/quant/indicators";

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
});

test("Monte Carlo E[S_T] = S0 e^{mu T}", () => {
  const g={mu:0.4,sigma:0.7,logDrift:0,dt:1/365};
  const mc=monteCarloGbm(100,g,365,20000,42);
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

test("OU recovery", () => {
  const dt=1/365, theta=6, mu=5, sigma=0.4;
  const b=Math.exp(-theta*dt), sd=sigma*Math.sqrt((1-b*b)/(2*theta));
  const x=[mu]; for(let i=0;i<60000;i++) x.push(mu*(1-b)+b*x[i]!+sd*randNorm(rng));
  const f=fitOu(x,dt);
  expect(f.theta).toBeCloseTo(theta,0);
  expect(f.mu).toBeCloseTo(mu,1);
  expect(f.sigma).toBeCloseTo(sigma,1);
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

test("portfolio annualization", () => {
  const R=Array.from({length:1000},()=>[0.001+0.01*randNorm(rng),0.0005+0.02*randNorm(rng)]);
  const a=analyzePortfolio(R,365);
  expect(Math.sqrt(a.cov[0]![0]!)).toBeCloseTo(0.01*Math.sqrt(365),1);
  expect(a.frontier.every(f=>Math.abs(f.weights.reduce((x,y)=>x+y,0)-1)<1e-8)).toBe(true);
});
