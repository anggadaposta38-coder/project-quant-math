/** Pembentuk Scene3D untuk tiap model matematika. */

import type { Scene3D, Vec3 } from "./engine3d";
import { extent, makeScaler } from "./engine3d";
import type { Palette } from "./palette";
import type { MonteCarloResult } from "@/lib/quant/stochastic";
import type { FrontierPoint } from "@/lib/quant/portfolio";

const lerpColor = (a: string, b: string, t: number) => (t < 0.5 ? a : b);

/** Monte Carlo Price Cone — sumbu: Waktu × Simulasi# × Harga */
export function monteCarloScene(
  mc: MonteCarloResult,
  s0: number,
  palette: Palette,
  maxPaths = 140,
): Scene3D {
  const steps = mc.paths[0] ? mc.paths[0].length - 1 : 0;
  const all = mc.paths.flat();
  const [lo, hi] = extent([...mc.bands.p05, ...mc.bands.p95, s0]);
  const pad = (hi - lo) * 0.08;
  const sy = makeScaler(Math.min(lo - pad, Math.min(...all) * 0.999), hi + pad);
  const sx = makeScaler(0, steps);

  const stride = Math.max(1, Math.floor(mc.paths.length / maxPaths));
  const lines: Scene3D["lines"] = [];

  for (let i = 0; i < mc.paths.length; i += stride) {
    const path = mc.paths[i]!;
    const z = ((i / mc.paths.length) * 2 - 1) * 0.9;
    const pts: Vec3[] = [];
    for (let t = 0; t <= steps; t += Math.max(1, Math.floor(steps / 60))) {
      pts.push([sx(t), sy(path[t]!), z]);
    }
    pts.push([sx(steps), sy(path[steps]!), z]);
    const up = path[steps]! >= s0;
    lines.push({
      pts,
      color: lerpColor(palette.bear, palette.bull, up ? 1 : 0),
      width: 0.8,
      alpha: 0.22,
    });
  }

  const bandLine = (band: number[], color: string, width: number, alpha: number) => {
    const pts: Vec3[] = [];
    for (let t = 0; t <= steps; t += Math.max(1, Math.floor(steps / 80)))
      pts.push([sx(t), sy(band[t]!), 0]);
    pts.push([sx(steps), sy(band[steps]!), 0]);
    lines.push({ pts, color, width, alpha });
  };
  bandLine(mc.bands.p50, palette.accent, 2.2, 1);
  bandLine(mc.bands.p95, palette.series[0]!, 1.4, 0.9);
  bandLine(mc.bands.p05, palette.series[0]!, 1.4, 0.9);

  return {
    lines,
    axisLabels: ["Waktu", "Harga", "Simulasi #"],
    ticks: [
      { pos: [1.02, sy(mc.bands.p95[steps]!), 0], text: "P95" },
      { pos: [1.02, sy(mc.bands.p50[steps]!), 0], text: "P50" },
      { pos: [1.02, sy(mc.bands.p05[steps]!), 0], text: "P05" },
    ],
  };
}

/** Volatility Surface — sumbu: Strike(moneyness) × Time to Expiry × Implied Vol */
export function volSurfaceScene(
  surface: { grid: number[][]; moneyness: number[]; maturities: number[] },
  palette: Palette,
): Scene3D {
  const { grid, moneyness, maturities } = surface;
  const flat = grid.flat();
  const [lo, hi] = extent(flat);
  const sy = makeScaler(lo, hi);
  const sx = makeScaler(Math.min(...moneyness), Math.max(...moneyness));
  const sz = makeScaler(Math.min(...maturities), Math.max(...maturities));

  const quads: Scene3D["quads"] = [];
  for (let i = 0; i < maturities.length - 1; i++) {
    for (let j = 0; j < moneyness.length - 1; j++) {
      const v = (grid[i]![j]! + grid[i]![j + 1]! + grid[i + 1]![j]! + grid[i + 1]![j + 1]!) / 4;
      const t = (v - lo) / Math.max(hi - lo, 1e-9);
      quads.push({
        pts: [
          [sx(moneyness[j]!), sy(grid[i]![j]!), sz(maturities[i]!)],
          [sx(moneyness[j + 1]!), sy(grid[i]![j + 1]!), sz(maturities[i]!)],
          [sx(moneyness[j + 1]!), sy(grid[i + 1]![j + 1]!), sz(maturities[i + 1]!)],
          [sx(moneyness[j]!), sy(grid[i + 1]![j]!), sz(maturities[i + 1]!)],
        ],
        color: t > 0.66 ? palette.bear : t > 0.33 ? palette.accent : palette.series[0]!,
        alpha: 0.55,
        stroke: palette.grid,
      });
    }
  }

  return {
    quads,
    axisLabels: ["K/S", "Implied Vol", "Maturity"],
    ticks: [
      { pos: [-1, sy(hi), -1.1], text: `${(hi * 100).toFixed(0)}%` },
      { pos: [-1, sy(lo), -1.1], text: `${(lo * 100).toFixed(0)}%` },
    ],
  };
}

/** HMM Hidden State Space — sumbu: Return × Volatilitas × Waktu */
export function hmmScene(
  data: { ret: number; vol: number; state: number }[],
  palette: Palette,
  K: number,
): Scene3D {
  const rets = data.map((d) => d.ret);
  const vols = data.map((d) => d.vol);
  const sx = makeScaler(...extent(rets));
  const sy = makeScaler(...extent(vols));
  const sz = makeScaler(0, Math.max(data.length - 1, 1));
  const stateColor = (s: number) =>
    K <= 2
      ? s === 0
        ? palette.bear
        : palette.bull
      : s === 0
        ? palette.bear
        : s === K - 1
          ? palette.bull
          : palette.neutral;

  const points = data.map((d, i) => ({
    p: [sx(d.ret), sy(d.vol), sz(i)] as Vec3,
    color: stateColor(d.state),
    size: 1.7,
    alpha: 0.85,
  }));

  const lines: Scene3D["lines"] = [];
  let run: Vec3[] = [];
  let runState = data[0]?.state ?? 0;
  data.forEach((d, i) => {
    const p: Vec3 = [sx(d.ret), sy(d.vol), sz(i)];
    if (d.state !== runState) {
      if (run.length > 1) lines.push({ pts: run, color: stateColor(runState), width: 1, alpha: 0.35 });
      run = [p];
      runState = d.state;
    } else run.push(p);
  });
  if (run.length > 1) lines.push({ pts: run, color: stateColor(runState), width: 1, alpha: 0.35 });

  return { points, lines, axisLabels: ["Return", "Volatilitas", "Waktu"] };
}

/** Efficient Frontier Surface — sumbu: Risk × Return × Bobot Aset */
export function frontierScene(
  frontier: FrontierPoint[],
  labels: string[],
  palette: Palette,
): Scene3D {
  const risks = frontier.map((f) => f.risk);
  const rets = frontier.map((f) => f.ret);
  const sx = makeScaler(...extent(risks));
  const sy = makeScaler(...extent(rets));
  const n = labels.length;
  const zOf = (j: number) => (n <= 1 ? 0 : (j / (n - 1)) * 2 - 1);

  const quads: Scene3D["quads"] = [];
  for (let i = 0; i < frontier.length - 1; i++) {
    for (let j = 0; j < n; j++) {
      const w0 = frontier[i]!.weights[j]!;
      const w1 = frontier[i + 1]!.weights[j]!;
      const zc = zOf(j);
      const halfW = n > 1 ? 1 / (n - 1) : 0.5;
      quads.push({
        pts: [
          [sx(frontier[i]!.risk), sy(frontier[i]!.ret), zc - halfW * 0.4],
          [sx(frontier[i]!.risk), sy(frontier[i]!.ret), zc + halfW * 0.4],
          [sx(frontier[i + 1]!.risk), sy(frontier[i + 1]!.ret), zc + halfW * 0.4],
          [sx(frontier[i + 1]!.risk), sy(frontier[i + 1]!.ret), zc - halfW * 0.4],
        ],
        color: (w0 + w1) / 2 >= 0 ? palette.series[j % palette.series.length]! : palette.bear,
        alpha: Math.min(0.85, 0.12 + Math.abs((w0 + w1) / 2) * 0.9),
      });
    }
  }

  const curve: Vec3[] = frontier.map((f) => [sx(f.risk), sy(f.ret), 0]);
  return {
    quads,
    lines: [{ pts: curve, color: palette.accent, width: 2.2, alpha: 1 }],
    axisLabels: ["Risk σ", "Return μ", "Aset"],
    ticks: labels.map((l, j) => ({ pos: [-1.12, -1, zOf(j)] as Vec3, text: l })),
  };
}

/** Eigen-space Cloud — sumbu: PC1 × PC2 × PC3 */
export function pcaScene(
  scores: { x: number; y: number; z: number }[],
  states: number[],
  palette: Palette,
  K: number,
): Scene3D {
  const sx = makeScaler(...extent(scores.map((s) => s.x)));
  const sy = makeScaler(...extent(scores.map((s) => s.y)));
  const sz = makeScaler(...extent(scores.map((s) => s.z)));
  const color = (s: number | undefined) =>
    s === undefined
      ? palette.neutral
      : s === 0
        ? palette.bear
        : s === K - 1
          ? palette.bull
          : palette.neutral;

  return {
    points: scores.map((s, i) => ({
      p: [sx(s.x), sy(s.y), sz(s.z)] as Vec3,
      color: color(states[i]),
      size: 1.8,
      alpha: 0.8,
    })),
    axisLabels: ["PC1", "PC2", "PC3"],
  };
}
