import type { Line3, Point3, Scene3D, Vec3 } from "./engine3d";

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function smoothstep(value: number) {
  const t = clamp01(value);
  return t * t * (3 - 2 * t);
}

function pingPong(time: number, seconds: number) {
  const cycle = Math.max(seconds, 0.1);
  const phase = ((time % cycle) + cycle) % cycle / cycle;
  return phase < 0.5 ? phase * 2 : 2 - phase * 2;
}

function pointOnLine(points: Vec3[], progress: number): Vec3 | null {
  if (points.length === 0) return null;
  if (points.length === 1) return points[0];

  const t = clamp01(progress);
  const scaled = t * (points.length - 1);
  const index = Math.floor(scaled);
  const nextIndex = Math.min(index + 1, points.length - 1);
  const local = scaled - index;

  const a = points[index];
  const b = points[nextIndex];

  if (!a || !b) return null;

  return [
    a[0] + (b[0] - a[0]) * local,
    a[1] + (b[1] - a[1]) * local,
    a[2] + (b[2] - a[2]) * local,
  ];
}

function addMotionPoint(
  points: Point3[],
  p: Vec3 | null,
  color: string,
  size: number,
  phase: number,
) {
  if (!p) return;
  points.push({
    p,
    color,
    size,
    alpha: 1,
    glow: true,
    phase,
    jitter: 0,
  });
}

function monteCarloMotion(scene: Scene3D, time: number): Scene3D {
  const lines = scene.lines ?? [];
  if (lines.length < 4) return scene;

  // The last three lines are P50/P95/P05 in scenes.ts.
  const bandStart = Math.max(0, lines.length - 3);
  const pathLines = lines.slice(0, bandStart);
  const points: Point3[] = [...(scene.points ?? [])];

  // The cursor moves through the actual simulated time axis.
  const baseProgress = smoothstep(pingPong(time, 7.5));

  // Highlight several actual simulation paths. Their phase is derived from
  // their z-position, so the motion remains tied to the dataset.
  const sampleCount = Math.min(8, pathLines.length);
  if (sampleCount > 0) {
    const stride = Math.max(1, Math.floor(pathLines.length / sampleCount));

    for (let i = 0; i < pathLines.length; i += stride) {
      const line = pathLines[i];
      if (!line) continue;

      const z = line.pts[0]?.[2] ?? 0;
      const pathPhase = (z + 1) / 2;
      const progress = clamp01(
        baseProgress * 0.82 + pathPhase * 0.18,
      );

      addMotionPoint(
        points,
        pointOnLine(line.pts, progress),
        line.color,
        2.5,
        pathPhase,
      );
    }
  }

  // Always show a stronger cursor on the median P50 path.
  const p50 = lines[bandStart];
  if (p50) {
    addMotionPoint(
      points,
      pointOnLine(p50.pts, baseProgress),
      p50.color,
      4,
      0.15,
    );
  }

  return { ...scene, points };
}

function hiddenStateMotion(scene: Scene3D, time: number): Scene3D {
  const source = scene.points ?? [];
  if (source.length === 0) return scene;

  // Z is the chronological coordinate in the HMM scene.
  const current = pingPong(time, 8);
  const points: Point3[] = [...source];

  let active: Point3 | null = null;
  let bestDistance = Infinity;

  for (const point of source) {
    const normalizedTime = (point.p[2] + 1) / 2;
    const distance = Math.abs(normalizedTime - current);

    if (distance < bestDistance) {
      bestDistance = distance;
      active = point;
    }
  }

  if (active) {
    addMotionPoint(
      points,
      active.p,
      active.color,
      Math.max(active.size ?? 1.7, 3.4),
      active.phase ?? 0,
    );
  }

  return { ...scene, points };
}

function frontierMotion(scene: Scene3D, time: number): Scene3D {
  const curve = scene.lines?.find(
    (line) => line.width !== undefined && line.width >= 2,
  );

  if (!curve) return scene;

  const progress = smoothstep(pingPong(time, 6.5));
  const points: Point3[] = [...(scene.points ?? [])];

  addMotionPoint(
    points,
    pointOnLine(curve.pts, progress),
    curve.color,
    4.2,
    0.25,
  );

  return { ...scene, points };
}

function volatilityMotion(scene: Scene3D, time: number): Scene3D {
  const quads = scene.quads ?? [];
  if (quads.length === 0) return scene;

  const points: Point3[] = [...(scene.points ?? [])];
  const progress = pingPong(time, 9);

  // Sweep through the actual surface cells. The marker position comes from
  // the surface geometry, not from a random animation coordinate.
  const index = Math.min(
    quads.length - 1,
    Math.floor(progress * quads.length),
  );
  const quad = quads[index];

  if (quad) {
    const center: Vec3 = [
      (quad.pts[0][0] + quad.pts[1][0] + quad.pts[2][0] + quad.pts[3][0]) / 4,
      (quad.pts[0][1] + quad.pts[1][1] + quad.pts[2][1] + quad.pts[3][1]) / 4,
      (quad.pts[0][2] + quad.pts[1][2] + quad.pts[2][2] + quad.pts[3][2]) / 4,
    ];

    addMotionPoint(points, center, quad.color, 3.2, progress);
  }

  return { ...scene, points };
}

export function applyDataMotion(scene: Scene3D, time: number): Scene3D {
  const labels = scene.axisLabels;

  if (!labels) return scene;

  if (
    labels[0] === "Waktu" &&
    labels[1] === "Harga" &&
    labels[2] === "Simulasi #"
  ) {
    return monteCarloMotion(scene, time);
  }

  if (
    labels[0] === "Return" &&
    labels[1] === "Volatilitas" &&
    labels[2] === "Waktu"
  ) {
    return hiddenStateMotion(scene, time);
  }

  if (
    labels[0] === "Risk σ" &&
    labels[1] === "Return μ" &&
    labels[2] === "Aset"
  ) {
    return frontierMotion(scene, time);
  }

  if (
    labels[0] === "K/S" &&
    labels[1] === "Implied Vol" &&
    labels[2] === "Maturity"
  ) {
    return volatilityMotion(scene, time);
  }

  return scene;
}
