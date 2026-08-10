/** Mesin render 3D ringan di atas Canvas 2D: proyeksi perspektif + depth sort. */

export type Vec3 = [number, number, number];

export interface Point3 {
  p: Vec3;
  color: string;
  size?: number;
  alpha?: number;
}
export interface Line3 {
  pts: Vec3[];
  color: string;
  width?: number;
  alpha?: number;
}
export interface Quad3 {
  pts: [Vec3, Vec3, Vec3, Vec3];
  color: string;
  alpha?: number;
  stroke?: string;
}

export interface Scene3D {
  points?: Point3[];
  lines?: Line3[];
  quads?: Quad3[];
  axisLabels?: [string, string, string];
  ticks?: { pos: Vec3; text: string }[];
}

export interface Camera {
  yaw: number;
  pitch: number;
  zoom: number;
}

export function rotate(p: Vec3, cam: Camera): Vec3 {
  const [x, y, z] = p;
  const cy = Math.cos(cam.yaw);
  const sy = Math.sin(cam.yaw);
  const x1 = x * cy - z * sy;
  const z1 = x * sy + z * cy;
  const cp = Math.cos(cam.pitch);
  const sp = Math.sin(cam.pitch);
  const y2 = y * cp - z1 * sp;
  const z2 = y * sp + z1 * cp;
  return [x1, y2, z2];
}

const DIST = 4.2;

export function project(
  p: Vec3,
  cam: Camera,
  w: number,
  h: number,
): { x: number; y: number; depth: number } {
  const [x, y, z] = rotate(p, cam);
  const depth = z + DIST;
  const f = (Math.min(w, h) / 2) * cam.zoom * 1.9;
  const s = f / Math.max(depth, 0.15);
  return { x: w / 2 + x * s, y: h / 2 - y * s, depth };
}

function withAlpha(ctx: CanvasRenderingContext2D, alpha: number, fn: () => void) {
  const prev = ctx.globalAlpha;
  ctx.globalAlpha = alpha;
  fn();
  ctx.globalAlpha = prev;
}

const AXIS_EDGES: [Vec3, Vec3][] = [
  [[-1, -1, -1], [1, -1, -1]],
  [[-1, -1, -1], [-1, 1, -1]],
  [[-1, -1, -1], [-1, -1, 1]],
  [[1, -1, -1], [1, -1, 1]],
  [[1, -1, -1], [1, 1, -1]],
  [[-1, -1, 1], [1, -1, 1]],
  [[-1, -1, 1], [-1, 1, 1]],
  [[-1, 1, -1], [1, 1, -1]],
  [[-1, 1, -1], [-1, 1, 1]],
  [[1, 1, -1], [1, 1, 1]],
  [[-1, 1, 1], [1, 1, 1]],
  [[1, -1, 1], [1, 1, 1]],
];

export function renderScene(
  ctx: CanvasRenderingContext2D,
  scene: Scene3D,
  cam: Camera,
  w: number,
  h: number,
  theme: { grid: string; text: string },
) {
  ctx.clearRect(0, 0, w, h);

  // Kerangka kubus
  ctx.lineWidth = 1;
  ctx.strokeStyle = theme.grid;
  withAlpha(ctx, 0.5, () => {
    for (const [a, b] of AXIS_EDGES) {
      const pa = project(a, cam, w, h);
      const pb = project(b, cam, w, h);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }
  });

  type Item = { depth: number; draw: () => void };
  const items: Item[] = [];

  for (const q of scene.quads ?? []) {
    const proj = q.pts.map((p) => project(p, cam, w, h));
    const depth = proj.reduce((s, p) => s + p.depth, 0) / 4;
    items.push({
      depth,
      draw: () => {
        ctx.beginPath();
        ctx.moveTo(proj[0]!.x, proj[0]!.y);
        for (let i = 1; i < 4; i++) ctx.lineTo(proj[i]!.x, proj[i]!.y);
        ctx.closePath();
        withAlpha(ctx, q.alpha ?? 1, () => {
          ctx.fillStyle = q.color;
          ctx.fill();
          if (q.stroke) {
            ctx.strokeStyle = q.stroke;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      },
    });
  }

  for (const l of scene.lines ?? []) {
    if (l.pts.length < 2) continue;
    const proj = l.pts.map((p) => project(p, cam, w, h));
    const depth = proj.reduce((s, p) => s + p.depth, 0) / proj.length;
    items.push({
      depth,
      draw: () => {
        withAlpha(ctx, l.alpha ?? 1, () => {
          ctx.strokeStyle = l.color;
          ctx.lineWidth = l.width ?? 1;
          ctx.beginPath();
          ctx.moveTo(proj[0]!.x, proj[0]!.y);
          for (let i = 1; i < proj.length; i++) ctx.lineTo(proj[i]!.x, proj[i]!.y);
          ctx.stroke();
        });
      },
    });
  }

  for (const pt of scene.points ?? []) {
    const pr = project(pt.p, cam, w, h);
    items.push({
      depth: pr.depth,
      draw: () => {
        const r = ((pt.size ?? 2) * 2.6) / Math.max(pr.depth, 0.3);
        withAlpha(ctx, pt.alpha ?? 1, () => {
          ctx.fillStyle = pt.color;
          ctx.beginPath();
          ctx.arc(pr.x, pr.y, Math.max(r, 0.6), 0, Math.PI * 2);
          ctx.fill();
        });
      },
    });
  }

  items.sort((a, b) => b.depth - a.depth);
  for (const it of items) it.draw();

  // Label sumbu
  if (scene.axisLabels) {
    ctx.font = "500 11px ui-monospace, monospace";
    ctx.fillStyle = theme.text;
    const anchors: Vec3[] = [
      [1.18, -1, -1],
      [-1, 1.18, -1],
      [-1, -1, 1.18],
    ];
    scene.axisLabels.forEach((label, i) => {
      const pr = project(anchors[i]!, cam, w, h);
      ctx.textAlign = "center";
      ctx.fillText(label, pr.x, pr.y);
    });
  }

  for (const t of scene.ticks ?? []) {
    const pr = project(t.pos, cam, w, h);
    ctx.font = "400 9px ui-monospace, monospace";
    ctx.fillStyle = theme.text;
    ctx.textAlign = "center";
    withAlpha(ctx, 0.75, () => ctx.fillText(t.text, pr.x, pr.y));
  }
}

/** Normalisasi linear ke rentang [-1, 1]. */
export function makeScaler(min: number, max: number) {
  const span = max - min;
  if (!Number.isFinite(span) || span === 0) return () => 0;
  return (v: number) => ((v - min) / span) * 2 - 1;
}

export function extent(values: number[]): [number, number] {
  let mn = Infinity;
  let mx = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (!Number.isFinite(mn)) return [0, 1];
  if (mn === mx) return [mn - 1, mx + 1];
  return [mn, mx];
}
