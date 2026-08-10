/**
 * Motion Engine V4 — Data-driven, low-cost motion for Canvas 3D scenes.
 *
 * Prinsip:
 * - Tidak memakai Math.random().
 * - Tidak mengubah nilai kuantitatif yang menjadi sumber scene.
 * - Gerakan dihitung dari posisi/geometry scene + waktu.
 * - Motion dibuat kecil agar data tetap terbaca.
 * - Cocok untuk mobile karena tidak melakukan O(n²) simulation.
 */

import type { Scene3D, Vec3 } from "@/lib/viz/engine3d";

export type MotionV4Profile =
  | "surface"
  | "cloud"
  | "frontier"
  | "cone"
  | "default";

export type MotionV4Options = {
  profile?: MotionV4Profile;
  intensity?: number;
  mobile?: boolean;
};

export type MotionV4Frame = {
  time: number;
  dt: number;
  activity: number;
};

const PROFILE_DEFAULTS: Record<MotionV4Profile, number> = {
  surface: 1,
  cloud: 0.82,
  frontier: 0.7,
  cone: 0.9,
  default: 0.72,
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function finite(value: number, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function length(v: Vec3) {
  return Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
}

function normalize(v: Vec3): Vec3 {
  const len = length(v);
  if (len < 1e-6) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale(v: Vec3, amount: number): Vec3 {
  return [v[0] * amount, v[1] * amount, v[2] * amount];
}

/**
 * Deterministic phase from index. Stable across renders/data refreshes.
 */
export function motionPhase(index: number, suppliedPhase?: number) {
  if (Number.isFinite(suppliedPhase)) return suppliedPhase as number;
  return (index * 0.6180339887498949) % 1;
}

/**
 * Produces a small deterministic displacement from the geometry itself.
 * The displacement is deliberately bounded; it is animation, not a data
 * transformation.
 */
export function dataMotionOffset(
  p: Vec3,
  time: number,
  phase: number,
  amplitude: number,
): Vec3 {
  const ph = phase * Math.PI * 2;
  const localScale = 0.55 + 0.45 * clamp(length(p) / 1.732, 0, 1);

  const x =
    Math.sin(time * 0.42 + p[2] * 1.7 + ph) * amplitude * localScale;
  const y =
    Math.sin(time * 0.68 + p[0] * 2.1 + p[2] * 0.8 + ph * 1.31) *
    amplitude *
    0.82 *
    localScale;
  const z =
    Math.cos(time * 0.36 + p[0] * 1.4 - p[1] * 1.1 + ph * 0.73) *
    amplitude *
    0.72 *
    localScale;

  return [x, y, z];
}

/**
 * Smooth, data-derived micro-flow for point clouds.
 * It intentionally moves each point differently so a cloud does not look
 * like one rigid object breathing in sync.
 */
export function animatePoints(
  scene: Scene3D,
  time: number,
  options: MotionV4Options = {},
): Scene3D {
  const profile = options.profile ?? "default";
  const profileScale = PROFILE_DEFAULTS[profile];
  const requested = clamp(finite(options.intensity, 1), 0, 1.5);
  const mobileScale = options.mobile ? 0.62 : 1;

  const amplitude = 0.012 * profileScale * requested * mobileScale;

  if (!scene.points?.length || amplitude <= 0) return scene;

  return {
    ...scene,
    points: scene.points.map((point, index) => {
      const phase = motionPhase(index, point.phase);
      const offset = dataMotionOffset(point.p, time, phase, amplitude);

      return {
        ...point,
        p: add(point.p, offset),
        // Preserve source size/alpha; only add a very small energy-linked glow.
        phase,
        glow: point.glow,
      };
    }),
  };
}

/**
 * Gives surface quads a small coherent ripple. engine3d.ts applies ripple
 * from the original quad vertices, so neighboring quads remain visually
 * connected.
 */
export function animateSurface(
  scene: Scene3D,
  options: MotionV4Options = {},
): Scene3D {
  const profile = options.profile ?? "surface";
  const profileScale = PROFILE_DEFAULTS[profile];
  const requested = clamp(finite(options.intensity, 1), 0, 1.5);
  const mobileScale = options.mobile ? 0.68 : 1;

  if (!scene.quads?.length) return scene;

  return {
    ...scene,
    quads: scene.quads.map((quad, index) => {
      const center: Vec3 = [
        (quad.pts[0][0] + quad.pts[1][0] + quad.pts[2][0] + quad.pts[3][0]) / 4,
        (quad.pts[0][1] + quad.pts[1][1] + quad.pts[2][1] + quad.pts[3][1]) / 4,
        (quad.pts[0][2] + quad.pts[1][2] + quad.pts[2][2] + quad.pts[3][2]) / 4,
      ];

      const yFactor = clamp(Math.abs(center[1]) * 0.65 + 0.35, 0.35, 1.2);
      const amplitude =
        0.028 * profileScale * requested * mobileScale * yFactor;

      return {
        ...quad,
        // Keep any existing explicit ripple, but never let V4 multiply it
        // into an excessive amplitude.
        ripple: clamp(
          Math.max(quad.ripple ?? 0, amplitude),
          0,
          0.055,
        ),
        pulse: quad.pulse ?? (index % 17) / 17,
      };
    }),
  };
}

/**
 * Main V4 scene decorator. It is intentionally stateless: the same scene and
 * time always produce the same frame, which makes it stable during React
 * rerenders and data refreshes.
 */
export function applyMotionV4(
  scene: Scene3D,
  frame: MotionV4Frame,
  options: MotionV4Options = {},
): Scene3D {
  const activity = clamp(finite(frame.activity, 1), 0, 1.5);
  const profile = options.profile ?? "default";
  const intensity = clamp(
    finite(options.intensity, 1) * activity,
    0,
    1.5,
  );

  let next = animatePoints(scene, frame.time, {
    ...options,
    profile,
    intensity,
  });

  if (profile === "surface" || profile === "cone") {
    next = animateSurface(next, {
      ...options,
      profile,
      intensity,
    });
  }

  return next;
}

/**
 * Converts pointer/interaction energy into a smooth motion activity value.
 * Used by Plot3D so motion reacts to interaction without suddenly jumping.
 */
export function interactionActivity(
  previous: number,
  pointerSpeed: number,
  dt: number,
): number {
  const target = clamp(pointerSpeed * 0.012, 0, 1);
  const response = 1 - Math.exp(-clamp(dt, 0, 0.05) * 8);
  return previous + (target - previous) * response;
}
