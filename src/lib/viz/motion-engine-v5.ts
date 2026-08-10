/**
 * Motion Engine V5 — Data Flow Engine
 *
 * V5 sits on top of the stable V4/V4.1 motion model.
 *
 * Design goals:
 * - movement should read as FLOW, not random jitter
 * - original data geometry remains the anchor
 * - local points receive related directional motion
 * - motion is deterministic
 * - mobile uses lower particle count/intensity supplied by the caller
 * - no changes to engine3d.ts are required
 *
 * The engine exposes pure helpers so Plot3D.tsx can decide when/how often
 * to evaluate the motion.
 */

import type { Scene3D, Vec3 } from "@/lib/viz/engine3d";

export type FlowProfile =
  | "hmm"
  | "cloud"
  | "surface"
  | "frontier"
  | "default";

export interface FlowOptions {
  profile?: FlowProfile;
  intensity?: number;
  mobile?: boolean;
  stateEnergy?: number;
  interactionEnergy?: number;
  time?: number;
}

export interface FlowFrame {
  time: number;
  dt: number;
  activity?: number;
}

const PROFILE_INTENSITY: Record<FlowProfile, number> = {
  hmm: 1,
  cloud: 0.9,
  surface: 0.82,
  frontier: 0.62,
  default: 0.72,
};

const GOLDEN = 0.6180339887498949;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function finite(value: number, fallback = 0): number {
  return Number.isFinite(value) ? value : fallback;
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function length(v: Vec3): number {
  return Math.sqrt(
    v[0] * v[0] +
    v[1] * v[1] +
    v[2] * v[2],
  );
}

function normalize(v: Vec3): Vec3 {
  const len = length(v);
  if (len < 1e-6) return [0, 0, 0];

  return [
    v[0] / len,
    v[1] / len,
    v[2] / len,
  ];
}

function smoothstep(t: number): number {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

/**
 * Stable per-point phase.
 *
 * No randomness: the same dataset produces the same motion layout.
 */
export function flowPhaseV5(
  index: number,
  phase?: number,
): number {
  if (Number.isFinite(phase)) {
    return phase as number;
  }

  return (index * GOLDEN) % 1;
}

/**
 * Estimate the point's normalized "energy".
 *
 * This is deliberately geometry-derived so the engine remains compatible
 * with the existing Scene3D interface.
 */
export function pointEnergyV5(p: Vec3): number {
  const radial = clamp(
    length(p) / 1.7320508,
    0,
    1,
  );

  const vertical = clamp(
    Math.abs(p[1]),
    0,
    1,
  );

  return clamp(
    radial * 0.55 +
    vertical * 0.45,
    0,
    1,
  );
}

/**
 * Local directional field.
 *
 * Instead of moving each particle independently, the direction changes
 * smoothly with position. Neighboring points therefore tend to travel
 * together, producing an actual "flow" impression.
 */
export function flowFieldV5(
  p: Vec3,
  time: number,
  phase: number,
): Vec3 {
  const phaseRad = phase * Math.PI * 2;

  const x = p[0];
  const y = p[1];
  const z = p[2];

  const fx =
    0.72 *
      Math.sin(
        time * 0.24 +
        z * 1.25 +
        phaseRad * 0.35,
      ) +
    0.28 *
      Math.cos(
        y * 1.65 -
        time * 0.13 +
        phaseRad,
      );

  const fy =
    0.62 *
      Math.cos(
        time * 0.31 +
        x * 1.15 +
        phaseRad * 0.72,
      ) +
    0.18 *
      Math.sin(
        z * 2.0 +
        time * 0.09,
      );

  const fz =
    0.68 *
      Math.sin(
        time * 0.21 +
        x * 1.35 -
        y * 0.85 +
        phaseRad * 0.58,
      ) +
    0.22 *
      Math.cos(
        z * 1.8 -
        time * 0.11,
      );

  return normalize([fx, fy, fz]);
}

/**
 * Slow global drift gives the whole cloud a coherent current.
 */
export function globalCurrentV5(
  time: number,
  profile: FlowProfile,
): Vec3 {
  if (profile === "hmm" || profile === "cloud") {
    return normalize([
      Math.sin(time * 0.12),
      0.22 * Math.cos(time * 0.17),
      Math.cos(time * 0.10),
    ]);
  }

  if (profile === "surface") {
    return normalize([
      0.82,
      0.18 * Math.sin(time * 0.14),
      0.36,
    ]);
  }

  return normalize([
    0.55,
    0.12 * Math.cos(time * 0.11),
    0.18,
  ]);
}

/**
 * Local coherence pulse.
 *
 * Neighboring points get related speed while their individual phase remains
 * slightly different. This is what separates V5 from simple jitter.
 */
export function flowPulseV5(
  p: Vec3,
  time: number,
  phase: number,
): number {
  return (
    0.62 +
    0.38 *
      (
        0.5 +
        0.5 *
          Math.sin(
            time * 0.27 +
            p[0] * 0.9 +
            p[2] * 0.75 +
            phase * Math.PI * 2,
          )
      )
  );
}

/**
 * Bounded flow offset.
 *
 * The maximum displacement is intentionally small so the visualization
 * continues to represent the original quantitative geometry.
 */
export function flowOffsetV5(
  p: Vec3,
  index: number,
  time: number,
  amplitude: number,
  profile: FlowProfile,
  stateEnergy: number,
): Vec3 {
  const phase = flowPhaseV5(index);
  const local = flowFieldV5(
    p,
    time,
    phase,
  );

  const current = globalCurrentV5(
    time,
    profile,
  );

  const localEnergy = pointEnergyV5(p);

  const pulse = flowPulseV5(
    p,
    time,
    phase,
  );

  const stateScale =
    0.76 +
    0.24 * clamp(stateEnergy, 0, 1);

  const depthScale =
    0.68 +
    0.32 *
      clamp(
        0.5 + 0.5 * p[2],
        0,
        1,
      );

  const direction = normalize(
    add(
      scale(local, 0.76),
      scale(current, 0.24),
    ),
  );

  const amount =
    amplitude *
    stateScale *
    depthScale *
    pulse *
    (
      0.72 +
      0.28 * localEnergy
    );

  return scale(
    direction,
    amount,
  );
}

/**
 * Apply coherent flow to point geometry.
 *
 * This is a visual-only decorator. It returns a new scene and does not mutate
 * the source scene.
 */
export function applyFlowV5(
  scene: Scene3D,
  frame: FlowFrame,
  options: FlowOptions = {},
): Scene3D {
  const profile =
    options.profile ?? "default";

  const baseIntensity =
    finite(
      options.intensity,
      1,
    );

  const profileIntensity =
    PROFILE_INTENSITY[profile];

  const activity =
    clamp(
      finite(
        frame.activity,
        0.65,
      ),
      0,
      1.5,
    );

  const stateEnergy =
    clamp(
      finite(
        options.stateEnergy,
        0.5,
      ),
      0,
      1,
    );

  const interactionEnergy =
    clamp(
      finite(
        options.interactionEnergy,
        0,
      ),
      0,
      1,
    );

  const mobileScale =
    options.mobile
      ? 0.52
      : 1;

  /*
   * Interaction temporarily increases flow, but only by a small amount.
   * This preserves the stable V4 gesture behavior.
   */
  const interactionBoost =
    1 +
    interactionEnergy * 0.22;

  const amplitude =
    0.019 *
    profileIntensity *
    clamp(
      baseIntensity,
      0,
      1.4,
    ) *
    (
      0.76 +
      0.24 * activity
    ) *
    mobileScale *
    interactionBoost;

  if (
    amplitude <= 0 ||
    !scene.points?.length
  ) {
    return scene;
  }

  return {
    ...scene,

    points: scene.points.map(
      (point, index) => {
        const offset =
          flowOffsetV5(
            point.p,
            index,
            frame.time,
            amplitude,
            profile,
            stateEnergy,
          );

        return {
          ...point,
          p: add(
            point.p,
            offset,
          ),
          phase:
            flowPhaseV5(
              index,
              point.phase,
            ),
        };
      },
    ),
  };
}

/**
 * Moving wave for surface geometry.
 *
 * The wave travels across the surface instead of simply pulsing every quad
 * at the same time.
 */
export function applySurfaceFlowV5(
  scene: Scene3D,
  frame: FlowFrame,
  options: FlowOptions = {},
): Scene3D {
  if (!scene.quads?.length) {
    return scene;
  }

  const mobileScale =
    options.mobile
      ? 0.55
      : 1;

  const stateEnergy =
    clamp(
      finite(
        options.stateEnergy,
        0.5,
      ),
      0,
      1,
    );

  const intensity =
    clamp(
      finite(
        options.intensity,
        1,
      ),
      0,
      1.4,
    );

  const waveSpeed =
    options.mobile
      ? 0.22
      : 0.30;

  const waveAmplitude =
    0.010 +
    0.022 *
      intensity *
      mobileScale *
      (
        0.72 +
        0.28 * stateEnergy
      );

  return {
    ...scene,

    quads: scene.quads.map(
      (quad, index) => {
        const center: Vec3 = [
          (
            quad.pts[0][0] +
            quad.pts[1][0] +
            quad.pts[2][0] +
            quad.pts[3][0]
          ) / 4,

          (
            quad.pts[0][1] +
            quad.pts[1][1] +
            quad.pts[2][1] +
            quad.pts[3][1]
          ) / 4,

          (
            quad.pts[0][2] +
            quad.pts[1][2] +
            quad.pts[2][2] +
            quad.pts[3][2]
          ) / 4,
        ];

        const phase =
          flowPhaseV5(index);

        const travellingWave =
          0.5 +
          0.5 *
            Math.sin(
              frame.time *
                waveSpeed +
              center[0] * 1.7 +
              center[2] * 1.15 +
              phase * Math.PI * 2,
            );

        return {
          ...quad,
          ripple: clamp(
            Math.max(
              quad.ripple ?? 0,
              waveAmplitude *
                (
                  0.65 +
                  0.35 *
                    travellingWave
                ),
            ),
            0,
            0.05,
          ),
        };
      },
    ),
  };
}

/**
 * Combined V5 decorator.
 */
export function animateV5(
  scene: Scene3D,
  frame: FlowFrame,
  options: FlowOptions = {},
): Scene3D {
  const profile =
    options.profile ?? "default";

  let next =
    applyFlowV5(
      scene,
      frame,
      options,
    );

  if (
    profile === "surface"
  ) {
    next =
      applySurfaceFlowV5(
        next,
        frame,
        options,
      );
  }

  return next;
}

/**
 * Convert pointer/drag speed into a small flow activity value.
 */
export function updateFlowInteractionV5(
  previous: number,
  pointerSpeed: number,
  dt: number,
): number {
  const target =
    clamp(
      pointerSpeed * 0.010,
      0,
      1,
    );

  const response =
    1 -
    Math.exp(
      -clamp(dt, 0, 0.05) * 8,
    );

  return clamp(
    previous +
      (
        target -
        previous
      ) *
        response,
    0,
    1,
  );
}
