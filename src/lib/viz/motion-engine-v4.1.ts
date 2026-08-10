/**
 * Motion Engine V4.1 — Data-Reactive Motion Polish
 *
 * Upgrade dari V4:
 * - deterministic / tidak memakai Math.random()
 * - gerakan mengikuti struktur geometry
 * - activity dipengaruhi interaction + data energy
 * - directional flow kecil pada point cloud
 * - depth-aware micro motion
 * - regime/state response tanpa mengubah nilai data
 * - mobile scale otomatis lebih ringan
 *
 * Prinsip:
 *   DATA GEOMETRY
 *        ↓
 *   DATA SIGNAL
 *        ↓
 *   DIRECTIONAL MOTION
 *        ↓
 *   SMALL BOUNDED OFFSET
 *
 * Engine ini hanya menghasilkan visual motion. Nilai kuantitatif sumber
 * scene tidak dimodifikasi secara permanen.
 */

import type { Scene3D, Vec3 } from "@/lib/viz/engine3d";

export type MotionV41Profile =
  | "surface"
  | "cloud"
  | "frontier"
  | "cone"
  | "default";

export type MotionV41Options = {
  profile?: MotionV41Profile;
  intensity?: number;
  mobile?: boolean;
  stateEnergy?: number;
  interactionEnergy?: number;
};

export type MotionV41Frame = {
  time: number;
  dt: number;
  activity?: number;
};

const PROFILE_SCALE: Record<MotionV41Profile, number> = {
  surface: 1,
  cloud: 0.9,
  frontier: 0.72,
  cone: 0.95,
  default: 0.76,
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function finite(value: number, fallback = 0): number {
  return Number.isFinite(value) ? value : fallback;
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

function add(a: Vec3, b: Vec3): Vec3 {
  return [
    a[0] + b[0],
    a[1] + b[1],
    a[2] + b[2],
  ];
}

function scale(v: Vec3, amount: number): Vec3 {
  return [
    v[0] * amount,
    v[1] * amount,
    v[2] * amount,
  ];
}

/**
 * Stable phase. Golden-ratio spacing prevents synchronized motion without
 * introducing randomness.
 */
export function motionPhaseV41(
  index: number,
  suppliedPhase?: number,
): number {
  if (Number.isFinite(suppliedPhase)) {
    return suppliedPhase as number;
  }

  return (
    index * 0.6180339887498949
  ) % 1;
}

/**
 * Estimates a bounded "data energy" from geometry.
 *
 * This is intentionally generic because Scene3D does not expose a dedicated
 * market-state field. The signal is derived only from point position and
 * therefore works with the existing renderer without changing its API.
 */
export function geometryEnergy(
  p: Vec3,
): number {
  const radial = clamp(
    length(p) / 1.732,
    0,
    1,
  );

  const vertical = clamp(
    Math.abs(p[1]),
    0,
    1,
  );

  return clamp(
    radial * 0.62 +
      vertical * 0.38,
    0,
    1,
  );
}

/**
 * Directional flow.
 *
 * The flow direction is derived from the point's own coordinates. It creates
 * a subtle field-like movement instead of independent random floating.
 */
export function directionalFlowV41(
  p: Vec3,
  time: number,
  phase: number,
): Vec3 {
  const ph = phase * Math.PI * 2;

  const dx =
    Math.sin(
      time * 0.28 +
        p[2] * 1.35 +
        ph,
    ) +
    0.35 *
      Math.cos(
        p[1] * 2.4 -
          time * 0.16 +
          ph * 0.7,
      );

  const dy =
    Math.cos(
      time * 0.36 +
        p[0] * 1.15 +
        ph * 1.17,
    ) * 0.72;

  const dz =
    Math.sin(
      time * 0.24 +
        p[0] * 1.45 -
        p[1] * 1.1 +
        ph * 0.83,
    ) * 0.68;

  return normalize([
    dx,
    dy,
    dz,
  ]);
}

/**
 * Small bounded point displacement.
 *
 * The displacement contains:
 * - directional flow
 * - radial/depth modulation
 * - low-frequency breathing
 *
 * It never becomes large enough to destroy the original data geometry.
 */
export function dataReactiveOffsetV41(
  p: Vec3,
  time: number,
  phase: number,
  amplitude: number,
  stateEnergy: number,
): Vec3 {
  const localEnergy =
    geometryEnergy(p);

  const flow =
    directionalFlowV41(
      p,
      time,
      phase,
    );

  const depth =
    0.55 +
    0.45 *
      clamp(
        Math.abs(p[2]),
        0,
        1,
      );

  const slowWave =
    0.72 +
    0.28 *
      Math.sin(
        time * 0.22 +
          p[0] * 1.2 +
          p[2] * 0.8 +
          phase * Math.PI * 2,
      );

  const energy =
    0.65 +
    0.35 *
      clamp(
        localEnergy * 0.65 +
          stateEnergy * 0.35,
        0,
        1,
      );

  const amount =
    amplitude *
    depth *
    slowWave *
    energy;

  return scale(
    flow,
    amount,
  );
}

/**
 * Adds a subtle data-reactive motion layer to point clouds.
 */
export function animatePointsV41(
  scene: Scene3D,
  time: number,
  options: MotionV41Options = {},
): Scene3D {
  const profile =
    options.profile ??
    "default";

  const profileScale =
    PROFILE_SCALE[profile];

  const intensity = clamp(
    finite(
      options.intensity,
      1,
    ),
    0,
    1.5,
  );

  const mobileScale =
    options.mobile
      ? 0.52
      : 1;

  const stateEnergy = clamp(
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

  /*
   * Interaction can temporarily amplify motion, but only slightly.
   * The visual never "explodes" because of a swipe.
   */
  const interactionBoost =
    1 +
    interactionEnergy *
      0.28;

  const amplitude =
    0.0105 *
    profileScale *
    intensity *
    mobileScale *
    interactionBoost;

  if (
    !scene.points?.length ||
    amplitude <= 0
  ) {
    return scene;
  }

  return {
    ...scene,

    points: scene.points.map(
      (point, index) => {
        const phase =
          motionPhaseV41(
            index,
            point.phase,
          );

        const offset =
          dataReactiveOffsetV41(
            point.p,
            time,
            phase,
            amplitude,
            stateEnergy,
          );

        return {
          ...point,
          p: add(
            point.p,
            offset,
          ),
          phase,
        };
      },
    ),
  };
}

/**
 * Surface motion.
 *
 * V4.1 does not replace engine3d's existing ripple system. It only changes
 * the ripple amount in a bounded way, using quad position as the signal.
 */
export function animateSurfaceV41(
  scene: Scene3D,
  time: number,
  options: MotionV41Options = {},
): Scene3D {
  if (!scene.quads?.length) {
    return scene;
  }

  const intensity = clamp(
    finite(
      options.intensity,
      1,
    ),
    0,
    1.5,
  );

  const profileScale =
    PROFILE_SCALE[
      options.profile ??
        "surface"
    ];

  const mobileScale =
    options.mobile
      ? 0.58
      : 1;

  const stateEnergy = clamp(
    finite(
      options.stateEnergy,
      0.5,
    ),
    0,
    1,
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
          motionPhaseV41(
            index,
          );

        const spatial =
          geometryEnergy(
            center,
          );

        const wave =
          0.5 +
          0.5 *
            Math.sin(
              time * 0.32 +
                center[0] * 1.4 +
                center[2] * 1.15 +
                phase *
                  Math.PI *
                  2,
            );

        const ripple =
          0.008 +
          0.020 *
            profileScale *
            intensity *
            mobileScale *
            (
              0.55 +
              0.45 *
                spatial
            ) *
            (
              0.65 +
              0.35 *
                stateEnergy
            ) *
            wave;

        return {
          ...quad,

          ripple: clamp(
            Math.max(
              quad.ripple ?? 0,
              ripple,
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
 * Smoothly converts interaction speed into an activity value.
 */
export function updateInteractionEnergyV41(
  previous: number,
  pointerSpeed: number,
  dt: number,
): number {
  const target =
    clamp(
      pointerSpeed *
        0.012,
      0,
      1,
    );

  const response =
    1 -
    Math.exp(
      -clamp(
        dt,
        0,
        0.05,
      ) * 7,
    );

  return (
    previous +
    (target - previous) *
      response
  );
}

/**
 * Main V4.1 decorator.
 */
export function applyMotionV41(
  scene: Scene3D,
  frame: MotionV41Frame,
  options: MotionV41Options = {},
): Scene3D {
  const baseActivity =
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

  const intensity =
    clamp(
      finite(
        options.intensity,
        1,
      ) *
        (
          0.72 +
          0.28 *
            baseActivity
        ) *
        (
          0.84 +
          0.16 *
            stateEnergy
        ),
      0,
      1.5,
    );

  const profile =
    options.profile ??
    "default";

  let next =
    animatePointsV41(
      scene,
      frame.time,
      {
        ...options,
        profile,
        intensity,
        stateEnergy,
        interactionEnergy,
      },
    );

  if (
    profile ===
      "surface" ||
    profile ===
      "cone"
  ) {
    next =
      animateSurfaceV41(
        next,
        frame.time,
        {
          ...options,
          profile,
          intensity,
          stateEnergy,
          interactionEnergy,
        },
      );
  }

  return next;
}
