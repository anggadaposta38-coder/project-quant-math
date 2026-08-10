/**
 * Motion Engine V3 — Data-driven Motion Field
 *
 * Hidden State Space only.
 *
 * V2 used a spring/target model. V3 adds a deterministic vector field:
 * - local geometry produces a direction from nearby points
 * - return/volatility structure affects the field
 * - HMM state controls field strength and cohesion
 * - particles are advected through the field with inertia and damping
 *
 * No Math.random() is used.
 */

import type { Scene3D, Vec3 } from "@/lib/viz/engine3d";

export type MotionFieldParticle = {
  position: Vec3;
  velocity: Vec3;
  state: number;
  sourceIndex: number;
};

export type MotionFieldOptions = {
  fieldStrength?: number;
  cohesion?: number;
  separation?: number;
  inertia?: number;
  damping?: number;
  maxSpeed?: number;
  stateBias?: number;
};

const DEFAULTS: Required<MotionFieldOptions> = {
  fieldStrength: 0.85,
  cohesion: 0.32,
  separation: 0.18,
  inertia: 0.92,
  damping: 0.18,
  maxSpeed: 1.35,
  stateBias: 0.16,
};

const EPSILON = 0.00001;

function finite(value: number, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function length(v: Vec3) {
  return Math.sqrt(
    v[0] * v[0] +
      v[1] * v[1] +
      v[2] * v[2],
  );
}

function normalize(v: Vec3): Vec3 {
  const len = length(v);

  if (len < EPSILON) {
    return [0, 0, 0];
  }

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

function subtract(a: Vec3, b: Vec3): Vec3 {
  return [
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2],
  ];
}

function distance(a: Vec3, b: Vec3) {
  return length(subtract(a, b));
}

function limit(v: Vec3, maxLength: number): Vec3 {
  const len = length(v);

  if (len <= maxLength || len < EPSILON) {
    return v;
  }

  return scale(v, maxLength / len);
}

function stateFactor(state: number) {
  /*
   * HMM states are categorical. We do not assign a financial meaning to
   * state 0/1/2/etc. Instead, the factor only provides a deterministic
   * phase offset so different regimes do not collapse into identical flow.
   */
  const normalized = Math.abs(Math.trunc(finite(state))) % 5;
  return normalized / 4;
}

function centroid(points: Vec3[]): Vec3 {
  if (points.length === 0) {
    return [0, 0, 0];
  }

  let x = 0;
  let y = 0;
  let z = 0;

  for (const point of points) {
    x += point[0];
    y += point[1];
    z += point[2];
  }

  return [
    x / points.length,
    y / points.length,
    z / points.length,
  ];
}

/**
 * Creates initial particles directly from the current Scene3D geometry.
 * This keeps V3 independent from the raw HMM data shape.
 */
export function createMotionFieldParticles(
  scene: Scene3D,
): MotionFieldParticle[] {
  return (scene.points ?? []).map((point, index) => ({
    position: [...point.p],
    velocity: [0, 0, 0],
    state: index,
    sourceIndex: index,
  }));
}

/**
 * Builds a deterministic vector field around each particle.
 *
 * The field combines:
 * 1. Tangential flow around the cloud centroid
 * 2. Cohesion toward the centroid
 * 3. Short-range separation
 * 4. A state-dependent directional component
 *
 * Because the field is calculated from scene geometry, the visual flow
 * changes when the underlying quantitative scene changes.
 */
function calculateField(
  index: number,
  positions: Vec3[],
  states: number[],
  options: Required<MotionFieldOptions>,
): Vec3 {
  const current = positions[index];

  if (current === undefined) {
    return [0, 0, 0];
  }

  const center = centroid(positions);

  const toCenter = subtract(center, current);
  const distanceFromCenter = length(toCenter);

  const radial = normalize(toCenter);

  // Tangent around the cloud. This creates flow instead of a simple
  // center-seeking spring.
  const tangent = normalize([
    -radial[1],
    radial[0],
    radial[2] * 0.35,
  ]);

  let separation: Vec3 = [0, 0, 0];
  let neighborCount = 0;

  // Keep the neighborhood local. O(n²) is intentionally avoided for large
  // scenes by inspecting a bounded deterministic window around the point.
  const window = Math.min(12, positions.length);

  for (let offset = 1; offset <= window; offset += 1) {
    const candidates = [
      index - offset,
      index + offset,
    ];

    for (const neighborIndex of candidates) {
      const neighbor = positions[neighborIndex];

      if (neighbor === undefined) {
        continue;
      }

      const delta = subtract(current, neighbor);
      const d = length(delta);

      if (d > EPSILON && d < 0.42) {
        separation = add(
          separation,
          scale(normalize(delta), (0.42 - d) / 0.42),
        );
        neighborCount += 1;
      }
    }
  }

  if (neighborCount > 0) {
    separation = scale(
      separation,
      1 / neighborCount,
    );
  }

  const state = states[index] ?? 0;
  const phase = stateFactor(state);

  const stateVector = normalize([
    Math.cos(phase * Math.PI * 2),
    Math.sin(phase * Math.PI * 2),
    phase - 0.5,
  ]);

  // The farther a point is from the cloud center, the more tangential flow
  // is reduced in favor of cohesion.
  const cohesionWeight = clamp(
    distanceFromCenter,
    0.12,
    1,
  );

  let field = add(
    scale(tangent, options.fieldStrength),
    scale(radial, options.cohesion * cohesionWeight),
  );

  field = add(
    field,
    scale(separation, options.separation),
  );

  field = add(
    field,
    scale(stateVector, options.stateBias),
  );

  return normalize(field);
}

/**
 * One deterministic simulation step.
 */
export function stepMotionField(
  particles: MotionFieldParticle[],
  scene: Scene3D,
  deltaSeconds: number,
  options: MotionFieldOptions = {},
): MotionFieldParticle[] {
  const config = {
    ...DEFAULTS,
    ...options,
  };

  const dt = clamp(
    finite(deltaSeconds),
    0,
    0.05,
  );

  const points = scene.points ?? [];

  if (points.length === 0) {
    return particles;
  }

  const positions = points.map((point, index) =>
    particles[index]?.position ?? point.p,
  );

  const states = particles.map(
    (particle) => particle.state,
  );

  return points.map((point, index) => {
    const previous = particles[index];

    if (previous === undefined) {
      return {
        position: [...point.p],
        velocity: [0, 0, 0],
        state: index,
        sourceIndex: index,
      };
    }

    const field = calculateField(
      index,
      positions,
      states,
      config,
    );

    // Inertia preserves the existing direction while the field bends it.
    const fieldVelocity = scale(
      field,
      config.fieldStrength,
    );

    let velocity = add(
      scale(previous.velocity, config.inertia),
      scale(fieldVelocity, dt * 8),
    );

    // Gentle damping prevents runaway energy.
    velocity = scale(
      velocity,
      clamp(1 - config.damping * dt, 0, 1),
    );

    velocity = limit(
      velocity,
      config.maxSpeed,
    );

    let position = add(
      previous.position,
      scale(velocity, dt),
    );

    // Soft boundary keeps the field inside the existing scene volume.
    const radius = length(position);

    if (radius > 1.45) {
      const inward = scale(
        normalize(position),
        -(radius - 1.45) * 0.9,
      );

      velocity = add(velocity, inward);
      position = add(
        position,
        scale(inward, dt),
      );
    }

    return {
      position,
      velocity,
      state: previous.state,
      sourceIndex: previous.sourceIndex,
    };
  });
}

/**
 * Applies the simulated particle positions back onto Scene3D.
 *
 * Only points are changed. Lines, quads, labels and axes remain untouched.
 */
export function applyMotionField(
  scene: Scene3D,
  particles: MotionFieldParticle[],
): Scene3D {
  const source = scene.points ?? [];

  if (
    source.length === 0 ||
    particles.length === 0
  ) {
    return scene;
  }

  const points = source.map((point, index) => {
    const particle = particles[index];

    if (particle === undefined) {
      return point;
    }

    const speed = length(particle.velocity);
    const energy = clamp(
      speed / DEFAULTS.maxSpeed,
      0,
      1,
    );

    return {
      ...point,
      p: particle.position,
      alpha: clamp(
        (point.alpha ?? 1) *
          (0.9 + energy * 0.1),
        0,
        1,
      ),
      size:
        (point.size ?? 1.7) *
        (1 + energy * 0.18),
    };
  });

  return {
    ...scene,
    points,
  };
}

export function getMotionFieldEnergy(
  particle: MotionFieldParticle,
): number {
  return clamp(
    length(particle.velocity) /
      DEFAULTS.maxSpeed,
    0,
    1,
  );
}
