/**
 * Motion Engine V2
 *
 * Data-driven particle simulation for the Hidden State Space.
 *
 * Design:
 *   data point -> target position -> velocity -> damping -> rendered position
 *
 * This file intentionally has no dependency on React, canvas, or random values.
 * It can therefore be tested independently before being connected to Plot3D.
 */

export type HiddenStatePoint = {
  ret: number;
  vol: number;
  state: number;
};

export type MotionParticle = {
  position: [number, number, number];
  velocity: [number, number, number];
  target: [number, number, number];
  state: number;
  sourceIndex: number;
};

export type HiddenStateMotionOptions = {
  /** How strongly particles follow the new data target. */
  stiffness?: number;
  /** How quickly velocity is damped. */
  damping?: number;
  /** Maximum movement per second on each axis. */
  maxSpeed?: number;
};

const DEFAULTS: Required<HiddenStateMotionOptions> = {
  stiffness: 7.5,
  damping: 5.2,
  maxSpeed: 2.8,
};

function finite(value: number, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function normalizeRange(
  value: number,
  min: number,
  max: number,
): number {
  const safeValue = finite(value);
  const span = max - min;

  if (!Number.isFinite(span) || Math.abs(span) < 1e-12) {
    return 0;
  }

  return clamp(((safeValue - min) / span) * 2 - 1, -1, 1);
}

function range(values: number[]) {
  let min = Infinity;
  let max = -Infinity;

  for (const value of values) {
    if (!Number.isFinite(value)) {
      continue;
    }

    min = Math.min(min, value);
    max = Math.max(max, value);
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }

  return { min, max };
}

/**
 * Maps the actual HMM state points into the 3D state-space coordinate system.
 *
 * X = return
 * Y = volatility
 * Z = chronological position
 *
 * State is NOT used to invent coordinates. It is metadata used by the
 * renderer to select regime appearance.
 */
export function mapHiddenStateTargets(
  data: HiddenStatePoint[],
): Array<{
  target: [number, number, number];
  state: number;
  sourceIndex: number;
}> {
  if (data.length === 0) {
    return [];
  }

  const returns = data.map((point) => finite(point.ret));
  const volatility = data.map((point) => finite(point.vol));

  const returnRange = range(returns);
  const volatilityRange = range(volatility);

  const lastIndex = Math.max(data.length - 1, 1);

  return data.map((point, index) => ({
    target: [
      normalizeRange(point.ret, returnRange.min, returnRange.max),
      normalizeRange(point.vol, volatilityRange.min, volatilityRange.max),
      (index / lastIndex) * 2 - 1,
    ],
    state: Number.isFinite(point.state) ? point.state : 0,
    sourceIndex: index,
  }));
}

function integrateAxis(
  position: number,
  velocity: number,
  target: number,
  dt: number,
  stiffness: number,
  damping: number,
  maxSpeed: number,
): [number, number] {
  const acceleration =
    (target - position) * stiffness - velocity * damping;

  let nextVelocity = velocity + acceleration * dt;
  nextVelocity = clamp(nextVelocity, -maxSpeed, maxSpeed);

  let nextPosition = position + nextVelocity * dt;

  // Prevent overshoot from making the cloud visibly oscillate around a
  // stationary data point.
  const targetDistance = target - position;
  const nextDistance = target - nextPosition;

  if (
    Math.sign(targetDistance) !== Math.sign(nextDistance) &&
    Math.abs(targetDistance) < Math.abs(nextVelocity * dt) * 1.5
  ) {
    nextPosition = target;
    nextVelocity = 0;
  }

  return [nextPosition, nextVelocity];
}

export function createHiddenStateParticles(
  data: HiddenStatePoint[],
): MotionParticle[] {
  const targets = mapHiddenStateTargets(data);

  return targets.map(({ target, state, sourceIndex }) => ({
    position: [...target],
    velocity: [0, 0, 0],
    target: [...target],
    state,
    sourceIndex,
  }));
}

/**
 * Updates particles toward the current HMM dataset.
 *
 * There is deliberately no Math.random() here.
 * Every target comes from `statePoints`.
 */
export function stepHiddenStateMotion(
  particles: MotionParticle[],
  data: HiddenStatePoint[],
  deltaSeconds: number,
  options: HiddenStateMotionOptions = {},
): MotionParticle[] {
  const config = {
    ...DEFAULTS,
    ...options,
  };

  const dt = clamp(
    finite(deltaSeconds),
    0,
    0.05,
  );

  const targets = mapHiddenStateTargets(data);

  if (targets.length === 0) {
    return particles;
  }

  const next: MotionParticle[] = [];

  for (let index = 0; index < targets.length; index += 1) {
    const target = targets[index];

    if (target === undefined) {
      continue;
    }

    const previous = particles[index];

    if (previous === undefined) {
      next.push({
        position: [...target.target],
        velocity: [0, 0, 0],
        target: [...target.target],
        state: target.state,
        sourceIndex: target.sourceIndex,
      });
      continue;
    }

    const [x, vx] = integrateAxis(
      previous.position[0],
      previous.velocity[0],
      target.target[0],
      dt,
      config.stiffness,
      config.damping,
      config.maxSpeed,
    );

    const [y, vy] = integrateAxis(
      previous.position[1],
      previous.velocity[1],
      target.target[1],
      dt,
      config.stiffness,
      config.damping,
      config.maxSpeed,
    );

    const [z, vz] = integrateAxis(
      previous.position[2],
      previous.velocity[2],
      target.target[2],
      dt,
      config.stiffness,
      config.damping,
      config.maxSpeed,
    );

    next.push({
      position: [x, y, z],
      velocity: [vx, vy, vz],
      target: [...target.target],
      state: target.state,
      sourceIndex: target.sourceIndex,
    });
  }

  return next;
}

/**
 * Returns a simple motion-energy value that can be used by the renderer for
 * subtle trail/glow intensity.
 *
 * This is derived from actual particle velocity, not a timer or random value.
 */
export function getMotionEnergy(
  particle: MotionParticle,
): number {
  const speedSquared =
    particle.velocity[0] * particle.velocity[0] +
    particle.velocity[1] * particle.velocity[1] +
    particle.velocity[2] * particle.velocity[2];

  return clamp(Math.sqrt(speedSquared) / 2.8, 0, 1);
}
