/**
 * Motion Engine V2
 *
 * Data-driven particle simulation for the Hidden State Space.
 *
 * Design:
 *   data/scene geometry -> target position -> acceleration -> velocity
 *   -> damping -> rendered position
 *
 * No Math.random() is used. Motion is derived from actual scene geometry.
 */

import type { Point3, Scene3D, Vec3 } from "@/lib/viz/engine3d";

export type HiddenStatePoint = {
  ret: number;
  vol: number;
  state: number;
};

export type MotionParticle = {
  position: Vec3;
  velocity: Vec3;
  target: Vec3;
  state: number;
  sourceIndex: number;
};

export type HiddenStateMotionOptions = {
  stiffness?: number;
  damping?: number;
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

function stepParticle(
  previous: MotionParticle | undefined,
  target: Vec3,
  state: number,
  sourceIndex: number,
  dt: number,
  options: Required<HiddenStateMotionOptions>,
): MotionParticle {
  if (previous === undefined) {
    return {
      position: [...target],
      velocity: [0, 0, 0],
      target: [...target],
      state,
      sourceIndex,
    };
  }

  const [x, vx] = integrateAxis(
    previous.position[0],
    previous.velocity[0],
    target[0],
    dt,
    options.stiffness,
    options.damping,
    options.maxSpeed,
  );

  const [y, vy] = integrateAxis(
    previous.position[1],
    previous.velocity[1],
    target[1],
    dt,
    options.stiffness,
    options.damping,
    options.maxSpeed,
  );

  const [z, vz] = integrateAxis(
    previous.position[2],
    previous.velocity[2],
    target[2],
    dt,
    options.stiffness,
    options.damping,
    options.maxSpeed,
  );

  return {
    position: [x, y, z],
    velocity: [vx, vy, vz],
    target: [...target],
    state,
    sourceIndex,
  };
}

export function createHiddenStateParticles(
  data: HiddenStatePoint[],
): MotionParticle[] {
  return data.map((point, index) => ({
    position: [
      finite(point.ret),
      finite(point.vol),
      index,
    ],
    velocity: [0, 0, 0],
    target: [
      finite(point.ret),
      finite(point.vol),
      index,
    ],
    state: Number.isFinite(point.state) ? point.state : 0,
    sourceIndex: index,
  }));
}

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

  const dt = clamp(finite(deltaSeconds), 0, 0.05);

  return data.map((point, index) =>
    stepParticle(
      particles[index],
      [
        finite(point.ret),
        finite(point.vol),
        index,
      ],
      Number.isFinite(point.state) ? point.state : 0,
      index,
      dt,
      config,
    ),
  );
}

/**
 * Scene bridge used by Plot3D.
 *
 * Plot3D receives a Scene3D rather than raw statePoints. The HMM scene already
 * contains coordinates generated from the actual return/volatility/time data,
 * so those coordinates become the particle targets here.
 */
export function createHiddenSceneParticles(
  scene: Scene3D,
): MotionParticle[] {
  return (scene.points ?? []).map((point, index) => ({
    position: [...point.p],
    velocity: [0, 0, 0],
    target: [...point.p],
    state: index,
    sourceIndex: index,
  }));
}

export function stepHiddenSceneMotion(
  particles: MotionParticle[],
  scene: Scene3D,
  deltaSeconds: number,
  options: HiddenStateMotionOptions = {},
): MotionParticle[] {
  const config = {
    ...DEFAULTS,
    ...options,
  };

  const dt = clamp(finite(deltaSeconds), 0, 0.05);
  const points = scene.points ?? [];

  return points.map((point, index) =>
    stepParticle(
      particles[index],
      point.p,
      index,
      index,
      dt,
      config,
    ),
  );
}

export function applyHiddenSceneMotion(
  scene: Scene3D,
  particles: MotionParticle[],
): Scene3D {
  const source = scene.points ?? [];

  if (source.length === 0 || particles.length === 0) {
    return scene;
  }

  const points: Point3[] = source.map((point, index) => {
    const particle = particles[index];

    if (particle === undefined) {
      return point;
    }

    const speed = Math.sqrt(
      particle.velocity[0] * particle.velocity[0] +
        particle.velocity[1] * particle.velocity[1] +
        particle.velocity[2] * particle.velocity[2],
    );

    const motionEnergy = clamp(speed / 2.8, 0, 1);

    return {
      ...point,
      p: particle.position,
      alpha: Math.min(
        1,
        (point.alpha ?? 1) * (0.88 + motionEnergy * 0.12),
      ),
      size: (point.size ?? 1.7) * (1 + motionEnergy * 0.12),
    };
  });

  return {
    ...scene,
    points,
  };
}

export function getMotionEnergy(
  particle: MotionParticle,
): number {
  const speed = Math.sqrt(
    particle.velocity[0] * particle.velocity[0] +
      particle.velocity[1] * particle.velocity[1] +
      particle.velocity[2] * particle.velocity[2],
  );

  return clamp(speed / 2.8, 0, 1);
}
