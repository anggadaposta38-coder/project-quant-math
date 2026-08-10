import type { Scene3D, Vec3 } from "@/lib/viz/engine3d";

export type MotionV42Profile =
  | "default"
  | "soft"
  | "dynamic"
  | "pulse"
  | "data";

export interface MotionV42Clock {
  time: number;
  dt: number;
  activity?: number;
}

export interface MotionV42Options {
  profile?: MotionV42Profile;
  intensity?: number;
  mobile?: boolean;
}

/*
 * V4.2
 *
 * Prinsip:
 * - tidak membuat data baru
 * - tidak menambah jumlah primitive
 * - hanya mendeformasi geometry yang sudah ada
 * - motion berasal dari posisi geometry + waktu
 * - setiap primitive diberi phase deterministik
 * - mobile menggunakan amplitudo dan workload lebih rendah
 */

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function hash(n: number) {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function smoothstep(t: number) {
  return t * t * (3 - 2 * t);
}

function profileSettings(profile: MotionV42Profile) {
  switch (profile) {
    case "soft":
      return {
        wave: 0.55,
        pulse: 0.35,
        drift: 0.45,
        frequency: 0.65,
      };

    case "dynamic":
      return {
        wave: 1.15,
        pulse: 0.85,
        drift: 0.9,
        frequency: 1.05,
      };

    case "pulse":
      return {
        wave: 0.65,
        pulse: 1.25,
        drift: 0.5,
        frequency: 0.9,
      };

    case "data":
      return {
        wave: 0.85,
        pulse: 0.7,
        drift: 0.8,
        frequency: 0.78,
      };

    default:
      return {
        wave: 0.8,
        pulse: 0.55,
        drift: 0.65,
        frequency: 0.78,
      };
  }
}

function motionStrength(
  index: number,
  x: number,
  y: number,
  z: number,
  time: number,
  settings: ReturnType<typeof profileSettings>,
  intensity: number,
  activity: number,
) {
  const seed = index * 0.173 + x * 0.41 + y * 0.29 + z * 0.17;

  const phase = hash(seed) * Math.PI * 2;

  /*
   * Multi-scale wave.
   *
   * Ini membuat permukaan/cloud/trajectory terasa seperti
   * mempunyai dinamika internal, bukan sekadar naik-turun global.
   */
  const waveA = Math.sin(
    time * 0.72 * settings.frequency +
      x * 1.7 +
      z * 1.15 +
      phase,
  );

  const waveB = Math.sin(
    time * 0.37 * settings.frequency -
      y * 1.25 +
      x * 0.65 +
      phase * 0.47,
  );

  const pulse =
    Math.sin(
      time * 0.95 * settings.frequency +
        phase +
        x * 0.5,
    ) *
    0.5 +
    0.5;

  const localEnergy =
    0.55 +
    Math.abs(waveA) * 0.3 +
    pulse * 0.15;

  return {
    phase,
    waveA,
    waveB,
    pulse,
    localEnergy,
    amount:
      intensity *
      activity *
      settings.wave *
      (0.55 + localEnergy * 0.45),
  };
}

function animatePoint(
  point: Vec3,
  index: number,
  time: number,
  settings: ReturnType<typeof profileSettings>,
  intensity: number,
  activity: number,
  mobile: boolean,
): Vec3 {
  const [x, y, z] = point;

  const motion = motionStrength(
    index,
    x,
    y,
    z,
    time,
    settings,
    intensity,
    activity,
  );

  /*
   * Mobile:
   * amplitude diperkecil, bukan animasi dimatikan.
   */
  const mobileFactor = mobile ? 0.58 : 1;

  const amount = motion.amount * mobileFactor;

  /*
   * Wave mengikuti posisi.
   *
   * x/z -> lateral movement
   * y   -> vertical/data movement
   */
  const vertical =
    motion.waveA *
    0.020 *
    amount *
    settings.drift;

  const lateral =
    motion.waveB *
    0.014 *
    amount;

  const depth =
    Math.cos(motion.phase + time * 0.42) *
    0.012 *
    amount;

  /*
   * Pulse sangat kecil agar tidak terlihat seperti random jitter.
   */
  const pulse =
    (motion.pulse - 0.5) *
    0.018 *
    settings.pulse *
    amount;

  return [
    x + lateral,
    y + vertical + pulse,
    z + depth,
  ];
}

function animateLine(
  points: Vec3[],
  lineIndex: number,
  time: number,
  settings: ReturnType<typeof profileSettings>,
  intensity: number,
  activity: number,
  mobile: boolean,
) {
  return points.map((point, pointIndex) =>
    animatePoint(
      point,
      lineIndex * 1000 + pointIndex,
      time,
      settings,
      intensity,
      activity,
      mobile,
    ),
  );
}

function animateQuad(
  points: Vec3[],
  quadIndex: number,
  time: number,
  settings: ReturnType<typeof profileSettings>,
  intensity: number,
  activity: number,
  mobile: boolean,
) {
  return points.map((point, pointIndex) =>
    animatePoint(
      point,
      quadIndex * 1000 + pointIndex,
      time,
      settings,
      intensity,
      activity,
      mobile,
    ),
  ) as [Vec3, Vec3, Vec3, Vec3];
}

/**
 * Main V4.2 motion pipeline.
 */
export function applyMotionV42(
  scene: Scene3D,
  clock: MotionV42Clock,
  options: MotionV42Options = {},
): Scene3D {
  const profile = options.profile ?? "default";
  const intensity = clamp(options.intensity ?? 1, 0, 2);
  const mobile = options.mobile ?? false;

  const settings = profileSettings(profile);

  /*
   * Activity berasal dari caller.
   * Clamp mencegah satu spike data menghasilkan deformasi besar.
   */
  const activity = clamp(clock.activity ?? 1, 0.25, 1.5);

  /*
   * Jangan membuat object geometry baru kalau intensity = 0.
   */
  if (intensity <= 0) {
    return scene;
  }

  /*
   * Mobile throttling:
   *
   * Geometry tetap dianimasikan, tetapi frekuensi temporalnya sedikit
   * diperlambat sehingga CPU tidak bekerja terlalu agresif.
   */
  const effectiveTime = mobile
    ? clock.time * 0.82
    : clock.time;

  /*
   * Points
   */
  const points = scene.points?.map((point, index) => ({
    ...point,
    p: animatePoint(
      point.p,
      index,
      effectiveTime,
      settings,
      intensity,
      activity,
      mobile,
    ),
  }));

  /*
   * Lines
   */
  const lines = scene.lines?.map((line, lineIndex) => ({
    ...line,
    pts: animateLine(
      line.pts,
      lineIndex,
      effectiveTime,
      settings,
      intensity,
      activity,
      mobile,
    ),
  }));

  /*
   * Quads / surface.
   */
  const quads = scene.quads?.map((quad, quadIndex) => ({
    ...quad,
    pts: animateQuad(
      quad.pts,
      quadIndex,
      effectiveTime,
      settings,
      intensity,
      activity,
      mobile,
    ),
  }));

  return {
    ...scene,
    points,
    lines,
    quads,
  };
}
