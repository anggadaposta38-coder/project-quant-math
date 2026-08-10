import type {
  Scene3D,
  Vec3,
} from "@/lib/viz/engine3d";

export type MotionV4Profile =
  | "default"
  | "monte-carlo"
  | "surface"
  | "hidden-state"
  | "frontier"
  | "eigen-space";

export interface MotionV4Context {
  time: number;
  dt: number;
  activity?: number;
}

export interface MotionV4Options {
  profile?: MotionV4Profile;
  intensity?: number;
  mobile?: boolean;
}

const TAU = Math.PI * 2;

function clamp(
  v: number,
  min: number,
  max: number,
) {
  return Math.max(
    min,
    Math.min(max, v),
  );
}

function phase(i: number) {
  return (
    (i * 0.61803398875) % 1
  );
}

function hash(i: number) {
  const x = Math.sin(
    i * 12.9898 + 78.233,
  ) * 43758.5453;

  return x - Math.floor(x);
}

function smoothstep(t: number) {
  return t * t * (3 - 2 * t);
}

function copyVec(p: Vec3): Vec3 {
  return [
    p[0],
    p[1],
    p[2],
  ];
}

function add(
  a: Vec3,
  b: Vec3,
): Vec3 {
  return [
    a[0] + b[0],
    a[1] + b[1],
    a[2] + b[2],
  ];
}

/* -------------------------------------------------------------------------- */
/* Monte Carlo                                                                */
/* -------------------------------------------------------------------------- */

function animateMonteCarlo(
  scene: Scene3D,
  time: number,
  intensity: number,
) {
  if (!scene.lines?.length) {
    return;
  }

  scene.lines = scene.lines.map(
    (line, lineIndex) => {
      if (line.pts.length < 2) {
        return line;
      }

      const p =
        phase(lineIndex);

      /*
       * Gerakan utama berjalan dari
       * t0 menuju horizon.
       */
      const travel =
        (time * 0.18 + p) % 1;

      const pts =
        line.pts.map(
          (point, i) => {
            const u =
              i /
              Math.max(
                1,
                line.pts.length - 1,
              );

            /*
             * Gelombang mengikuti posisi
             * sepanjang path.
             */
            const wave =
              Math.sin(
                u * 8 +
                  time * 2.2 +
                  p * TAU,
              );

            /*
             * Energy meningkat sedikit
             * saat wave mendekati front.
             */
            const front =
              Math.exp(
                -Math.pow(
                  (u -
                    travel) *
                    9,
                  2,
                ),
              );

            return [
              point[0],
              point[1] +
                wave *
                  0.012 *
                  intensity +
                front *
                  0.025 *
                  intensity,
              point[2],
            ] as Vec3;
          },
        );

      return {
        ...line,
        pts,
        flow: true,
        flowSpeed:
          0.8 +
          p * 0.7,
      };
    },
  );
}

/* -------------------------------------------------------------------------- */
/* Volatility surface                                                         */
/* -------------------------------------------------------------------------- */

function animateSurface(
  scene: Scene3D,
  time: number,
  intensity: number,
) {
  if (!scene.quads?.length) {
    return;
  }

  scene.quads =
    scene.quads.map(
      (quad, index) => {
        const p =
          phase(index);

        /*
         * Posisi pusat quad.
         */
        const cx =
          (quad.pts[0]![0] +
            quad.pts[1]![0] +
            quad.pts[2]![0] +
            quad.pts[3]![0]) /
          4;

        const cz =
          (quad.pts[0]![2] +
            quad.pts[1]![2] +
            quad.pts[2]![2] +
            quad.pts[3]![2]) /
          4;

        /*
         * Wave mengikuti koordinat
         * permukaan, bukan random.
         */
        const wave =
          Math.sin(
            cx * 3.5 +
              cz * 2.7 +
              time * 1.4,
          ) *
            0.55 +
          Math.sin(
            cx * 7 -
              cz * 2 +
              time * 0.9 +
              p * TAU,
          ) *
            0.25;

        const amp =
          0.035 *
          intensity;

        const pts =
          quad.pts.map(
            (point) =>
              [
                point[0],
                point[1] +
                  wave * amp,
                point[2],
              ] as Vec3,
          ) as [
            Vec3,
            Vec3,
            Vec3,
            Vec3,
          ];

        return {
          ...quad,
          pts,
          ripple:
            Math.max(
              quad.ripple ?? 0,
              0.012 *
                intensity,
            ),
          pulse: p,
        };
      },
    );
}

/* -------------------------------------------------------------------------- */
/* Hidden state / Eigen cloud                                                 */
/* -------------------------------------------------------------------------- */

function animateCloud(
  scene: Scene3D,
  time: number,
  intensity: number,
  strength: number,
) {
  if (!scene.points?.length) {
    return;
  }

  scene.points =
    scene.points.map(
      (point, index) => {
        const p =
          phase(index);

        /*
         * Data position tetap menjadi
         * pusat gerakan.
         */
        const base =
          point.p;

        /*
         * Local breathing.
         */
        const breathe =
          Math.sin(
            time * 1.2 +
              p * TAU,
          );

        /*
         * Cluster drift.
         */
        const driftX =
          Math.sin(
            time * 0.28 +
              base[1] * 1.7,
          );

        const driftY =
          Math.cos(
            time * 0.34 +
              base[0] * 1.4,
          );

        const driftZ =
          Math.sin(
            time * 0.22 +
              base[2] * 1.8,
          );

        /*
         * Makin jauh dari pusat,
         * gerak makin terasa sebagai
         * orbit kecil.
         */
        const radius =
          Math.min(
            1.5,
            Math.hypot(
              base[0],
              base[1],
              base[2],
            ),
          );

        const local =
          strength *
          intensity *
          (0.008 +
            radius * 0.006);

        return {
          ...point,

          p: [
            base[0] +
              driftX * local,
            base[1] +
              driftY * local +
              breathe *
                local *
                0.55,
            base[2] +
              driftZ * local,
          ],

          phase: p,

          glow:
            point.glow ??
            true,

          jitter:
            Math.max(
              point.jitter ?? 0,
              0.004 *
                intensity,
            ),
        };
      },
    );
}

/* -------------------------------------------------------------------------- */
/* Frontier                                                                   */
/* -------------------------------------------------------------------------- */

function animateFrontier(
  scene: Scene3D,
  time: number,
  intensity: number,
) {
  if (!scene.lines?.length) {
    return;
  }

  scene.lines =
    scene.lines.map(
      (line, index) => {
        const p =
          phase(index);

        if (
          line.pts.length < 2
        ) {
          return line;
        }

        const pts =
          line.pts.map(
            (point, i) => {
              const u =
                i /
                Math.max(
                  1,
                  line.pts.length - 1,
                );

              /*
               * Sweep bergerak dari kiri
               * ke kanan.
               */
              const sweep =
                (time * 0.24 +
                  p) %
                1;

              const distance =
                Math.abs(
                  u - sweep,
                );

              const highlight =
                Math.exp(
                  -distance *
                    distance *
                    180,
                );

              return [
                point[0],
                point[1] +
                  highlight *
                    0.035 *
                    intensity,
                point[2],
              ] as Vec3;
            },
          );

        return {
          ...line,
          pts,
          flow: true,
          flowSpeed:
            0.7 +
            p * 0.8,
        };
      },
    );
}

/* -------------------------------------------------------------------------- */
/* Generic 3D motion                                                          */
/* -------------------------------------------------------------------------- */

function animateGeneric(
  scene: Scene3D,
  time: number,
  intensity: number,
  mobile: boolean,
) {
  /*
   * Lines
   */
  if (scene.lines) {
    scene.lines =
      scene.lines.map(
        (line, index) => {
          const p =
            phase(index);

          if (
            line.pts.length <
            2
          ) {
            return line;
          }

          const pts =
            line.pts.map(
              (point, i) => {
                const u =
                  i /
                  Math.max(
                    1,
                    line.pts.length -
                      1,
                  );

                const wave =
                  Math.sin(
                    u * 5 +
                      time * 1.1 +
                      p * TAU,
                  );

                return [
                  point[0],
                  point[1] +
                    wave *
                      0.008 *
                      intensity,
                  point[2],
                ] as Vec3;
              },
            );

          return {
            ...line,
            pts,
            flow:
              line.flow ??
              true,
            flowSpeed:
              line.flowSpeed ??
              0.7,
          };
        },
      );
  }

  /*
   * Points
   */
  if (scene.points) {
    const strength =
      mobile
        ? 0.45
        : 1;

    scene.points =
      scene.points.map(
        (point, index) => {
          const p =
            phase(index);

          const wave =
            Math.sin(
              time * 1.15 +
                p * TAU,
            );

          const amount =
            0.004 *
            intensity *
            strength;

          return {
            ...point,
            p: [
              point.p[0],
              point.p[1] +
                wave * amount,
              point.p[2],
            ],
            phase: p,
            glow:
              point.glow ??
              true,
          };
        },
      );
  }

  /*
   * Quads
   */
  if (scene.quads) {
    scene.quads =
      scene.quads.map(
        (quad, index) => ({
          ...quad,
          ripple:
            quad.ripple ??
            0.008 *
              intensity,
          pulse:
            quad.pulse ??
            phase(index),
        }),
      );
  }
}

/* -------------------------------------------------------------------------- */
/* Profile detection                                                          */
/* -------------------------------------------------------------------------- */

function detectProfile(
  scene: Scene3D,
): MotionV4Profile {
  const lines =
    scene.lines?.length ?? 0;

  const quads =
    scene.quads?.length ?? 0;

  const points =
    scene.points?.length ?? 0;

  /*
   * Surface:
   * banyak quad dan relatif
   * sedikit point.
   */
  if (
    quads > 20 &&
    quads >= points * 0.4
  ) {
    return "surface";
  }

  /*
   * Monte Carlo:
   * banyak line dengan banyak
   * titik per line.
   */
  if (
    lines > 30 &&
    (scene.lines?.[0]
      ?.pts.length ?? 0) >
      10
  ) {
    return "monte-carlo";
  }

  /*
   * Cloud:
   * banyak point.
   */
  if (
    points > 100
  ) {
    return "hidden-state";
  }

  return "default";
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

export function applyMotionV4(
  scene: Scene3D,
  context: MotionV4Context,
  options: MotionV4Options = {},
): Scene3D {
  const time =
    context.time ?? 0;

  const activity =
    clamp(
      context.activity ?? 1,
      0,
      2,
    );

  const intensity =
    clamp(
      options.intensity ??
        1,
      0,
      2,
    ) *
    activity;

  const mobile =
    options.mobile ?? false;

  /*
   * Jangan mutate scene asli.
   */
  const output: Scene3D = {
    ...scene,

    points:
      scene.points?.map(
        (p) => ({
          ...p,
          p: copyVec(p.p),
        }),
      ),

    lines:
      scene.lines?.map(
        (l) => ({
          ...l,
          pts: l.pts.map(
            copyVec,
          ),
        }),
      ),

    quads:
      scene.quads?.map(
        (q) => ({
          ...q,
          pts: q.pts.map(
            copyVec,
          ) as [
            Vec3,
            Vec3,
            Vec3,
            Vec3,
          ],
        }),
      ),
  };

  const profile =
    options.profile &&
    options.profile !==
      "default"
      ? options.profile
      : detectProfile(
          output,
        );

  switch (profile) {
    case "monte-carlo":
      animateMonteCarlo(
        output,
        time,
        intensity,
      );
      break;

    case "surface":
      animateSurface(
        output,
        time,
        intensity,
      );
      break;

    case "hidden-state":
      animateCloud(
        output,
        time,
        intensity,
        1,
      );
      break;

    case "eigen-space":
      animateCloud(
        output,
        time,
        intensity,
        1.35,
      );
      break;

    case "frontier":
      animateFrontier(
        output,
        time,
        intensity,
      );
      break;

    default:
      animateGeneric(
        output,
        time,
        intensity,
        mobile,
      );
      break;
  }

  /*
   * Global low-cost motion.
   *
   * Tidak membuat kamera bergerak.
   * Hanya memperkuat visualisasi data.
   */
  if (
    output.lines
  ) {
    output.lines =
      output.lines.map(
        (line, index) => {
          const p =
            phase(index);

          return {
            ...line,
            flow:
              line.flow ??
              true,
            flowSpeed:
              line.flowSpeed ??
              0.6 +
                p * 0.5,
          };
        },
      );
  }

  /*
   * Surface selalu diberi pulse
   * sangat halus.
   */
  if (
    output.quads
  ) {
    output.quads =
      output.quads.map(
        (quad, index) => ({
          ...quad,
          pulse:
            quad.pulse ??
            phase(index),
        }),
      );
  }

  /*
   * Mobile: turunkan intensitas
   * micro-motion supaya tidak berat.
   */
  if (mobile) {
    if (
      output.points
    ) {
      output.points =
        output.points.map(
          (point) => ({
            ...point,
            jitter:
              point.jitter
                ? point.jitter *
                  0.55
                : undefined,
          }),
        );
    }
  }

  return output;
}
