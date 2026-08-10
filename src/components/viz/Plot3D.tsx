import {
  memo,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";

import {
  renderScene,
  type Camera,
  type Scene3D,
  type Vec3,
} from "@/lib/viz/engine3d";

import { THEME_CHANGE_EVENT } from "@/lib/theme";

import {
  createMotionFieldParticles,
  stepMotionField,
  type MotionFieldParticle,
} from "@/lib/viz/motion-engine-v3";

interface Plot3DProps {
  scene: Scene3D;
  height?: number;
  autoRotate?: boolean;
}

/* -------------------------------------------------------------------------- */
/* Constants                                                                  */
/* -------------------------------------------------------------------------- */

const MOBILE_BREAKPOINT = 768;

const MOBILE_MAX_RENDER_POINTS = 420;
const DESKTOP_MAX_RENDER_POINTS = 900;

const MOBILE_MAX_MOTION_POINTS = 120;
const DESKTOP_MAX_MOTION_POINTS = 240;

const MOBILE_DPR = 1.25;
const DESKTOP_DPR = 1.75;

const MOBILE_FPS = 30;
const DESKTOP_FPS = 45;

const MORPH_DURATION_MS = 700;

/*
 * Motion amplitude sengaja kecil.
 *
 * V3 menghasilkan posisi particle sendiri. Kita TIDAK langsung memakai
 * posisi tersebut sebagai posisi final karena itu dapat membuat cloud
 * "terbang" dari geometry data asli.
 *
 * Yang digunakan hanya:
 *
 *     originalPosition + motionOffset * amplitude
 *
 * Dengan begitu motion terasa hidup tetapi struktur data tetap terlihat.
 */
const MOBILE_MOTION_AMPLITUDE = 0.085;
const DESKTOP_MOTION_AMPLITUDE = 0.16;

/* -------------------------------------------------------------------------- */
/* Utilities                                                                  */
/* -------------------------------------------------------------------------- */

function cssVar(
  el: HTMLElement,
  name: string,
  fallback: string,
): string {
  const value = getComputedStyle(el).getPropertyValue(name).trim();
  return value || fallback;
}

function clamp(
  value: number,
  min: number,
  max: number,
): number {
  return Math.max(min, Math.min(max, value));
}

function easeInOut(t: number): number {
  return t < 0.5
    ? 2 * t * t
    : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function lerp(
  a: number,
  b: number,
  t: number,
): number {
  return a + (b - a) * t;
}

function lerpVec(
  a: Vec3,
  b: Vec3,
  t: number,
): Vec3 {
  return [
    lerp(a[0], b[0], t),
    lerp(a[1], b[1], t),
    lerp(a[2], b[2], t),
  ];
}

function isMobileViewport(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  return window.innerWidth <= MOBILE_BREAKPOINT;
}

/* -------------------------------------------------------------------------- */
/* Scene detection                                                             */
/* -------------------------------------------------------------------------- */

/**
 * HMM Hidden State Space mempunyai axis:
 *
 *   Return × Volatilitas × Waktu
 *
 * Scene lain tidak menggunakan V3 particle field.
 */
function isHiddenStateScene(scene: Scene3D): boolean {
  const labels = scene.axisLabels;

  if (!labels || labels.length !== 3) {
    return false;
  }

  return (
    labels[0] === "Return" &&
    labels[1] === "Volatilitas" &&
    labels[2] === "Waktu"
  );
}

/* -------------------------------------------------------------------------- */
/* Deterministic point sampling                                                */
/* -------------------------------------------------------------------------- */

/**
 * Mengambil titik secara merata tanpa random.
 *
 * Contoh:
 *
 * 10.000 points -> 420 points
 *
 * Titik pertama dan terakhir selalu dipertahankan.
 */
function sampleIndices(
  length: number,
  maxPoints: number,
): number[] {
  if (length <= maxPoints) {
    return Array.from(
      { length },
      (_, index) => index,
    );
  }

  if (maxPoints <= 1) {
    return [0];
  }

  const result: number[] = [];

  for (let i = 0; i < maxPoints; i += 1) {
    const ratio = i / (maxPoints - 1);
    const index = Math.round(
      ratio * (length - 1),
    );

    result.push(index);
  }

  return result;
}

/* -------------------------------------------------------------------------- */
/* Mobile rendering optimization                                               */
/* -------------------------------------------------------------------------- */

function optimizeSceneForDevice(
  scene: Scene3D,
  mobile: boolean,
): Scene3D {
  const maxPoints = mobile
    ? MOBILE_MAX_RENDER_POINTS
    : DESKTOP_MAX_RENDER_POINTS;

  const sourcePoints = scene.points ?? [];

  if (sourcePoints.length === 0) {
    return scene;
  }

  const indices = sampleIndices(
    sourcePoints.length,
    maxPoints,
  );

  const points = indices.map((index) => {
    const point = sourcePoints[index]!;

    if (!mobile) {
      return point;
    }

    /*
     * Mobile:
     *
     * - glow OFF
     * - point sedikit lebih kecil
     * - alpha sedikit diturunkan
     * - jitter diperkecil
     *
     * Ini menghindari halo antar particle menutup satu sama lain.
     */
    return {
      ...point,
      glow: false,
      size: Math.min(
        point.size ?? 1.7,
        1.35,
      ) * 0.82,
      alpha: clamp(
        (point.alpha ?? 1) * 0.78,
        0.28,
        0.82,
      ),
      jitter: Math.min(
        point.jitter ?? 0,
        0.006,
      ),
    };
  });

  /*
   * Untuk line kita tetap mempertahankan geometry asli.
   *
   * Line HMM jauh lebih sedikit daripada point particle sehingga tidak perlu
   * dipotong agresif.
   */
  return {
    ...scene,
    points,
  };
}

/* -------------------------------------------------------------------------- */
/* Motion source                                                               */
/* -------------------------------------------------------------------------- */

interface MotionModel {
  scene: Scene3D;
  sourceIndices: number[];
  particles: MotionFieldParticle[];
}

function createMotionModel(
  scene: Scene3D,
  mobile: boolean,
): MotionModel | null {
  if (!isHiddenStateScene(scene)) {
    return null;
  }

  const points = scene.points ?? [];

  if (points.length === 0) {
    return null;
  }

  const maxMotionPoints = mobile
    ? MOBILE_MAX_MOTION_POINTS
    : DESKTOP_MAX_MOTION_POINTS;

  const sourceIndices = sampleIndices(
    points.length,
    maxMotionPoints,
  );

  const motionPoints = sourceIndices.map(
    (index) => points[index]!,
  );

  const motionScene: Scene3D = {
    points: motionPoints,
  };

  const particles =
    createMotionFieldParticles(
      motionScene,
    );

  return {
    scene: motionScene,
    sourceIndices,
    particles,
  };
}

/* -------------------------------------------------------------------------- */
/* V2 geometry morph                                                           */
/* -------------------------------------------------------------------------- */

/**
 * V2-style geometry transition.
 *
 * Ini menangani perubahan:
 *
 * DATA A -> DATA B
 *
 * sedangkan V3 hanya menangani motion kontinu setelah geometry stabil.
 */
function morphScene(
  from: Scene3D | null,
  to: Scene3D,
  progress: number,
): Scene3D {
  if (!from) {
    return to;
  }

  const t = easeInOut(
    clamp(progress, 0, 1),
  );

  const points =
    from.points &&
    to.points &&
    from.points.length === to.points.length
      ? to.points.map((point, index) => {
          const previous =
            from.points?.[index];

          if (!previous) {
            return point;
          }

          return {
            ...point,
            p: lerpVec(
              previous.p,
              point.p,
              t,
            ),
          };
        })
      : to.points;

  const lines =
    from.lines &&
    to.lines &&
    from.lines.length === to.lines.length
      ? to.lines.map((line, lineIndex) => {
          const previous =
            from.lines?.[lineIndex];

          if (
            !previous ||
            previous.pts.length !==
              line.pts.length
          ) {
            return line;
          }

          return {
            ...line,
            pts: line.pts.map(
              (point, pointIndex) =>
                lerpVec(
                  previous.pts[
                    pointIndex
                  ]!,
                  point,
                  t,
                ),
            ),
          };
        })
      : to.lines;

  const quads =
    from.quads &&
    to.quads &&
    from.quads.length === to.quads.length
      ? to.quads.map(
          (quad, quadIndex) => {
            const previous =
              from.quads?.[quadIndex];

            if (
              !previous ||
              previous.pts.length !==
                quad.pts.length
            ) {
              return quad;
            }

            return {
              ...quad,
              pts: quad.pts.map(
                (point, pointIndex) =>
                  lerpVec(
                    previous.pts[
                      pointIndex
                    ]!,
                    point,
                    t,
                  ),
              ) as [
                Vec3,
                Vec3,
                Vec3,
                Vec3,
              ],
            };
          },
        )
      : to.quads;

  return {
    ...to,
    points,
    lines,
    quads,
  };
}

/* -------------------------------------------------------------------------- */
/* Apply V3 as a controlled overlay                                             */
/* -------------------------------------------------------------------------- */

/**
 * V3 tidak mengambil alih geometry.
 *
 * Hanya:
 *
 *     data position
 *       +
 *     small deterministic motion offset
 *
 * Dengan demikian cloud tetap merepresentasikan data asli.
 */
function applyControlledMotion(
  scene: Scene3D,
  motionModel: MotionModel,
  amplitude: number,
): Scene3D {
  const sourcePoints = scene.points ?? [];

  if (
    sourcePoints.length === 0 ||
    motionModel.particles.length === 0
  ) {
    return scene;
  }

  const points = sourcePoints.slice();

  for (
    let i = 0;
    i < motionModel.sourceIndices.length;
    i += 1
  ) {
    const sourceIndex =
      motionModel.sourceIndices[i];

    const particle =
      motionModel.particles[i];

    const base =
      sourcePoints[sourceIndex];

    if (!particle || !base) {
      continue;
    }

    const dx =
      particle.position[0] -
      motionModel.scene.points![i]!.p[0];

    const dy =
      particle.position[1] -
      motionModel.scene.points![i]!.p[1];

    const dz =
      particle.position[2] -
      motionModel.scene.points![i]!.p[2];

    /*
     * Extra clamp supaya particle tidak pernah bisa meloncat jauh
     * dari geometry data.
     */
    const maxOffset = amplitude;

    const offsetLength = Math.sqrt(
      dx * dx +
        dy * dy +
        dz * dz,
    );

    let scale = 1;

    if (
      offsetLength > maxOffset &&
      offsetLength > 0.000001
    ) {
      scale =
        maxOffset / offsetLength;
    }

    points[sourceIndex] = {
      ...base,
      p: [
        base.p[0] + dx * scale,
        base.p[1] + dy * scale,
        base.p[2] + dz * scale,
      ],
    };
  }

  return {
    ...scene,
    points,
  };
}

/* -------------------------------------------------------------------------- */
/* Component                                                                   */
/* -------------------------------------------------------------------------- */

export const Plot3D = memo(
  function Plot3D({
    scene,
    height = 320,
    autoRotate = true,
  }: Plot3DProps) {
    const canvasRef =
      useRef<HTMLCanvasElement | null>(
        null,
      );

    const wrapRef =
      useRef<HTMLDivElement | null>(
        null,
      );

    const camRef = useRef<Camera>({
      yaw: -0.75,
      pitch: 0.42,
      zoom: 1.02,
    });

    const dragRef =
      useRef<{
        x: number;
        y: number;
      } | null>(null);

    const mobileRef =
      useRef(false);

    const sceneRef =
      useRef<Scene3D>(scene);

    const targetSceneRef =
      useRef<Scene3D>(scene);

    const transitionRef =
      useRef<{
        from: Scene3D;
        startedAt: number;
      } | null>(null);

    const motionRef =
      useRef<MotionModel | null>(
        null,
      );

    const motionClockRef =
      useRef(0);

    const [spin, setSpin] =
      useState(autoRotate);

    const spinRef =
      useRef(spin);

    spinRef.current = spin;

    /* ---------------------------------------------------------------------- */
    /* Scene updates                                                           */
    /* ---------------------------------------------------------------------- */

    useEffect(() => {
      const mobile =
        mobileRef.current;

      const optimized =
        optimizeSceneForDevice(
          scene,
          mobile,
        );

      targetSceneRef.current =
        optimized;

      transitionRef.current = {
        from: sceneRef.current,
        startedAt:
          performance.now(),
      };

      /*
       * Reset V3 terhadap geometry baru.
       *
       * Ini penting supaya particle tidak membawa posisi lama ke dataset baru.
       */
      motionRef.current =
        createMotionModel(
          optimized,
          mobile,
        );

      motionClockRef.current = 0;
    }, [scene]);

    /* ---------------------------------------------------------------------- */
    /* Main renderer                                                           */
    /* ---------------------------------------------------------------------- */

    useEffect(() => {
      const canvas =
        canvasRef.current;

      const wrap =
        wrapRef.current;

      if (!canvas || !wrap) {
        return;
      }

      const ctx =
        canvas.getContext("2d");

      if (!ctx) {
        return;
      }

      let raf = 0;

      let last = performance.now();

      let lastRenderedAt =
        performance.now();

      let clock = 0;

      let paused =
        document.hidden;

      mobileRef.current =
        isMobileViewport();

      const themeRef = {
        current: {
          grid: cssVar(
            wrap,
            "--chart-grid",
            "#2a3346",
          ),
          text: cssVar(
            wrap,
            "--chart-axis",
            "#8b98b0",
          ),
        },
      };

      const refreshTheme = () => {
        themeRef.current = {
          grid: cssVar(
            wrap,
            "--chart-grid",
            "#2a3346",
          ),
          text: cssVar(
            wrap,
            "--chart-axis",
            "#8b98b0",
          ),
        };
      };

      const refreshDeviceMode = () => {
        const nextMobile =
          isMobileViewport();

        if (
          nextMobile ===
          mobileRef.current
        ) {
          return;
        }

        mobileRef.current =
          nextMobile;

        const optimized =
          optimizeSceneForDevice(
            targetSceneRef.current,
            nextMobile,
          );

        sceneRef.current =
          optimizeSceneForDevice(
            sceneRef.current,
            nextMobile,
          );

        targetSceneRef.current =
          optimized;

        motionRef.current =
          createMotionModel(
            optimized,
            nextMobile,
          );
      };

      const onVisibilityChange =
        () => {
          paused =
            document.hidden;

          last =
            performance.now();

          lastRenderedAt =
            last;

          if (
            !paused &&
            raf === 0
          ) {
            raf =
              requestAnimationFrame(
                draw,
              );
          }
        };

      const draw = (
        now: number,
      ) => {
        raf = 0;

        if (paused) {
          return;
        }

        const mobile =
          mobileRef.current;

        const targetFps = mobile
          ? MOBILE_FPS
          : DESKTOP_FPS;

        const frameInterval =
          1000 / targetFps;

        /*
         * FPS cap.
         *
         * Canvas tetap requestAnimationFrame,
         * tetapi renderScene tidak dipanggil
         * pada setiap browser frame.
         */
        if (
          now - lastRenderedAt <
          frameInterval
        ) {
          raf =
            requestAnimationFrame(
              draw,
            );
          return;
        }

        const dt = clamp(
          (now - last) / 1000,
          0,
          0.05,
        );

        last = now;
        lastRenderedAt = now;

        clock += dt;

        /* -------------------------------------------------------------- */
        /* Camera                                                           */
        /* -------------------------------------------------------------- */

        if (
          spinRef.current &&
          !dragRef.current
        ) {
          camRef.current.yaw +=
            dt *
            (mobile
              ? 0.095
              : 0.16);
        }

        /* -------------------------------------------------------------- */
        /* V2 geometry transition                                           */
        /* -------------------------------------------------------------- */

        let activeScene =
          sceneRef.current;

        const transition =
          transitionRef.current;

        if (transition) {
          const progress =
            Math.min(
              1,
              (now -
                transition.startedAt) /
                MORPH_DURATION_MS,
            );

          activeScene =
            morphScene(
              transition.from,
              targetSceneRef.current,
              progress,
            );

          sceneRef.current =
            activeScene;

          if (progress >= 1) {
            sceneRef.current =
              targetSceneRef.current;

            transitionRef.current =
              null;

            motionRef.current =
              createMotionModel(
                targetSceneRef.current,
                mobile,
              );
          }
        }

        /* -------------------------------------------------------------- */
        /* V3 motion                                                        */
        /* -------------------------------------------------------------- */

        const motion =
          motionRef.current;

        /*
         * Jangan menjalankan V3 ketika geometry masih morph.
         *
         * Ini mencegah dua sistem sekaligus mengubah posisi particle.
         */
        if (
          motion &&
          !transitionRef.current &&
          isHiddenStateScene(
            activeScene,
          )
        ) {
          /*
           * Motion simulation dibuat lebih lambat daripada render.
           *
           * Mobile:
           *   ~20 Hz
           *
           * Desktop:
           *   mengikuti render cap.
           */
          const motionInterval =
            mobile
              ? 1 / 20
              : 1 / 35;

          motionClockRef.current +=
            dt;

          if (
            motionClockRef.current >=
            motionInterval
          ) {
            const motionDt =
              motionClockRef.current;

            motionClockRef.current =
              0;

            motion.particles =
              stepMotionField(
                motion.particles,
                motion.scene,
                Math.min(
                  motionDt,
                  0.05,
                ),
                {
                  fieldStrength:
                    mobile
                      ? 0.34
                      : 0.58,

                  cohesion:
                    mobile
                      ? 0.10
                      : 0.18,

                  separation:
                    mobile
                      ? 0.08
                      : 0.12,

                  inertia:
                    mobile
                      ? 0.86
                      : 0.9,

                  damping:
                    mobile
                      ? 0.28
                      : 0.22,

                  maxSpeed:
                    mobile
                      ? 0.55
                      : 0.85,

                  stateBias:
                    mobile
                      ? 0.07
                      : 0.11,
                },
              );
          }

          activeScene =
            applyControlledMotion(
              activeScene,
              motion,
              mobile
                ? MOBILE_MOTION_AMPLITUDE
                : DESKTOP_MOTION_AMPLITUDE,
            );
        }

        /* -------------------------------------------------------------- */
        /* Canvas                                                           */
        /* -------------------------------------------------------------- */

        const dpr = Math.min(
          window.devicePixelRatio ||
            1,
          mobile
            ? MOBILE_DPR
            : DESKTOP_DPR,
        );

        const w =
          wrap.clientWidth;

        const h = height;

        if (
          canvas.width !==
            Math.floor(w * dpr) ||
          canvas.height !==
            Math.floor(h * dpr)
        ) {
          canvas.width =
            Math.floor(w * dpr);

          canvas.height =
            Math.floor(h * dpr);

          canvas.style.width =
            `${w}px`;

          canvas.style.height =
            `${h}px`;
        }

        ctx.setTransform(
          dpr,
          0,
          0,
          dpr,
          0,
          0,
        );

        renderScene(
          ctx,
          activeScene,
          camRef.current,
          w,
          h,
          themeRef.current,
          clock,
        );

        raf =
          requestAnimationFrame(
            draw,
          );
      };

      window.addEventListener(
        THEME_CHANGE_EVENT,
        refreshTheme,
      );

      window.addEventListener(
        "resize",
        refreshDeviceMode,
      );

      document.addEventListener(
        "visibilitychange",
        onVisibilityChange,
      );

      raf =
        requestAnimationFrame(
          draw,
        );

      return () => {
        cancelAnimationFrame(raf);

        window.removeEventListener(
          THEME_CHANGE_EVENT,
          refreshTheme,
        );

        window.removeEventListener(
          "resize",
          refreshDeviceMode,
        );

        document.removeEventListener(
          "visibilitychange",
          onVisibilityChange,
        );
      };
    }, [height]);

    /* ---------------------------------------------------------------------- */
    /* Pointer controls                                                       */
    /* ---------------------------------------------------------------------- */

    const onPointerDown = (
      event: PointerEvent<HTMLCanvasElement>,
    ) => {
      dragRef.current = {
        x: event.clientX,
        y: event.clientY,
      };

      (
        event.currentTarget as HTMLCanvasElement
      ).setPointerCapture?.(
        event.pointerId,
      );
    };

    const onPointerMove = (
      event: PointerEvent<HTMLCanvasElement>,
    ) => {
      const drag =
        dragRef.current;

      if (!drag) {
        return;
      }

      camRef.current.yaw +=
        (event.clientX -
          drag.x) *
        0.008;

      camRef.current.pitch =
        Math.max(
          -1.4,
          Math.min(
            1.4,
            camRef.current.pitch +
              (event.clientY -
                drag.y) *
                0.006,
          ),
        );

      dragRef.current = {
        x: event.clientX,
        y: event.clientY,
      };
    };

    const endDrag = () => {
      dragRef.current = null;
    };

    /* ---------------------------------------------------------------------- */
    /* Keyboard controls                                                      */
    /* ---------------------------------------------------------------------- */

    const onKeyDown = (
      event: KeyboardEvent<HTMLCanvasElement>,
    ) => {
      const step =
        event.shiftKey
          ? 0.18
          : 0.08;

      switch (event.key) {
        case "ArrowLeft":
          event.preventDefault();
          camRef.current.yaw -=
            step;
          break;

        case "ArrowRight":
          event.preventDefault();
          camRef.current.yaw +=
            step;
          break;

        case "ArrowUp":
          event.preventDefault();

          camRef.current.pitch =
            Math.min(
              1.4,
              camRef.current.pitch +
                step,
            );

          break;

        case "ArrowDown":
          event.preventDefault();

          camRef.current.pitch =
            Math.max(
              -1.4,
              camRef.current.pitch -
                step,
            );

          break;

        case "+":
        case "=":
          event.preventDefault();

          camRef.current.zoom =
            Math.min(
              2.2,
              camRef.current.zoom *
                1.06,
            );

          break;

        case "-":
        case "_":
          event.preventDefault();

          camRef.current.zoom =
            Math.max(
              0.35,
              camRef.current.zoom *
                0.94,
            );

          break;

        case " ":
          event.preventDefault();

          setSpin(
            (value) => !value,
          );

          break;
      }
    };

    /* ---------------------------------------------------------------------- */
    /* Render                                                                  */
    /* ---------------------------------------------------------------------- */

    return (
      <div
        ref={wrapRef}
        className="relative w-full select-none"
      >
        <canvas
          ref={canvasRef}
          role="img"
          aria-label="Visualisasi 3D data kuantitatif. Gunakan geser untuk rotasi, scroll untuk zoom, dan tombol pause untuk menghentikan rotasi."
          tabIndex={0}
          className="w-full cursor-grab touch-none rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-primary active:cursor-grabbing"
          style={{
            height,
            touchAction: "none",
          }}
          onPointerDown={
            onPointerDown
          }
          onPointerMove={
            onPointerMove
          }
          onPointerUp={endDrag}
          onPointerCancel={
            endDrag
          }
          onPointerLeave={
            endDrag
          }
          onKeyDown={onKeyDown}
          onWheel={(event) => {
            event.preventDefault();

            camRef.current.zoom =
              Math.max(
                0.35,
                Math.min(
                  2.2,
                  camRef.current.zoom *
                    (event.deltaY > 0
                      ? 0.94
                      : 1.06),
                ),
              );
          }}
        />

        <button
          type="button"
          aria-label={
            spin
              ? "Jeda rotasi visualisasi 3D"
              : "Mulai rotasi visualisasi 3D"
          }
          aria-pressed={spin}
          onClick={() =>
            setSpin(
              (value) => !value,
            )
          }
          className="absolute right-2 top-2 rounded border border-border/60 bg-card/70 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground backdrop-blur transition-colors hover:text-foreground"
        >
          {spin
            ? "pause"
            : "rotate"}
        </button>
      </div>
    );
  },
);
