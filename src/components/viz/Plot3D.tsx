import { memo, useEffect, useRef, useState } from "react";
import type { PointerEvent, KeyboardEvent } from "react";

import {
  renderScene,
  type Camera,
  type Scene3D,
  type Vec3,
} from "@/lib/viz/engine3d";

import {
  applyMotionV42,
  type MotionV42Profile,
} from "@/lib/viz/motion-engine-v4.2";

import { THEME_CHANGE_EVENT } from "@/lib/theme";

interface Plot3DProps {
  scene: Scene3D;
  height?: number;
  autoRotate?: boolean;
  motionProfile?: MotionV42Profile;
  motionIntensity?: number;
}

function cssVar(
  el: HTMLElement,
  name: string,
  fallback: string,
) {
  const value = getComputedStyle(el)
    .getPropertyValue(name)
    .trim();

  return value || fallback;
}

const easeInOut = (t: number) =>
  t < 0.5
    ? 2 * t * t
    : 1 - Math.pow(-2 * t + 2, 2) / 2;

function lerp(a: number, b: number, t: number) {
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

/**
 * Morph quantitative geometry between snapshots.
 *
 * Data transition acontece antes do motion engine.
 */
function morphScene(
  from: Scene3D | null,
  to: Scene3D,
  progress: number,
): Scene3D {
  if (!from) return to;

  const t = easeInOut(progress);

  const points =
    from.points && to.points
      ? to.points.map((point, index) => {
          const previous = from.points?.[index];

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
    from.lines && to.lines
      ? to.lines.map((line, lineIndex) => {
          const previous =
            from.lines?.[lineIndex];

          if (
            !previous ||
            previous.pts.length !== line.pts.length
          ) {
            return line;
          }

          return {
            ...line,
            pts: line.pts.map(
              (point, pointIndex) =>
                lerpVec(
                  previous.pts[pointIndex]!,
                  point,
                  t,
                ),
            ),
          };
        })
      : to.lines;

  const quads =
    from.quads && to.quads
      ? to.quads.map((quad, quadIndex) => {
          const previous =
            from.quads?.[quadIndex];

          if (
            !previous ||
            previous.pts.length !== quad.pts.length
          ) {
            return quad;
          }

          return {
            ...quad,
            pts: quad.pts.map(
              (point, pointIndex) =>
                lerpVec(
                  previous.pts[pointIndex]!,
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
        })
      : to.quads;

  return {
    ...to,
    points,
    lines,
    quads,
  };
}

export const Plot3D = memo(
  function Plot3D({
    scene,
    height = 320,
    autoRotate = false,
    motionProfile = "default",
    motionIntensity = 1,
  }: Plot3DProps) {
    const canvasRef =
      useRef<HTMLCanvasElement | null>(null);

    const wrapRef =
      useRef<HTMLDivElement | null>(null);

    /*
     * Kamera sengaja relatif stabil.
     *
     * V4.2:
     * geometry yang hidup,
     * bukan kamera yang berputar terus.
     */
    const camRef = useRef<Camera>({
      yaw: -0.75,
      pitch: 0.42,
      zoom: 1.02,
    });

    const dragRef = useRef<{
      x: number;
      y: number;
      lastTime: number;
    } | null>(null);

    const velocityRef = useRef({
      yaw: 0,
      pitch: 0,
    });

    const sceneRef =
      useRef<Scene3D>(scene);

    const targetSceneRef =
      useRef<Scene3D>(scene);

    const transitionRef =
      useRef<{
        from: Scene3D;
        startedAt: number;
      } | null>(null);

    const [spin, setSpin] =
      useState(autoRotate);

    const spinRef = useRef(spin);

    spinRef.current = spin;

    const profileRef =
      useRef<MotionV42Profile>(
        motionProfile,
      );

    profileRef.current =
      motionProfile;

    const intensityRef =
      useRef(motionIntensity);

    intensityRef.current =
      motionIntensity;

    /*
     * Data activity.
     *
     * Ini bukan random.
     * Activity dipakai untuk mengontrol "energi" motion.
     *
     * Nilai dihitung dari perubahan geometry antar snapshot.
     */
    const activityRef =
      useRef(1);

    useEffect(() => {
      const previous =
        targetSceneRef.current;

      targetSceneRef.current = scene;

      /*
       * Hitung activity sederhana berdasarkan perubahan
       * posisi geometry.
       */
      let total = 0;
      let count = 0;

      if (
        previous.points &&
        scene.points
      ) {
        const length = Math.min(
          previous.points.length,
          scene.points.length,
        );

        for (
          let i = 0;
          i < length;
          i++
        ) {
          const a =
            previous.points[i]!.p;

          const b =
            scene.points[i]!.p;

          total +=
            Math.abs(b[0] - a[0]) +
            Math.abs(b[1] - a[1]) +
            Math.abs(b[2] - a[2]);

          count++;
        }
      }

      if (count > 0) {
        const average =
          total / count;

        /*
         * Sensitivity sengaja rendah.
         * Perubahan kecil tetap menghasilkan motion,
         * tetapi tidak membuat grafik meloncat.
         */
        activityRef.current =
          Math.max(
            0.35,
            Math.min(
              1.35,
              0.55 +
                average * 5,
            ),
          );
      }

      transitionRef.current = {
        from:
          sceneRef.current,
        startedAt:
          performance.now(),
      };
    }, [scene]);

    useEffect(() => {
      const canvas =
        canvasRef.current;

      const wrap =
        wrapRef.current;

      if (!canvas || !wrap) {
        return;
      }

      const ctx =
        canvas.getContext("2d", {
          alpha: true,
          desynchronized: true,
        });

      if (!ctx) {
        return;
      }

      let raf = 0;

      let last =
        performance.now();

      let clock = 0;

      let paused =
        document.hidden;

      /*
       * Mobile detection.
       */
      const mobile =
        window.matchMedia(
          "(max-width: 640px)",
        ).matches ||
        /Android|iPhone|iPad|iPod/i.test(
          navigator.userAgent,
        );

      /*
       * Mobile frame pacing.
       *
       * Kita tidak memaksa geometry update
       * pada setiap refresh-rate tinggi.
       */
      const targetFrame =
        mobile
          ? 1000 / 45
          : 1000 / 60;

      let frameAccumulator = 0;

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

      const onVisibilityChange =
        () => {
          paused =
            document.hidden;

          last =
            performance.now();

          frameAccumulator = 0;

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

        const elapsed =
          Math.min(
            Math.max(
              (now - last) /
                1000,
              0,
            ),
            0.05,
          );

        last = now;

        frameAccumulator +=
          elapsed * 1000;

        /*
         * Camera inertia.
         */
        if (!dragRef.current) {
          const velocity =
            velocityRef.current;

          camRef.current.yaw +=
            velocity.yaw *
            elapsed;

          camRef.current.pitch +=
            velocity.pitch *
            elapsed;

          velocity.yaw *=
            Math.exp(
              -elapsed * 7.5,
            );

          velocity.pitch *=
            Math.exp(
              -elapsed * 8.5,
            );

          if (
            Math.abs(
              velocity.yaw,
            ) < 0.00002
          ) {
            velocity.yaw = 0;
          }

          if (
            Math.abs(
              velocity.pitch,
            ) < 0.00002
          ) {
            velocity.pitch = 0;
          }
        }

        /*
         * Optional extremely slow camera drift.
         *
         * Default false.
         *
         * Kalau user memilih autoRotate,
         * tetap sangat kecil.
         */
        if (
          spinRef.current &&
          !dragRef.current
        ) {
          camRef.current.yaw +=
            elapsed * 0.012;

          camRef.current.pitch +=
            (
              Math.sin(
                clock * 0.18,
              ) *
                0.004 -
              (
                camRef.current
                  .pitch -
                0.42
              )
            ) *
            elapsed *
            0.5;
        }

        camRef.current.pitch =
          Math.max(
            -1.25,
            Math.min(
              1.25,
              camRef.current.pitch,
            ),
          );

        /*
         * Clock berjalan terus,
         * tetapi motion tidak dibuat terlalu cepat.
         */
        clock +=
          mobile
            ? elapsed * 0.92
            : elapsed;

        /*
         * Data transition.
         */
        const transition =
          transitionRef.current;

        if (transition) {
          const progress =
            Math.min(
              1,
              (
                now -
                transition.startedAt
              ) / 900,
            );

          sceneRef.current =
            morphScene(
              transition.from,
              targetSceneRef.current,
              progress,
            );

          if (
            progress >= 1
          ) {
            sceneRef.current =
              targetSceneRef.current;

            transitionRef.current =
              null;
          }
        }

        const width =
          wrap.clientWidth;

        if (width <= 0) {
          raf =
            requestAnimationFrame(
              draw,
            );

          return;
        }

        /*
         * Mobile rendering budget.
         */
        const rawDpr =
          window.devicePixelRatio ||
          1;

        const dpr =
          mobile
            ? Math.min(
                rawDpr,
                1.25,
              )
            : Math.min(
                rawDpr,
                1.65,
              );

        const w = width;
        const h = height;

        const pixelWidth =
          Math.max(
            1,
            Math.floor(
              w * dpr,
            ),
          );

        const pixelHeight =
          Math.max(
            1,
            Math.floor(
              h * dpr,
            ),
          );

        if (
          canvas.width !==
            pixelWidth ||
          canvas.height !==
            pixelHeight
        ) {
          canvas.width =
            pixelWidth;

          canvas.height =
            pixelHeight;

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

        /*
         * Mobile:
         *
         * render setiap frame agar gesture tetap
         * responsif, tetapi geometry motion sedikit
         * diperlambat.
         *
         * Kita tetap memakai requestAnimationFrame,
         * karena canvas rendering harus sinkron dengan
         * browser.
         */
        const animatedScene =
          applyMotionV42(
            sceneRef.current,
            {
              time: clock,
              dt: elapsed,
              activity:
                activityRef.current,
            },
            {
              profile:
                profileRef.current,
              intensity:
                intensityRef.current,
              mobile,
            },
          );

        /*
         * Render final.
         */
        renderScene(
          ctx,
          animatedScene,
          camRef.current,
          w,
          h,
          themeRef.current,
          clock,
        );

        /*
         * Jangan membiarkan frameAccumulator
         * terus membesar pada device 120/144Hz.
         */
        if (
          frameAccumulator >
          targetFrame * 2
        ) {
          frameAccumulator =
            targetFrame;
        }

        raf =
          requestAnimationFrame(
            draw,
          );
      };

      window.addEventListener(
        THEME_CHANGE_EVENT,
        refreshTheme,
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
        cancelAnimationFrame(
          raf,
        );

        window.removeEventListener(
          THEME_CHANGE_EVENT,
          refreshTheme,
        );

        document.removeEventListener(
          "visibilitychange",
          onVisibilityChange,
        );
      };
    }, [height]);

    /*
     * Pointer interaction.
     */
    const onPointerDown = (
      event: PointerEvent<HTMLCanvasElement>,
    ) => {
      dragRef.current = {
        x: event.clientX,
        y: event.clientY,
        lastTime:
          performance.now(),
      };

      velocityRef.current.yaw =
        0;

      velocityRef.current.pitch =
        0;

      event.currentTarget.setPointerCapture?.(
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

      const now =
        performance.now();

      const dt =
        Math.max(
          0.008,
          Math.min(
            (
              now -
              drag.lastTime
            ) / 1000,
            0.05,
          ),
        );

      const dx =
        event.clientX -
        drag.x;

      const dy =
        event.clientY -
        drag.y;

      /*
       * Drag camera tetap cepat/responsif.
       */
      const yawDelta =
        dx * 0.007;

      const pitchDelta =
        dy * 0.005;

      camRef.current.yaw +=
        yawDelta;

      camRef.current.pitch =
        Math.max(
          -1.25,
          Math.min(
            1.25,
            camRef.current
              .pitch +
              pitchDelta,
          ),
        );

      velocityRef.current.yaw =
        Math.max(
          -2.4,
          Math.min(
            2.4,
            yawDelta / dt,
          ),
        );

      velocityRef.current.pitch =
        Math.max(
          -2,
          Math.min(
            2,
            pitchDelta / dt,
          ),
        );

      dragRef.current = {
        x: event.clientX,
        y: event.clientY,
        lastTime: now,
      };
    };

    const endDrag = (
      event?: PointerEvent<HTMLCanvasElement>,
    ) => {
      if (event) {
        try {
          event.currentTarget.releasePointerCapture?.(
            event.pointerId,
          );
        } catch {
          // Browser may have already released it.
        }
      }

      dragRef.current =
        null;
    };

    /*
     * Keyboard navigation.
     */
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
              1.25,
              camRef.current
                .pitch +
                step,
            );

          break;

        case "ArrowDown":
          event.preventDefault();

          camRef.current.pitch =
            Math.max(
              -1.25,
              camRef.current
                .pitch -
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
            value => !value,
          );

          break;
      }
    };

    return (
      <div
        ref={wrapRef}
        className="relative w-full select-none"
        style={{
          contain:
            "layout paint",
        }}
      >
        <canvas
          ref={canvasRef}
          role="img"
          aria-label="Visualisasi 3D data kuantitatif. Geometry bergerak mengikuti data. Gunakan sentuhan atau mouse untuk rotasi."
          tabIndex={0}
          className="w-full cursor-grab touch-none rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-primary active:cursor-grabbing"
          style={{
            height,
          }}
          onPointerDown={
            onPointerDown
          }
          onPointerMove={
            onPointerMove
          }
          onPointerUp={
            endDrag
          }
          onPointerCancel={
            endDrag
          }
          onPointerLeave={event => {
            if (
              dragRef.current
            ) {
              endDrag(event);
            }
          }}
          onKeyDown={
            onKeyDown
          }
          onWheel={event => {
            camRef.current.zoom =
              Math.max(
                0.35,
                Math.min(
                  2.2,
                  camRef.current
                    .zoom *
                    (
                      event.deltaY >
                      0
                        ? 0.94
                        : 1.06
                    ),
                ),
              );
          }}
        />

        <button
          type="button"
          aria-label={
            spin
              ? "Jeda gerakan kamera"
              : "Mulai gerakan kamera"
          }
          aria-pressed={spin}
          onClick={() =>
            setSpin(
              value => !value,
            )
          }
          className="absolute right-2 top-2 rounded border border-border/60 bg-card/70 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground backdrop-blur transition-colors hover:text-foreground"
        >
          {spin
            ? "pause"
            : "play"}
        </button>
      </div>
    );
  },
);
