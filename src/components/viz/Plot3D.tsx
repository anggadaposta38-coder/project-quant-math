import { memo, useEffect, useRef, useState } from "react";
import type { PointerEvent, KeyboardEvent } from "react";

import {
  renderScene,
  type Camera,
  type Scene3D,
  type Vec3,
} from "@/lib/viz/engine3d";
import {
  animateV5,
  type FlowProfile,
} from "@/lib/viz/motion-engine-v5";
import { THEME_CHANGE_EVENT } from "@/lib/theme";

interface Plot3DProps {
  scene: Scene3D;
  height?: number;
  autoRotate?: boolean;
  motionProfile?: FlowProfile;
  motionIntensity?: number;
}

function cssVar(el: HTMLElement, name: string, fallback: string) {
  const value = getComputedStyle(el).getPropertyValue(name).trim();
  return value || fallback;
}

const easeInOut = (t: number) =>
  t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function lerpVec(a: Vec3, b: Vec3, t: number): Vec3 {
  return [
    lerp(a[0], b[0], t),
    lerp(a[1], b[1], t),
    lerp(a[2], b[2], t),
  ];
}

/**
 * Morphs the quantitative geometry first.
 *
 * V5 is applied only after the data transition has been calculated, so the
 * flow layer never replaces the underlying quantitative geometry.
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
          return previous
            ? { ...point, p: lerpVec(previous.p, point.p, t) }
            : point;
        })
      : to.points;

  const lines =
    from.lines && to.lines
      ? to.lines.map((line, lineIndex) => {
          const previous = from.lines?.[lineIndex];

          if (
            !previous ||
            previous.pts.length !== line.pts.length
          ) {
            return line;
          }

          return {
            ...line,
            pts: line.pts.map((point, pointIndex) =>
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
          const previous = from.quads?.[quadIndex];

          if (
            !previous ||
            previous.pts.length !== quad.pts.length
          ) {
            return quad;
          }

          return {
            ...quad,
            pts: quad.pts.map((point, pointIndex) =>
              lerpVec(
                previous.pts[pointIndex]!,
                point,
                t,
              ),
            ) as [Vec3, Vec3, Vec3, Vec3],
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

export const Plot3D = memo(function Plot3D({
  scene,
  height = 320,
  autoRotate = true,
  motionProfile = "default",
  motionIntensity = 1,
}: Plot3DProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

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

  const sceneRef = useRef<Scene3D>(scene);
  const targetSceneRef = useRef<Scene3D>(scene);

  const transitionRef = useRef<{
    from: Scene3D;
    startedAt: number;
  } | null>(null);

  /*
   * Flow activity is intentionally separate from camera velocity.
   * A swipe can temporarily energize the data flow, then it naturally fades.
   */
  const flowActivityRef = useRef(0);

  const [spin, setSpin] = useState(autoRotate);
  const spinRef = useRef(spin);
  spinRef.current = spin;

  const profileRef = useRef<FlowProfile>(motionProfile);
  profileRef.current = motionProfile;

  const intensityRef = useRef(motionIntensity);
  intensityRef.current = motionIntensity;

  useEffect(() => {
    targetSceneRef.current = scene;

    transitionRef.current = {
      from: sceneRef.current,
      startedAt: performance.now(),
    };
  }, [scene]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;

    if (!canvas || !wrap) return;

    const ctx = canvas.getContext("2d", {
      alpha: true,
      desynchronized: true,
    });

    if (!ctx) return;

    let raf = 0;
    let last = performance.now();
    let clock = 0;
    let paused = document.hidden;

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

    const onVisibilityChange = () => {
      paused = document.hidden;
      last = performance.now();

      if (!paused && raf === 0) {
        raf = requestAnimationFrame(draw);
      }
    };

    const draw = (now: number) => {
      raf = 0;

      if (paused) return;

      const dt = Math.min(
        Math.max(
          (now - last) / 1000,
          0,
        ),
        0.05,
      );

      last = now;
      clock += dt;

      // --------------------------------------
      // 1. Manual-drag inertia
      // --------------------------------------
      if (!dragRef.current) {
        const velocity = velocityRef.current;

        camRef.current.yaw +=
          velocity.yaw * dt;

        camRef.current.pitch +=
          velocity.pitch * dt;

        velocity.yaw *=
          Math.exp(-dt * 7.5);

        velocity.pitch *=
          Math.exp(-dt * 8.5);

        if (
          Math.abs(velocity.yaw) <
          0.00002
        ) {
          velocity.yaw = 0;
        }

        if (
          Math.abs(velocity.pitch) <
          0.00002
        ) {
          velocity.pitch = 0;
        }
      }

      // --------------------------------------
      // 2. Gentle idle camera drift
      // --------------------------------------
      if (
        spinRef.current &&
        !dragRef.current
      ) {
        camRef.current.yaw +=
          dt * 0.035;

        camRef.current.pitch +=
          (
            Math.sin(clock * 0.22) *
              0.008 -
            (
              camRef.current.pitch -
              0.42
            )
          ) *
          dt *
          0.7;
      }

      camRef.current.pitch =
        Math.max(
          -1.25,
          Math.min(
            1.25,
            camRef.current.pitch,
          ),
        );

      // --------------------------------------
      // 3. Data geometry transition
      // --------------------------------------
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

        if (progress >= 1) {
          sceneRef.current =
            targetSceneRef.current;

          transitionRef.current =
            null;
        }
      }

      // --------------------------------------
      // 4. Mobile detection
      // --------------------------------------
      const width =
        wrap.clientWidth;

      const isMobile =
        width <= 640 ||
        (
          typeof navigator !==
            "undefined" &&
          /Android|iPhone|iPad|iPod/i.test(
            navigator.userAgent,
          )
        );

      // --------------------------------------
      // 5. V5 coherent data flow
      // --------------------------------------
      const animatedScene =
        animateV5(
          sceneRef.current,
          {
            time: clock,
            dt,
            activity:
              0.72 +
              flowActivityRef.current *
                0.28,
          },
          {
            profile:
              profileRef.current,
            intensity:
              intensityRef.current,
            mobile: isMobile,
            stateEnergy: 0.5,
            interactionEnergy:
              flowActivityRef.current,
          },
        );

      // --------------------------------------
      // 6. Fade interaction energy
      // --------------------------------------
      flowActivityRef.current *=
        Math.exp(-dt * 5.5);

      if (
        flowActivityRef.current <
        0.0005
      ) {
        flowActivityRef.current = 0;
      }

      // --------------------------------------
      // 7. Mobile-aware canvas resolution
      // --------------------------------------
      const rawDpr =
        window.devicePixelRatio ||
        1;

      const dpr = isMobile
        ? Math.min(rawDpr, 1.45)
        : Math.min(rawDpr, 1.8);

      const w = width;
      const h = height;

      const pixelWidth =
        Math.max(
          1,
          Math.floor(w * dpr),
        );

      const pixelHeight =
        Math.max(
          1,
          Math.floor(h * dpr),
        );

      if (
        canvas.width !== pixelWidth ||
        canvas.height !== pixelHeight
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

      renderScene(
        ctx,
        animatedScene,
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

      document.removeEventListener(
        "visibilitychange",
        onVisibilityChange,
      );
    };
  }, [height]);

  const onPointerDown = (
    event: PointerEvent<HTMLCanvasElement>,
  ) => {
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      lastTime: performance.now(),
    };

    velocityRef.current.yaw = 0;
    velocityRef.current.pitch = 0;

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

    if (!drag) return;

    const now =
      performance.now();

    const dt = Math.max(
      0.008,
      Math.min(
        (now - drag.lastTime) /
          1000,
        0.05,
      ),
    );

    const dx =
      event.clientX - drag.x;

    const dy =
      event.clientY - drag.y;

    const yawDelta =
      dx * 0.008;

    const pitchDelta =
      dy * 0.006;

    camRef.current.yaw +=
      yawDelta;

    camRef.current.pitch =
      Math.max(
        -1.25,
        Math.min(
          1.25,
          camRef.current.pitch +
            pitchDelta,
        ),
      );

    velocityRef.current.yaw =
      Math.max(
        -2.8,
        Math.min(
          2.8,
          yawDelta / dt,
        ),
      );

    velocityRef.current.pitch =
      Math.max(
        -2.2,
        Math.min(
          2.2,
          pitchDelta / dt,
        ),
      );

    /*
     * The swipe now also creates a short-lived flow impulse.
     * It does NOT change camera behavior; it only tells V5 that the scene
     * has just been interacted with.
     */
    const gestureSpeed =
      Math.sqrt(
        yawDelta * yawDelta +
        pitchDelta * pitchDelta,
      ) / dt;

    const gestureEnergy =
      Math.max(
        0,
        Math.min(
          1,
          gestureSpeed * 0.10,
        ),
      );

    flowActivityRef.current =
      Math.max(
        flowActivityRef.current,
        gestureEnergy,
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
        // Pointer capture may already have been released.
      }
    }

    dragRef.current = null;
  };

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
        camRef.current.yaw -= step;
        break;

      case "ArrowRight":
        event.preventDefault();
        camRef.current.yaw += step;
        break;

      case "ArrowUp":
        event.preventDefault();
        camRef.current.pitch =
          Math.min(
            1.25,
            camRef.current.pitch +
              step,
          );
        break;

      case "ArrowDown":
        event.preventDefault();
        camRef.current.pitch =
          Math.max(
            -1.25,
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

  return (
    <div
      ref={wrapRef}
      className="relative w-full select-none"
      style={{
        contain: "layout paint",
      }}
    >
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="Visualisasi 3D data kuantitatif. Gunakan sentuhan atau mouse untuk rotasi."
        tabIndex={0}
        className="w-full cursor-grab touch-none rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-primary active:cursor-grabbing"
        style={{ height }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={(event) => {
          if (dragRef.current) {
            endDrag(event);
          }
        }}
        onKeyDown={onKeyDown}
        onWheel={(event) => {
          camRef.current.zoom =
            Math.max(
              0.35,
              Math.min(
                2.2,
                camRef.current.zoom *
                  (
                    event.deltaY > 0
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
            ? "Jeda gerakan visualisasi 3D"
            : "Mulai gerakan visualisasi 3D"
        }
        aria-pressed={spin}
        onClick={() =>
          setSpin(
            (value) => !value,
          )
        }
        className="absolute right-2 top-2 rounded border border-border/60 bg-card/70 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground backdrop-blur transition-colors hover:text-foreground"
      >
        {spin ? "pause" : "play"}
      </button>
    </div>
  );
});
