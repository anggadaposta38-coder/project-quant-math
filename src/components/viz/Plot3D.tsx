import {
  memo,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";

import {
  renderScene,
  type Camera,
  type Scene3D,
  type Vec3,
} from "@/lib/viz/engine3d";

import { THEME_CHANGE_EVENT } from "@/lib/theme";

import {
  applyMotionField,
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

const easeInOut = (t: number) =>
  t < 0.5
    ? 2 * t * t
    : 1 - Math.pow(-2 * t + 2, 2) / 2;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpVec(a: Vec3, b: Vec3, t: number): Vec3 {
  return [
    lerp(a[0], b[0], t),
    lerp(a[1], b[1], t),
    lerp(a[2], b[2], t),
  ];
}

/* -------------------------------------------------------------------------- */
/* V2 geometry morph                                                          */
/*                                                                            */
/* V2 remains responsible for transitioning from the old quant result to the */
/* new quant result. V3 then takes over continuous particle motion.           */
/* -------------------------------------------------------------------------- */

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
            ? {
                ...point,
                p: lerpVec(previous.p, point.p, t),
              }
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

          if (!previous) {
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

/* -------------------------------------------------------------------------- */
/* V3 particle helpers                                                        */
/* -------------------------------------------------------------------------- */

/**
 * V3 particles are tied to the current Scene3D point count.
 *
 * When the data shape changes significantly, rebuild the particle state.
 * This prevents stale particles from one visualization mode leaking into
 * another visualization mode.
 */
function particlesMatchScene(
  particles: MotionFieldParticle[],
  scene: Scene3D,
): boolean {
  return particles.length === (scene.points?.length ?? 0);
}

/**
 * Creates a stable particle state from the scene.
 *
 * No random initialization.
 */
function createParticles(
  scene: Scene3D,
): MotionFieldParticle[] {
  return createMotionFieldParticles(scene);
}

/**
 * The V3 engine moves only point geometry.
 *
 * Everything else from the scene is preserved.
 */
function applyParticles(
  scene: Scene3D,
  particles: MotionFieldParticle[],
): Scene3D {
  return applyMotionField(scene, particles);
}

/* -------------------------------------------------------------------------- */
/* Component                                                                  */
/* -------------------------------------------------------------------------- */

export const Plot3D = memo(function Plot3D({
  scene,
  height = 320,
  autoRotate = true,
}: Plot3DProps) {
  const canvasRef =
    useRef<HTMLCanvasElement | null>(null);

  const wrapRef =
    useRef<HTMLDivElement | null>(null);

  /* ------------------------------------------------------------------------ */
  /* Camera                                                                   */
  /* ------------------------------------------------------------------------ */

  const camRef = useRef<Camera>({
    yaw: -0.75,
    pitch: 0.42,
    zoom: 1.02,
  });

  const dragRef =
    useRef<{ x: number; y: number } | null>(null);

  /* ------------------------------------------------------------------------ */
  /* Scene state                                                              */
  /* ------------------------------------------------------------------------ */

  const sceneRef = useRef<Scene3D>(scene);

  const targetSceneRef =
    useRef<Scene3D>(scene);

  const transitionRef = useRef<{
    from: Scene3D;
    startedAt: number;
  } | null>(null);

  /* ------------------------------------------------------------------------ */
  /* V3 motion state                                                          */
  /* ------------------------------------------------------------------------ */

  const particlesRef =
    useRef<MotionFieldParticle[]>([]);

  const motionSceneRef =
    useRef<Scene3D>(scene);

  /*
   * Prevent V3 from immediately replacing the V2 morph.
   *
   * Flow:
   *
   * new data
   *    ↓
   * V2 morph
   *    ↓
   * stable target
   *    ↓
   * V3 vector field
   *    ↓
   * continuous motion
   */
  const v3EnabledRef = useRef(false);

  /* ------------------------------------------------------------------------ */
  /* UI state                                                                 */
  /* ------------------------------------------------------------------------ */

  const [spin, setSpin] =
    useState(autoRotate);

  const spinRef =
    useRef(spin);

  spinRef.current = spin;

  /* ------------------------------------------------------------------------ */
  /* New scene arrives                                                        */
  /* ------------------------------------------------------------------------ */

  useEffect(() => {
    const previous = sceneRef.current;

    targetSceneRef.current = scene;

    transitionRef.current = {
      from: previous,
      startedAt: performance.now(),
    };

    /*
     * Disable V3 temporarily.
     *
     * This is important because the incoming scene represents a new
     * quantitative state. We first want V2 to morph into that state.
     */
    v3EnabledRef.current = false;

    /*
     * The particle simulation is rebuilt from the new geometry.
     *
     * We intentionally do not mutate it here. The render loop will
     * initialize it at the correct moment.
     */
    particlesRef.current = [];

    motionSceneRef.current = scene;
  }, [scene]);

  /* ------------------------------------------------------------------------ */
  /* Renderer + animation loop                                                */
  /* ------------------------------------------------------------------------ */

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;

    if (!canvas || !wrap) {
      return;
    }

    const ctx = canvas.getContext("2d");

    if (!ctx) {
      return;
    }

    let raf = 0;

    let last = performance.now();

    let clock = 0;

    let paused = document.hidden;

    /* ---------------------------------------------------------------------- */
    /* Theme                                                                   */
    /* ---------------------------------------------------------------------- */

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

    /* ---------------------------------------------------------------------- */
    /* Visibility                                                              */
    /* ---------------------------------------------------------------------- */

    const onVisibilityChange = () => {
      paused = document.hidden;

      last = performance.now();

      if (!paused && raf === 0) {
        raf = requestAnimationFrame(draw);
      }
    };

    /* ---------------------------------------------------------------------- */
    /* V3 simulation                                                           */
    /* ---------------------------------------------------------------------- */

    const updateMotion = (
      currentScene: Scene3D,
      deltaSeconds: number,
    ): Scene3D => {
      const pointCount =
        currentScene.points?.length ?? 0;

      if (pointCount === 0) {
        return currentScene;
      }

      /*
       * First frame after a new scene:
       * initialize deterministic particles directly from geometry.
       */
      if (
        !particlesMatchScene(
          particlesRef.current,
          currentScene,
        )
      ) {
        particlesRef.current =
          createParticles(currentScene);

        motionSceneRef.current =
          currentScene;

        return currentScene;
      }

      /*
       * Step the V3 vector field.
       *
       * V3 itself handles:
       * - tangential flow
       * - cohesion
       * - separation
       * - inertia
       * - damping
       * - bounded movement
       */
      particlesRef.current =
        stepMotionField(
          particlesRef.current,
          currentScene,
          deltaSeconds,
          {
            fieldStrength: 0.85,
            cohesion: 0.32,
            separation: 0.18,
            inertia: 0.92,
            damping: 0.18,
            maxSpeed: 1.35,
            stateBias: 0.16,
          },
        );

      const nextScene =
        applyParticles(
          currentScene,
          particlesRef.current,
        );

      motionSceneRef.current =
        nextScene;

      return nextScene;
    };

    /* ---------------------------------------------------------------------- */
    /* Main draw loop                                                          */
    /* ---------------------------------------------------------------------- */

    const draw = (now: number) => {
      raf = 0;

      if (paused) {
        return;
      }

      const dt = Math.min(
        (now - last) / 1000,
        0.05,
      );

      last = now;

      clock += dt;

      /* -------------------------------------------------------------------- */
      /* Camera auto rotation                                                 */
      /* -------------------------------------------------------------------- */

      if (
        spinRef.current &&
        !dragRef.current
      ) {
        camRef.current.yaw += dt * 0.16;
      }

      /* -------------------------------------------------------------------- */
      /* V2 transition                                                        */
      /* -------------------------------------------------------------------- */

      const transition =
        transitionRef.current;

      if (transition) {
        const progress = Math.min(
          1,
          (now - transition.startedAt) / 700,
        );

        const morphed =
          morphScene(
            transition.from,
            targetSceneRef.current,
            progress,
          );

        sceneRef.current = morphed;

        /*
         * During V2 morph:
         *
         * V3 is OFF.
         *
         * This keeps the new quant result visually coherent instead of
         * allowing the particle field to fight the incoming geometry.
         */
        if (progress >= 1) {
          sceneRef.current =
            targetSceneRef.current;

          motionSceneRef.current =
            targetSceneRef.current;

          particlesRef.current =
            createParticles(
              targetSceneRef.current,
            );

          v3EnabledRef.current = true;

          transitionRef.current = null;
        }
      } else {
        /*
         * Once the new data has settled, V3 continuously drives the points.
         */
        if (v3EnabledRef.current) {
          sceneRef.current =
            updateMotion(
              sceneRef.current,
              dt,
            );
        }
      }

      /* -------------------------------------------------------------------- */
      /* Canvas sizing                                                         */
      /* -------------------------------------------------------------------- */

      const dpr = Math.min(
        window.devicePixelRatio || 1,
        2,
      );

      const w = wrap.clientWidth;

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

      /* -------------------------------------------------------------------- */
      /* Render                                                                */
      /* -------------------------------------------------------------------- */

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
        sceneRef.current,
        camRef.current,
        w,
        h,
        themeRef.current,
        clock,
      );

      raf =
        requestAnimationFrame(draw);
    };

    /* ---------------------------------------------------------------------- */
    /* Events                                                                  */
    /* ---------------------------------------------------------------------- */

    window.addEventListener(
      THEME_CHANGE_EVENT,
      refreshTheme,
    );

    document.addEventListener(
      "visibilitychange",
      onVisibilityChange,
    );

    raf =
      requestAnimationFrame(draw);

    /* ---------------------------------------------------------------------- */
    /* Cleanup                                                                 */
    /* ---------------------------------------------------------------------- */

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

  /* ------------------------------------------------------------------------ */
  /* Pointer controls                                                         */
  /* ------------------------------------------------------------------------ */

  const onPointerDown = (
    event: ReactPointerEvent<HTMLCanvasElement>,
  ) => {
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
    };

    (
      event.target as Element
    ).setPointerCapture?.(
      event.pointerId,
    );
  };

  const onPointerMove = (
    event: ReactPointerEvent<HTMLCanvasElement>,
  ) => {
    const drag =
      dragRef.current;

    if (!drag) {
      return;
    }

    camRef.current.yaw +=
      (event.clientX - drag.x) *
      0.008;

    camRef.current.pitch =
      Math.max(
        -1.4,
        Math.min(
          1.4,
          camRef.current.pitch +
            (event.clientY - drag.y) *
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

  /* ------------------------------------------------------------------------ */
  /* Keyboard controls                                                        */
  /* ------------------------------------------------------------------------ */

  const onKeyDown = (
    event: ReactKeyboardEvent<HTMLCanvasElement>,
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

  /* ------------------------------------------------------------------------ */
  /* Render UI                                                                */
  /* ------------------------------------------------------------------------ */

  return (
    <div
      ref={wrapRef}
      className="relative w-full select-none"
    >
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="Visualisasi 3D data kuantitatif. Gunakan tombol panah untuk rotasi dan plus atau minus untuk zoom."
        tabIndex={0}
        className="w-full cursor-grab touch-none rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-primary active:cursor-grabbing"
        style={{ height }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
        onKeyDown={onKeyDown}
        onWheel={(event) => {
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
        {spin ? "pause" : "rotate"}
      </button>
    </div>
  );
});
