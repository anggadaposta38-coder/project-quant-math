import { memo, useEffect, useRef, useState } from "react";

import {
  renderScene,
  type Camera,
  type Scene3D,
  type Vec3,
} from "@/lib/viz/engine3d";

import { applyDataMotion } from "@/lib/viz/motion-engine";
import { THEME_CHANGE_EVENT } from "@/lib/theme";

interface Plot3DProps {
  scene: Scene3D;
  height?: number;
  autoRotate?: boolean;
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

function morphScene(
  from: Scene3D | null,
  to: Scene3D,
  progress: number,
): Scene3D {
  if (from === null) {
    return to;
  }

  const t = easeInOut(progress);

  const points =
    from.points && to.points
      ? to.points.map((point, index) => {
          const previous = from.points?.[index];

          if (previous === undefined) {
            return point;
          }

          return {
            ...point,
            p: lerpVec(previous.p, point.p, t),
          };
        })
      : to.points;

  const lines =
    from.lines && to.lines
      ? to.lines.map((line, lineIndex) => {
          const previous = from.lines?.[lineIndex];

          if (
            previous === undefined ||
            previous.pts.length !== line.pts.length
          ) {
            return line;
          }

          return {
            ...line,
            pts: line.pts.map((point, pointIndex) => {
              const previousPoint = previous.pts[pointIndex];

              if (previousPoint === undefined) {
                return point;
              }

              return lerpVec(previousPoint, point, t);
            }),
          };
        })
      : to.lines;

  const quads =
    from.quads && to.quads
      ? to.quads.map((quad, quadIndex) => {
          const previous = from.quads?.[quadIndex];

          if (previous === undefined) {
            return quad;
          }

          const nextPoints: Vec3[] = quad.pts.map(
            (point, pointIndex) => {
              const previousPoint = previous.pts[pointIndex];

              if (previousPoint === undefined) {
                return point;
              }

              return lerpVec(previousPoint, point, t);
            },
          );

          const p0 = nextPoints[0];
          const p1 = nextPoints[1];
          const p2 = nextPoints[2];
          const p3 = nextPoints[3];

          if (
            p0 === undefined ||
            p1 === undefined ||
            p2 === undefined ||
            p3 === undefined
          ) {
            return quad;
          }

          return {
            ...quad,
            pts: [p0, p1, p2, p3] as [Vec3, Vec3, Vec3, Vec3],
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
}: Plot3DProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const camRef = useRef<Camera>({
    yaw: -0.75,
    pitch: 0.42,
    zoom: 1.02,
  });

  const dragRef = useRef<{ x: number; y: number } | null>(null);

  const sceneRef = useRef<Scene3D>(scene);
  const targetSceneRef = useRef<Scene3D>(scene);

  const transitionRef = useRef<{
    from: Scene3D;
    startedAt: number;
  } | null>(null);

  const [spin, setSpin] = useState(autoRotate);

  const spinRef = useRef(spin);
  spinRef.current = spin;

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

    if (canvas === null || wrap === null) {
      return;
    }

    const ctx = canvas.getContext("2d");

    if (ctx === null) {
      return;
    }

    let raf = 0;
    let last = performance.now();
    let clock = 0;
    let paused = document.hidden;

    const themeRef = {
      current: {
        grid: cssVar(wrap, "--chart-grid", "#2a3346"),
        text: cssVar(wrap, "--chart-axis", "#8b98b0"),
      },
    };

    const refreshTheme = () => {
      themeRef.current = {
        grid: cssVar(wrap, "--chart-grid", "#2a3346"),
        text: cssVar(wrap, "--chart-axis", "#8b98b0"),
      };
    };

    const onVisibilityChange = () => {
      paused = document.hidden;
      last = performance.now();

      if (paused === false && raf === 0) {
        raf = requestAnimationFrame(draw);
      }
    };

    const draw = (now: number) => {
      raf = 0;

      if (paused) {
        return;
      }

      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      clock += dt;

      if (spinRef.current && dragRef.current === null) {
        camRef.current.yaw += dt * 0.16;
      }

      const transition = transitionRef.current;

      if (transition !== null) {
        const progress = Math.min(
          1,
          (now - transition.startedAt) / 700,
        );

        sceneRef.current = morphScene(
          transition.from,
          targetSceneRef.current,
          progress,
        );

        if (progress >= 1) {
          sceneRef.current = targetSceneRef.current;
          transitionRef.current = null;
        }
      }

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = wrap.clientWidth;
      const h = height;

      if (
        canvas.width !== Math.floor(w * dpr) ||
        canvas.height !== Math.floor(h * dpr)
      ) {
        canvas.width = Math.floor(w * dpr);
        canvas.height = Math.floor(h * dpr);

        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const animatedScene = applyDataMotion(
        sceneRef.current,
        clock,
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

      raf = requestAnimationFrame(draw);
    };

    window.addEventListener(
      THEME_CHANGE_EVENT,
      refreshTheme,
    );

    document.addEventListener(
      "visibilitychange",
      onVisibilityChange,
    );

    raf = requestAnimationFrame(draw);

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
    event: React.PointerEvent,
  ) => {
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
    };

    (event.target as Element).setPointerCapture?.(
      event.pointerId,
    );
  };

  const onPointerMove = (
    event: React.PointerEvent,
  ) => {
    const drag = dragRef.current;

    if (drag === null) {
      return;
    }

    camRef.current.yaw +=
      (event.clientX - drag.x) * 0.008;

    camRef.current.pitch = Math.max(
      -1.4,
      Math.min(
        1.4,
        camRef.current.pitch +
          (event.clientY - drag.y) * 0.006,
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

  const onKeyDown = (
    event: React.KeyboardEvent<HTMLCanvasElement>,
  ) => {
    const step = event.shiftKey ? 0.18 : 0.08;

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
        camRef.current.pitch = Math.min(
          1.4,
          camRef.current.pitch + step,
        );
        break;

      case "ArrowDown":
        event.preventDefault();
        camRef.current.pitch = Math.max(
          -1.4,
          camRef.current.pitch - step,
        );
        break;

      case "+":
      case "=":
        event.preventDefault();

        camRef.current.zoom = Math.min(
          2.2,
          camRef.current.zoom * 1.06,
        );

        break;

      case "-":
      case "_":
        event.preventDefault();

        camRef.current.zoom = Math.max(
          0.35,
          camRef.current.zoom * 0.94,
        );

        break;

      case " ":
        event.preventDefault();
        setSpin((value) => !value);
        break;

      default:
        break;
    }
  };

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
          camRef.current.zoom = Math.max(
            0.35,
            Math.min(
              2.2,
              camRef.current.zoom *
                (event.deltaY > 0 ? 0.94 : 1.06),
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
        onClick={() => setSpin((value) => !value)}
        className="absolute right-2 top-2 rounded border border-border/60 bg-card/70 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground backdrop-blur transition-colors hover:text-foreground"
      >
        {spin ? "pause" : "rotate"}
      </button>
    </div>
  );
});
