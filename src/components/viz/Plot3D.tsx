import { useEffect, useRef, useState } from "react";
import { renderScene, type Camera, type Scene3D } from "@/lib/viz/engine3d";

interface Plot3DProps {
  scene: Scene3D;
  height?: number;
  autoRotate?: boolean;
}

function cssVar(el: HTMLElement, name: string, fallback: string) {
  const v = getComputedStyle(el).getPropertyValue(name).trim();
  return v || fallback;
}

export function Plot3D({ scene, height = 320, autoRotate = true }: Plot3DProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const camRef = useRef<Camera>({ yaw: -0.75, pitch: 0.42, zoom: 1.02 });
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [spin, setSpin] = useState(autoRotate);
  const sceneRef = useRef(scene);
  sceneRef.current = scene;
  const spinRef = useRef(spin);
  spinRef.current = spin;

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let last = performance.now();

    const theme = {
      grid: cssVar(wrap, "--chart-grid", "#2a3346"),
      text: cssVar(wrap, "--chart-axis", "#8b98b0"),
    };

    const draw = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (spinRef.current && !dragRef.current) camRef.current.yaw += dt * 0.22;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = wrap.clientWidth;
      const h = height;
      if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
        canvas.width = Math.floor(w * dpr);
        canvas.height = Math.floor(h * dpr);
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      renderScene(ctx, sceneRef.current, camRef.current, w, h, theme);
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [height]);

  const onPointerDown = (e: React.PointerEvent) => {
    dragRef.current = { x: e.clientX, y: e.clientY };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    camRef.current.yaw += (e.clientX - d.x) * 0.008;
    camRef.current.pitch = Math.max(
      -1.4,
      Math.min(1.4, camRef.current.pitch + (e.clientY - d.y) * 0.006),
    );
    dragRef.current = { x: e.clientX, y: e.clientY };
  };
  const endDrag = () => {
    dragRef.current = null;
  };

  return (
    <div ref={wrapRef} className="relative w-full select-none">
      <canvas
        ref={canvasRef}
        className="w-full cursor-grab touch-none rounded-lg active:cursor-grabbing"
        style={{ height }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
        onWheel={(e) => {
          camRef.current.zoom = Math.max(
            0.35,
            Math.min(2.2, camRef.current.zoom * (e.deltaY > 0 ? 0.94 : 1.06)),
          );
        }}
      />
      <button
        type="button"
        onClick={() => setSpin((s) => !s)}
        className="absolute right-2 top-2 rounded border border-border/60 bg-card/70 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground backdrop-blur transition-colors hover:text-foreground"
      >
        {spin ? "pause" : "rotate"}
      </button>
    </div>
  );
}
