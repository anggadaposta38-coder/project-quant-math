import { memo, useEffect, useRef, useState } from "react";
import { renderScene, type Camera, type Scene3D } from "@/lib/viz/engine3d";
import { THEME_CHANGE_EVENT } from "@/lib/theme";

interface Plot3DProps {
  scene: Scene3D;
  height?: number;
  autoRotate?: boolean;
}

function cssVar(el: HTMLElement, name: string, fallback: string) {
  const v = getComputedStyle(el).getPropertyValue(name).trim();
  return v || fallback;
}

export const Plot3D = memo(function Plot3D({ scene, height = 320, autoRotate = true }: Plot3DProps) {
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
    let paused = document.hidden;
    // Jam animasi terus berjalan (breathing/jitter/flow) walau kamera di-pause
    // lewat tombol rotate/pause — hanya rotasi kamera yang berhenti, bukan
    // "kehidupan" visualnya.
    let clock = 0;

    const themeRef = {
      current: {
        grid: cssVar(wrap, "--chart-grid", "#2a3346"),
        text: cssVar(wrap, "--chart-axis", "#8b98b0"),
      },
    };
    // Re-baca warna tema saat toggle light/dark — tanpa ini kanvas akan
    // "nyangkut" memakai grid/label color tema lama (warna gelap di atas
    // background terang jadi nyaris tak terlihat, atau sebaliknya).
    const refreshTheme = () => {
      themeRef.current = {
        grid: cssVar(wrap, "--chart-grid", "#2a3346"),
        text: cssVar(wrap, "--chart-axis", "#8b98b0"),
      };
    };
    window.addEventListener(THEME_CHANGE_EVENT, refreshTheme);
    const onVisibilityChange = () => {
      paused = document.hidden;
      last = performance.now();
      if (!paused && raf === 0) raf = requestAnimationFrame(draw);
    };
    document.addEventListener("visibilitychange", onVisibilityChange);

    const draw = (now: number) => {
      raf = 0;
      if (paused) return;
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      clock += dt;
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
      renderScene(ctx, sceneRef.current, camRef.current, w, h, themeRef.current, clock);
      raf = requestAnimationFrame(draw);
    };
    if (!paused) raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      raf = 0;
      window.removeEventListener(THEME_CHANGE_EVENT, refreshTheme);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
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

  const onKeyDown = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    const step = e.shiftKey ? 0.18 : 0.08;
    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        camRef.current.yaw -= step;
        break;
      case "ArrowRight":
        e.preventDefault();
        camRef.current.yaw += step;
        break;
      case "ArrowUp":
        e.preventDefault();
        camRef.current.pitch = Math.min(1.4, camRef.current.pitch + step);
        break;
      case "ArrowDown":
        e.preventDefault();
        camRef.current.pitch = Math.max(-1.4, camRef.current.pitch - step);
        break;
      case "+":
      case "=":
        e.preventDefault();
        camRef.current.zoom = Math.min(2.2, camRef.current.zoom * 1.06);
        break;
      case "-":
      case "_":
        e.preventDefault();
        camRef.current.zoom = Math.max(0.35, camRef.current.zoom * 0.94);
        break;
      case " ":
        e.preventDefault();
        setSpin((value) => !value);
        break;
    }
  };

  return (
    <div ref={wrapRef} className="relative w-full select-none">
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
        onWheel={(e) => {
          camRef.current.zoom = Math.max(
            0.35,
            Math.min(2.2, camRef.current.zoom * (e.deltaY > 0 ? 0.94 : 1.06)),
          );
        }}
      />
      <button
        type="button"
        aria-label={spin ? "Jeda rotasi visualisasi 3D" : "Mulai rotasi visualisasi 3D"}
        aria-pressed={spin}
        onClick={() => setSpin((s) => !s)}
        className="absolute right-2 top-2 rounded border border-border/60 bg-card/70 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground backdrop-blur transition-colors hover:text-foreground"
      >
        {spin ? "pause" : "rotate"}
      </button>
    </div>
  );
});
