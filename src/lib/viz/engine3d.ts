/** 
 * Lightweight 3D Canvas renderer.
 *
 * V4.2 renderer:
 * - perspective projection
 * - depth sorting
 * - data-driven surface shading
 * - subtle glow
 * - moving line flow
 * - point breathing
 * - surface ripple
 * - adaptive mobile quality
 *
 * API sengaja dipertahankan kompatibel dengan Scene3D/Plot3D lama.
 */

export type Vec3 = [number, number, number];

export interface Point3 {
  p: Vec3;
  color: string;
  size?: number;
  alpha?: number;
  phase?: number;
  glow?: boolean;
  jitter?: number;
}

export interface Line3 {
  pts: Vec3[];
  color: string;
  width?: number;
  alpha?: number;
  flow?: boolean;
  flowSpeed?: number;
}

export interface Quad3 {
  pts: [Vec3, Vec3, Vec3, Vec3];
  color: string;
  alpha?: number;
  stroke?: string;
  pulse?: number;
  ripple?: number;
}

export interface Scene3D {
  points?: Point3[];
  lines?: Line3[];
  quads?: Quad3[];
  axisLabels?: [string, string, string];
  ticks?: { pos: Vec3; text: string }[];
}

export interface Camera {
  yaw: number;
  pitch: number;
  zoom: number;
}

/* -------------------------------------------------------------------------- */
/* Math                                                                       */
/* -------------------------------------------------------------------------- */

export function rotate(p: Vec3, cam: Camera): Vec3 {
  const [x, y, z] = p;

  const cy = Math.cos(cam.yaw);
  const sy = Math.sin(cam.yaw);

  const x1 = x * cy - z * sy;
  const z1 = x * sy + z * cy;

  const cp = Math.cos(cam.pitch);
  const sp = Math.sin(cam.pitch);

  const y2 = y * cp - z1 * sp;
  const z2 = y * sp + z1 * cp;

  return [x1, y2, z2];
}

const DIST = 4.2;

export function project(
  p: Vec3,
  cam: Camera,
  w: number,
  h: number,
): { x: number; y: number; depth: number } {
  const [x, y, z] = rotate(p, cam);

  const depth = z + DIST;

  const f =
    (Math.min(w, h) / 2) *
    cam.zoom *
    1.9;

  const s = f / Math.max(depth, 0.15);

  return {
    x: w / 2 + x * s,
    y: h / 2 - y * s,
    depth,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function withAlpha(
  ctx: CanvasRenderingContext2D,
  alpha: number,
  fn: () => void,
) {
  const previous = ctx.globalAlpha;

  ctx.globalAlpha = clamp(alpha, 0, 1);

  fn();

  ctx.globalAlpha = previous;
}

/* -------------------------------------------------------------------------- */
/* Color helpers                                                              */
/* -------------------------------------------------------------------------- */

function parseColor(color: string): {
  r: number;
  g: number;
  b: number;
} | null {
  const value = color.trim();

  if (value.startsWith("#")) {
    const hex = value.slice(1);

    if (hex.length === 3) {
      return {
        r: parseInt(hex[0]! + hex[0]!, 16),
        g: parseInt(hex[1]! + hex[1]!, 16),
        b: parseInt(hex[2]! + hex[2]!, 16),
      };
    }

    if (hex.length === 6) {
      return {
        r: parseInt(hex.slice(0, 2), 16),
        g: parseInt(hex.slice(2, 4), 16),
        b: parseInt(hex.slice(4, 6), 16),
      };
    }
  }

  const rgb = value.match(
    /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/
  );

  if (rgb) {
    return {
      r: Number(rgb[1]),
      g: Number(rgb[2]),
      b: Number(rgb[3]),
    };
  }

  return null;
}

function shadeColor(
  color: string,
  amount: number,
): string {
  const parsed = parseColor(color);

  if (!parsed) return color;

  const r = clamp(
    Math.round(parsed.r + amount * 255),
    0,
    255,
  );

  const g = clamp(
    Math.round(parsed.g + amount * 255),
    0,
    255,
  );

  const b = clamp(
    Math.round(parsed.b + amount * 255),
    0,
    255,
  );

  return `rgb(${r}, ${g}, ${b})`;
}

/* -------------------------------------------------------------------------- */
/* Vector helpers                                                             */
/* -------------------------------------------------------------------------- */

function subtract(a: Vec3, b: Vec3): Vec3 {
  return [
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2],
  ];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(
    v[0],
    v[1],
    v[2],
  );

  if (length < 0.00001) {
    return [0, 1, 0];
  }

  return [
    v[0] / length,
    v[1] / length,
    v[2] / length,
  ];
}

/* -------------------------------------------------------------------------- */
/* Motion helpers                                                             */
/* -------------------------------------------------------------------------- */

function jitterOffset(
  p: Vec3,
  amount: number,
  phase: number,
  time: number,
): Vec3 {
  const ph = phase * Math.PI * 2;

  const jx =
    Math.sin(time * 0.6 + ph) *
    amount;

  const jy =
    Math.cos(time * 0.5 + ph * 1.3) *
    amount;

  const jz =
    Math.sin(time * 0.4 + ph * 0.7) *
    amount;

  return [
    p[0] + jx,
    p[1] + jy,
    p[2] + jz,
  ];
}

function rippleOffset(
  p: Vec3,
  amp: number,
  time: number,
): Vec3 {
  const [x, y, z] = p;

  const wave =
    Math.sin(
      x * 2.4 +
      z * 1.7 +
      time * 1.1,
    ) *
      0.6 +
    Math.sin(
      x * 1.15 -
      z * 2.6 +
      time * 0.75,
    ) *
      0.4;

  return [
    x,
    y + wave * amp,
    z,
  ];
}

/* -------------------------------------------------------------------------- */
/* Adaptive quality                                                           */
/* -------------------------------------------------------------------------- */

interface RenderQuality {
  mobile: boolean;
  pointLimit: number;
  quadStroke: boolean;
  glow: boolean;
  flow: boolean;
}

function detectQuality(
  w: number,
  h: number,
): RenderQuality {
  const mobile =
    w <= 640 ||
    (typeof navigator !== "undefined" &&
      /Android|iPhone|iPad|iPod/i.test(
        navigator.userAgent,
      ));

  if (mobile) {
    return {
      mobile: true,
      pointLimit: 900,
      quadStroke: false,
      glow: false,
      flow: true,
    };
  }

  return {
    mobile: false,
    pointLimit: 2200,
    quadStroke: true,
    glow: true,
    flow: true,
  };
}

/* -------------------------------------------------------------------------- */
/* Axis                                                                       */
/* -------------------------------------------------------------------------- */

const AXIS_EDGES: [Vec3, Vec3][] = [
  [
    [-1, -1, -1],
    [1, -1, -1],
  ],
  [
    [-1, -1, -1],
    [-1, 1, -1],
  ],
  [
    [-1, -1, -1],
    [-1, -1, 1],
  ],
  [
    [1, -1, -1],
    [1, -1, 1],
  ],
  [
    [1, -1, -1],
    [1, 1, -1],
  ],
  [
    [-1, -1, 1],
    [1, -1, 1],
  ],
  [
    [-1, -1, 1],
    [-1, 1, 1],
  ],
  [
    [-1, 1, -1],
    [1, 1, -1],
  ],
  [
    [-1, 1, -1],
    [-1, 1, 1],
  ],
  [
    [1, 1, -1],
    [1, 1, 1],
  ],
  [
    [-1, 1, 1],
    [1, 1, 1],
  ],
  [
    [1, -1, 1],
    [1, 1, 1],
  ],
];

/* -------------------------------------------------------------------------- */
/* Surface rendering                                                          */
/* -------------------------------------------------------------------------- */

function drawQuad(
  ctx: CanvasRenderingContext2D,
  q: Quad3,
  cam: Camera,
  w: number,
  h: number,
  time: number,
  quality: RenderQuality,
) {
  const worldPts = q.ripple
    ? q.pts.map((p) =>
        rippleOffset(
          p,
          q.ripple!,
          time,
        ),
      )
    : q.pts;

  const proj = worldPts.map((p) =>
    project(p, cam, w, h),
  );

  const edgeA = subtract(
    worldPts[1]!,
    worldPts[0]!,
  );

  const edgeB = subtract(
    worldPts[3]!,
    worldPts[0]!,
  );

  const normal = normalize(
    cross(edgeA, edgeB),
  );

  /*
   * Light direction is intentionally simple.
   * This is not physically accurate rendering.
   * It is visual depth lighting.
   */
  const light: Vec3 = [
    -0.45,
    0.75,
    0.55,
  ];

  const lightDirection =
    normalize(light);

  const lighting =
    normal[0] *
      lightDirection[0] +
    normal[1] *
      lightDirection[1] +
    normal[2] *
      lightDirection[2];

  const lightAmount =
    clamp(lighting, -1, 1);

  const pulse =
    q.pulse !== undefined
      ? Math.sin(
          time * 1.15 +
            q.pulse *
              Math.PI *
              2,
        ) * 0.06
      : 0;

  const shade =
    0.08 +
    lightAmount * 0.16 +
    pulse;

  const color = shadeColor(
    q.color,
    shade,
  );

  ctx.beginPath();

  ctx.moveTo(
    proj[0]!.x,
    proj[0]!.y,
  );

  ctx.lineTo(
    proj[1]!.x,
    proj[1]!.y,
  );

  ctx.lineTo(
    proj[2]!.x,
    proj[2]!.y,
  );

  ctx.lineTo(
    proj[3]!.x,
    proj[3]!.y,
  );

  ctx.closePath();

  withAlpha(
    ctx,
    q.alpha ?? 1,
    () => {
      ctx.fillStyle = color;
      ctx.fill();

      if (
        quality.quadStroke &&
        q.stroke
      ) {
        ctx.strokeStyle =
          q.stroke;

        ctx.lineWidth = 0.45;

        ctx.stroke();
      }
    },
  );
}

/* -------------------------------------------------------------------------- */
/* Line rendering                                                             */
/* -------------------------------------------------------------------------- */

function drawLine(
  ctx: CanvasRenderingContext2D,
  line: Line3,
  cam: Camera,
  w: number,
  h: number,
  time: number,
  quality: RenderQuality,
) {
  if (line.pts.length < 2) {
    return;
  }

  const proj = line.pts.map(
    (p) =>
      project(
        p,
        cam,
        w,
        h,
      ),
  );

  const width =
    line.width ?? 1;

  withAlpha(
    ctx,
    line.alpha ?? 1,
    () => {
      ctx.strokeStyle =
        line.color;

      ctx.lineWidth = width;

      /*
       * Main trajectory.
       */
      ctx.beginPath();

      ctx.moveTo(
        proj[0]!.x,
        proj[0]!.y,
      );

      for (
        let i = 1;
        i < proj.length;
        i++
      ) {
        ctx.lineTo(
          proj[i]!.x,
          proj[i]!.y,
        );
      }

      ctx.stroke();

      /*
       * Moving flow highlight.
       *
       * We intentionally use a second,
       * thinner stroke rather than a huge
       * particle system. This keeps mobile cheap.
       */
      if (
        line.flow &&
        quality.flow
      ) {
        const dash =
          Math.max(
            4,
            width * 3,
          );

        ctx.save();

        ctx.globalAlpha =
          (line.alpha ?? 1) *
          0.45;

        ctx.lineWidth =
          Math.max(
            0.5,
            width * 0.55,
          );

        ctx.setLineDash([
          dash,
          dash * 2.2,
        ]);

        ctx.lineDashOffset =
          -time *
          24 *
          (line.flowSpeed ?? 1);

        ctx.beginPath();

        ctx.moveTo(
          proj[0]!.x,
          proj[0]!.y,
        );

        for (
          let i = 1;
          i < proj.length;
          i++
        ) {
          ctx.lineTo(
            proj[i]!.x,
            proj[i]!.y,
          );
        }

        ctx.stroke();

        ctx.restore();
      }
    },
  );
}

/* -------------------------------------------------------------------------- */
/* Point rendering                                                            */
/* -------------------------------------------------------------------------- */

function drawPoint(
  ctx: CanvasRenderingContext2D,
  point: Point3,
  cam: Camera,
  w: number,
  h: number,
  time: number,
  quality: RenderQuality,
) {
  const phase =
    point.phase ?? 0;

  const worldP =
    point.jitter
      ? jitterOffset(
          point.p,
          point.jitter,
          phase,
          time,
        )
      : point.p;

  const projected =
    project(
      worldP,
      cam,
      w,
      h,
    );

  const pulse =
    point.glow
      ? 0.82 +
        0.18 *
          Math.sin(
            time * 1.6 +
              phase *
                Math.PI *
                2,
          )
      : 1;

  const baseRadius =
    ((point.size ?? 2) *
      2.6) /
    Math.max(
      projected.depth,
      0.3,
    );

  const radius = Math.max(
    0.65,
    baseRadius * pulse,
  );

  /*
   * Desktop only:
   * small halo, deliberately subtle.
   */
  if (
    point.glow &&
    quality.glow
  ) {
    const gradient =
      ctx.createRadialGradient(
        projected.x,
        projected.y,
        0,
        projected.x,
        projected.y,
        radius * 3,
      );

    gradient.addColorStop(
      0,
      point.color,
    );

    gradient.addColorStop(
      1,
      "rgba(0,0,0,0)",
    );

    withAlpha(
      ctx,
      (point.alpha ?? 1) *
        0.18,
      () => {
        ctx.fillStyle =
          gradient;

        ctx.beginPath();

        ctx.arc(
          projected.x,
          projected.y,
          radius * 3,
          0,
          Math.PI * 2,
        );

        ctx.fill();
      },
    );
  }

  withAlpha(
    ctx,
    (point.alpha ?? 1) *
      (point.glow
        ? 0.72 +
          0.28 * pulse
        : 1),
    () => {
      ctx.fillStyle =
        point.color;

      ctx.beginPath();

      ctx.arc(
        projected.x,
        projected.y,
        radius,
        0,
        Math.PI * 2,
      );

      ctx.fill();
    },
  );
}

/* -------------------------------------------------------------------------- */
/* Main renderer                                                              */
/* -------------------------------------------------------------------------- */

export function renderScene(
  ctx: CanvasRenderingContext2D,
  scene: Scene3D,
  cam: Camera,
  w: number,
  h: number,
  theme: {
    grid: string;
    text: string;
  },
  time = 0,
) {
  const quality =
    detectQuality(
      w,
      h,
    );

  ctx.clearRect(
    0,
    0,
    w,
    h,
  );

  /*
   * ------------------------------------------------------------------------
   * 1. Cube / coordinate frame
   * ------------------------------------------------------------------------
   */

  ctx.lineWidth = 1;
  ctx.strokeStyle =
    theme.grid;

  withAlpha(
    ctx,
    quality.mobile
      ? 0.28
      : 0.42,
    () => {
      for (
        const [a, b] of
        AXIS_EDGES
      ) {
        const pa =
          project(
            a,
            cam,
            w,
            h,
          );

        const pb =
          project(
            b,
            cam,
            w,
            h,
          );

        ctx.beginPath();

        ctx.moveTo(
          pa.x,
          pa.y,
        );

        ctx.lineTo(
          pb.x,
          pb.y,
        );

        ctx.stroke();
      }
    },
  );

  /*
   * ------------------------------------------------------------------------
   * 2. Depth sorted geometry
   * ------------------------------------------------------------------------
   */

  type Item = {
    depth: number;
    draw: () => void;
  };

  const items: Item[] =
    [];

  /*
   * QUADS
   */
  for (
    const quad of
    scene.quads ?? []
  ) {
    const pts = quad.ripple
      ? quad.pts.map(
          (p) =>
            rippleOffset(
              p,
              quad.ripple!,
              time,
            ),
        )
      : quad.pts;

    const projected =
      pts.map((p) =>
        project(
          p,
          cam,
          w,
          h,
        ),
      );

    const depth =
      projected.reduce(
        (sum, p) =>
          sum + p.depth,
        0,
      ) / 4;

    items.push({
      depth,
      draw: () =>
        drawQuad(
          ctx,
          quad,
          cam,
          w,
          h,
          time,
          quality,
        ),
    });
  }

  /*
   * LINES
   */
  for (
    const line of
    scene.lines ?? []
  ) {
    if (
      line.pts.length < 2
    ) {
      continue;
    }

    const projected =
      line.pts.map((p) =>
        project(
          p,
          cam,
          w,
          h,
        ),
      );

    const depth =
      projected.reduce(
        (sum, p) =>
          sum + p.depth,
        0,
      ) /
      projected.length;

    items.push({
      depth,
      draw: () =>
        drawLine(
          ctx,
          line,
          cam,
          w,
          h,
          time,
          quality,
        ),
    });
  }

  /*
   * POINT CLOUD
   *
   * Mobile point cap:
   * prevents a huge scene from becoming
   * an expensive canvas operation.
   */
  const points =
    scene.points ?? [];

  const pointStride =
    quality.mobile &&
    points.length >
      quality.pointLimit
      ? Math.ceil(
          points.length /
            quality.pointLimit,
        )
      : 1;

  for (
    let i = 0;
    i < points.length;
    i += pointStride
  ) {
    const point =
      points[i]!;

    const worldP =
      point.jitter
        ? jitterOffset(
            point.p,
            point.jitter,
            point.phase ??
              0,
            time,
          )
        : point.p;

    const projected =
      project(
        worldP,
        cam,
        w,
        h,
      );

    items.push({
      depth:
        projected.depth,
      draw: () =>
        drawPoint(
          ctx,
          point,
          cam,
          w,
          h,
          time,
          quality,
        ),
    });
  }

  /*
   * ------------------------------------------------------------------------
   * 3. Painter's algorithm
   * ------------------------------------------------------------------------
   */

  items.sort(
    (a, b) =>
      b.depth -
      a.depth,
  );

  for (
    const item of items
  ) {
    item.draw();
  }

  /*
   * ------------------------------------------------------------------------
   * 4. Axis labels
   * ------------------------------------------------------------------------
   */

  if (
    scene.axisLabels
  ) {
    ctx.font =
      quality.mobile
        ? "500 10px ui-monospace, monospace"
        : "500 11px ui-monospace, monospace";

    ctx.fillStyle =
      theme.text;

    const anchors: Vec3[] =
      [
        [1.18, -1, -1],
        [-1, 1.18, -1],
        [-1, -1, 1.18],
      ];

    scene.axisLabels.forEach(
      (label, index) => {
        const anchor =
          anchors[index];

        if (!anchor) {
          return;
        }

        const projected =
          project(
            anchor,
            cam,
            w,
            h,
          );

        ctx.textAlign =
          "center";

        withAlpha(
          ctx,
          0.9,
          () =>
            ctx.fillText(
              label,
              projected.x,
              projected.y,
            ),
        );
      },
    );
  }

  /*
   * ------------------------------------------------------------------------
   * 5. Tick labels
   * ------------------------------------------------------------------------
   */

  for (
    const tick of
    scene.ticks ?? []
  ) {
    const projected =
      project(
        tick.pos,
        cam,
        w,
        h,
      );

    ctx.font =
      quality.mobile
        ? "400 8px ui-monospace, monospace"
        : "400 9px ui-monospace, monospace";

    ctx.fillStyle =
      theme.text;

    ctx.textAlign =
      "center";

    withAlpha(
      ctx,
      0.72,
      () =>
        ctx.fillText(
          tick.text,
          projected.x,
          projected.y,
        ),
    );
  }
}

/* -------------------------------------------------------------------------- */
/* Data helpers                                                               */
/* -------------------------------------------------------------------------- */

/** Normalisasi linear ke rentang [-1, 1]. */
export function makeScaler(
  min: number,
  max: number,
) {
  const span =
    max - min;

  if (
    !Number.isFinite(
      span,
    ) ||
    span === 0
  ) {
    return () => 0;
  }

  return (value: number) =>
    ((value - min) /
      span) *
      2 -
    1;
}

export function extent(
  values: number[],
): [number, number] {
  let min = Infinity;
  let max = -Infinity;

  for (
    const value of values
  ) {
    if (
      !Number.isFinite(
        value,
      )
    ) {
      continue;
    }

    if (value < min) {
      min = value;
    }

    if (value > max) {
      max = value;
    }
  }

  if (
    !Number.isFinite(
      min,
    )
  ) {
    return [0, 1];
  }

  if (min === max) {
    return [
      min - 1,
      max + 1,
    ];
  }

  return [min, max];
}

/**
 * Deterministic phase 0..1.
 *
 * Golden-ratio distribution membuat
 * breathing/jitter tidak sinkron antar titik.
 */
export function phaseOf(
  index: number,
): number {
  return (
    (index *
      0.6180339887498949) %
    1
  );
}
