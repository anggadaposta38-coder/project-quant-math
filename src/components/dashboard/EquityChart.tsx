export interface EquitySeries {
  name: string;
  colorClass: string; // tailwind stroke class, e.g. "stroke-primary"
  points: { t: number; equity: number }[];
}

/** Grafik garis multi-seri sederhana untuk membandingkan equity curve backtest. */
export function EquityChart({
  series,
  height = 220,
}: {
  series: EquitySeries[];
  height?: number;
}) {
  const all = series.flatMap((s) => s.points);
  if (all.length < 2) {
    return (
      <div style={{ height }} className="grid place-items-center text-xs text-muted-foreground">
        Belum ada data.
      </div>
    );
  }
  const tMin = Math.min(...all.map((p) => p.t));
  const tMax = Math.max(...all.map((p) => p.t));
  const eMin = Math.min(...all.map((p) => p.equity));
  const eMax = Math.max(...all.map((p) => p.equity));
  const tSpan = tMax - tMin || 1;
  const eSpan = eMax - eMin || 1;
  const pad = 4;

  const pathFor = (points: { t: number; equity: number }[]) =>
    points
      .map((p, i) => {
        const x = ((p.t - tMin) / tSpan) * 100;
        const y = height - pad - ((p.equity - eMin) / eSpan) * (height - 2 * pad);
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

  const zeroY = height - pad - ((1 - eMin) / eSpan) * (height - 2 * pad);

  return (
    <div>
      <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" className="h-[220px] w-full">
        {eMin < 1 && eMax > 1 ? (
          <line
            x1={0}
            x2={100}
            y1={zeroY}
            y2={zeroY}
            className="stroke-border"
            strokeWidth={1}
            strokeDasharray="2,2"
            vectorEffect="non-scaling-stroke"
          />
        ) : null}
        {series.map((s) => (
          <path
            key={s.name}
            d={pathFor(s.points)}
            fill="none"
            strokeWidth={1.6}
            vectorEffect="non-scaling-stroke"
            className={s.colorClass}
          />
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {series.map((s) => (
          <span key={s.name} className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className={`size-2 rounded-full ${s.colorClass.replace("stroke-", "bg-")}`} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}
