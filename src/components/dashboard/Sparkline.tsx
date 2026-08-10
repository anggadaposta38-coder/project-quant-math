export function Sparkline({
  values,
  height = 34,
  tone,
}: {
  values: number[];
  height?: number;
  tone: "bull" | "bear";
}) {
  const pts = values.slice(-80);
  if (pts.length < 2) return <div style={{ height }} />;
  const min = Math.min(...pts);
  const max = Math.max(...pts);
  const span = max - min || 1;
  const d = pts
    .map((v, i) => {
      const x = (i / (pts.length - 1)) * 100;
      const y = height - ((v - min) / span) * (height - 4) - 2;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" className="h-full w-full">
      <path
        d={d}
        fill="none"
        strokeWidth={1.4}
        vectorEffect="non-scaling-stroke"
        className={tone === "bull" ? "stroke-bull" : "stroke-bear"}
      />
    </svg>
  );
}

export function ProbBar({ probs, labels }: { probs: number[]; labels: string[] }) {
  const tone = ["bg-bear", "bg-flat", "bg-bull"];
  return (
    <div>
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-surface-2">
        {probs.map((p, i) => (
          <div
            key={labels[i]}
            className={tone[i] ?? "bg-flat"}
            style={{ width: `${Math.max(p * 100, 0)}%` }}
          />
        ))}
      </div>
      <div className="mt-1.5 flex justify-between">
        {probs.map((p, i) => (
          <span key={labels[i]} className="tabular text-[10px] text-muted-foreground">
            {labels[i]} {(p * 100).toFixed(1)}%
          </span>
        ))}
      </div>
    </div>
  );
}
