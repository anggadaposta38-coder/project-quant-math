import type { ReactNode } from "react";

export function Panel({
  title,
  subtitle,
  formula,
  right,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  formula?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel p-4 ${className}`}>
      <header className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">{title}</h3>
          {subtitle ? (
            <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
          ) : null}
          {formula ? (
            <code className="mt-1.5 block tabular text-[11px] text-primary/80">
              {formula}
            </code>
          ) : null}
        </div>
        {right}
      </header>
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  tone = "default",
  hint,
}: {
  label: string;
  value: string;
  tone?: "default" | "bull" | "bear" | "accent";
  hint?: string;
}) {
  const toneClass =
    tone === "bull"
      ? "text-bull"
      : tone === "bear"
        ? "text-bear"
        : tone === "accent"
          ? "text-accent"
          : "text-foreground";
  return (
    <div className="rounded-md border border-border/70 bg-surface-2/40 px-3 py-2">
      <div className="mono-label">{label}</div>
      <div className={`tabular mt-1 text-base font-semibold ${toneClass}`}>{value}</div>
      {hint ? <div className="mt-0.5 text-[10px] text-muted-foreground">{hint}</div> : null}
    </div>
  );
}
