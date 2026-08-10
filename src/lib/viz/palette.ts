import { useEffect, useState } from "react";

export interface Palette {
  bull: string;
  bear: string;
  neutral: string;
  series: string[];
  accent: string;
  grid: string;
}

const FALLBACK: Palette = {
  bull: "#3ddc97",
  bear: "#ff5c7a",
  neutral: "#7d8aa5",
  series: ["#4cc9f0", "#ffb703", "#b5179e", "#3ddc97", "#f77f00", "#8093f1", "#ff5c7a", "#48cae4"],
  accent: "#ffb703",
  grid: "#2a3346",
};

const VAR_NAMES = [
  "--viz-1",
  "--viz-2",
  "--viz-3",
  "--viz-4",
  "--viz-5",
  "--viz-6",
  "--viz-7",
  "--viz-8",
];

/** Membaca token warna visualisasi dari design system (CSS custom properties). */
export function usePalette(): Palette {
  const [palette, setPalette] = useState<Palette>(FALLBACK);

  useEffect(() => {
    const cs = getComputedStyle(document.documentElement);
    const read = (name: string, fb: string) => cs.getPropertyValue(name).trim() || fb;
    setPalette({
      bull: read("--viz-bull", FALLBACK.bull),
      bear: read("--viz-bear", FALLBACK.bear),
      neutral: read("--viz-neutral", FALLBACK.neutral),
      accent: read("--viz-accent", FALLBACK.accent),
      grid: read("--chart-grid", FALLBACK.grid),
      series: VAR_NAMES.map((n, i) => read(n, FALLBACK.series[i] ?? FALLBACK.accent)),
    });
  }, []);

  return palette;
}
