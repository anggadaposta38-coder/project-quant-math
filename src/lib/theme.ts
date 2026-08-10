/**
 * Theme toggle (dark/light). Default aplikasi adalah dark ("quant terminal").
 * Preferensi disimpan di localStorage; tema light diaktifkan lewat class
 * `.light` di elemen <html> (lihat blok `.light { ... }` di styles.css).
 *
 * `THEME_CHANGE_EVENT` di-broadcast lewat window setiap kali tema berganti,
 * supaya consumer yang membaca CSS custom properties secara manual (canvas
 * 3D di Plot3D.tsx, usePalette) bisa re-read warnanya — kalau tidak, mereka
 * akan "nyangkut" di warna tema lama karena hanya membaca sekali saat mount.
 */

import { useCallback, useEffect, useState } from "react";

export const THEME_STORAGE_KEY = "quant-terminal-theme";
export const THEME_CHANGE_EVENT = "quant-terminal:themechange";

export type ThemeName = "dark" | "light";

/**
 * Skrip inline (dijalankan sebelum hydration) agar tidak ada flash tema
 * salah saat load: baca preferensi tersimpan sebelum browser sempat paint.
 */
export function getInitialThemeScript(): string {
  return `(function(){try{var k=${JSON.stringify(THEME_STORAGE_KEY)};var t=localStorage.getItem(k);if(t==='light'){document.documentElement.classList.add('light');}}catch(e){}})();`;
}

export function getCurrentTheme(): ThemeName {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.classList.contains("light") ? "light" : "dark";
}

export function applyTheme(theme: ThemeName): void {
  if (typeof document === "undefined") return;
  document.documentElement.classList.toggle("light", theme === "light");
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // localStorage tidak tersedia (mode privat dsb.) — tema tetap berlaku
    // untuk sesi ini, hanya tidak persisten.
  }
  window.dispatchEvent(new CustomEvent(THEME_CHANGE_EVENT, { detail: theme }));
}

/**
 * Hook React untuk komponen yang menampilkan tombol toggle tema.
 * Nilai awal "dark" dipakai saat SSR/mount pertama (cocok dengan default
 * server-rendered markup); disinkronkan ke tema aktual sekali di effect
 * pertama agar tidak terjadi hydration mismatch.
 */
export function useTheme(): { theme: ThemeName; toggleTheme: () => void } {
  const [theme, setTheme] = useState<ThemeName>("dark");

  useEffect(() => {
    setTheme(getCurrentTheme());
  }, []);

  const toggleTheme = useCallback(() => {
    const next: ThemeName = getCurrentTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    setTheme(next);
  }, []);

  return { theme, toggleTheme };
}
