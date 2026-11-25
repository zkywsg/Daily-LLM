import { useEffect, useState } from "react";
import styles from "./ThemeToggle.module.css";

type Theme = "light" | "dark";
const THEME_STORAGE_KEY = "daily-llm.theme";

function readStoredTheme(): Theme | null {
  try {
    const v = window.localStorage.getItem(THEME_STORAGE_KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

function detectSystemTheme(): Theme {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  } catch {
    // jsdom 默认无 matchMedia
    return "light";
  }
}

function applyTheme(theme: Theme) {
  document.documentElement.setAttribute("data-theme", theme);
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = readStoredTheme();
    return stored ?? detectSystemTheme();
  });

  useEffect(() => {
    applyTheme(theme);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // 忽略
    }
  }, [theme]);

  // 监听系统主题变化(用户没显式选过的情况下跟随)
  useEffect(() => {
    if (readStoredTheme()) return; // 已手动选过就不跟随
    let mql: MediaQueryList;
    try {
      mql = window.matchMedia("(prefers-color-scheme: dark)");
    } catch {
      return;
    }
    const onChange = (e: MediaQueryListEvent) =>
      setTheme(e.matches ? "dark" : "light");
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  const isDark = theme === "dark";
  return (
    <button
      type="button"
      className={styles.toggle}
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label={isDark ? "切换到浅色主题" : "切换到深色主题"}
      title={isDark ? "切换到浅色" : "切换到深色"}
    >
      {isDark ? "☀" : "☾"}
    </button>
  );
}
