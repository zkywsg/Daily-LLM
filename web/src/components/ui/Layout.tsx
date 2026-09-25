import type { ReactNode } from "react";
import { Link } from "react-router";
import styles from "./Layout.module.css";
import { SearchPalette } from "../search/SearchPalette";
import { ThemeToggle } from "./ThemeToggle";

export function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <header className={styles.header}>
        <Link to="/" className={styles.logo}>
          Daily-LLM · 深度学习与大模型
        </Link>
        <div className={styles.headerActions}>
          <Link to="/market" className={styles.marketLink}>
            竞品看板
          </Link>
          <SearchPalette />
          <ThemeToggle />
        </div>
      </header>
      <main style={{ flex: 1 }}>{children}</main>
      <footer className={styles.footer}>Daily-LLM · 2026</footer>
    </>
  );
}
