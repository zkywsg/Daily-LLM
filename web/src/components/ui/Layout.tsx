import type { ReactNode } from "react";
import { Link } from "react-router";
import styles from "./Layout.module.css";

export function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <header className={styles.header}>
        <Link to="/" className={styles.logo}>
          Daily-LLM · 被逼出来的历史
        </Link>
      </header>
      <main style={{ flex: 1 }}>{children}</main>
      <footer className={styles.footer}>Daily-LLM · 2026</footer>
    </>
  );
}
