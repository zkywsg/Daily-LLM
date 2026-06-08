import type { ReactNode } from "react";
import { Link } from "react-router";

export function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <header
        style={{
          padding: "var(--space-4) var(--space-8)",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-surface)",
        }}
      >
        <Link
          to="/"
          style={{
            fontSize: "var(--fs-lg)",
            fontWeight: 600,
            color: "var(--ink-primary)",
          }}
        >
          Daily-LLM · 被逼出来的历史
        </Link>
      </header>
      <main style={{ flex: 1 }}>{children}</main>
      <footer
        style={{
          padding: "var(--space-4) var(--space-8)",
          borderTop: "1px solid var(--border)",
          fontSize: "var(--fs-sm)",
          color: "var(--ink-muted)",
        }}
      >
        Daily-LLM · 2026
      </footer>
    </>
  );
}
