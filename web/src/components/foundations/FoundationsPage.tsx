import { useEffect, useMemo, useState } from "react";
import { Link, useParams, Navigate } from "react-router";
import { MarkdownRenderer } from "../node/MarkdownRenderer";

// 编译期收集 foundations/*/README.md 的 raw 文本(懒加载)
const foundationModules = import.meta.glob(
  "../../../../foundations/*/README.md",
  { query: "?raw", import: "default", eager: false }
) as Record<string, () => Promise<string>>;

interface Foundation {
  slug: string; // e.g. "01-neural-network-basics"
  path: string; // repo-root-relative
  moduleKey: string; // glob key
  title: string;
}

// 从 module key 抽 slug,只列一次(import.meta.glob 在 module scope 求值)
const foundations: Foundation[] = Object.keys(foundationModules)
  .map((key) => {
    // ../../../../foundations/01-neural-network-basics/README.md
    const m = key.match(/foundations\/([^/]+)\/README\.md$/);
    if (!m) return null;
    return {
      slug: m[1],
      path: `foundations/${m[1]}/README.md`,
      moduleKey: key,
      title: m[1], // 初始用 slug,加载 README 后改 H1
    };
  })
  .filter((x): x is Foundation => x !== null)
  .sort((a, b) => a.slug.localeCompare(b.slug));

/** /foundations — 列表页 */
export function FoundationsListPage() {
  // 异步取每个 README 的 H1 当真实标题
  const [titles, setTitles] = useState<Record<string, string>>({});
  useEffect(() => {
    let cancelled = false;
    Promise.all(
      foundations.map((f) =>
        foundationModules[f.moduleKey]().then((md) => {
          const h1 = md.match(/^#\s+(.+)$/m)?.[1] ?? f.slug;
          return [f.slug, h1] as const;
        })
      )
    ).then((pairs) => {
      if (cancelled) return;
      setTitles(Object.fromEntries(pairs));
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div
      style={{
        padding: "var(--space-6) var(--space-4)",
        maxWidth: 900,
        margin: "0 auto",
      }}
    >
      <Link
        to="/"
        style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)" }}
      >
        ← 返回主页
      </Link>
      <h1
        style={{
          fontSize: "var(--fs-3xl)",
          marginTop: "var(--space-4)",
          marginBottom: "var(--space-2)",
        }}
      >
        基础概念(Foundations)
      </h1>
      <p
        style={{
          fontSize: "var(--fs-lg)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        横切性的基础原理 — 不绑定某一家族,被多个节点引用。
      </p>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
          gap: "var(--space-4)",
        }}
      >
        {foundations.map((f) => (
          <Link
            key={f.slug}
            to={`/foundations/${f.slug}`}
            style={{
              display: "block",
              padding: "var(--space-4)",
              background: "var(--bg-surface)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              color: "var(--ink-primary)",
              textDecoration: "none",
              transition:
                "transform var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow = "var(--shadow-md)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "";
              e.currentTarget.style.boxShadow = "";
            }}
          >
            <div
              style={{
                fontSize: "var(--fs-sm)",
                color: "var(--ink-muted)",
                marginBottom: "var(--space-2)",
              }}
            >
              {f.slug}
            </div>
            <div
              style={{ fontSize: "var(--fs-md)", fontWeight: 600 }}
            >
              {titles[f.slug] ?? f.slug}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}

/** /foundations/:slug — 单个基础页 */
export function FoundationPage() {
  const { foundationSlug } = useParams<{ foundationSlug: string }>();
  const foundation = useMemo(
    () => foundations.find((f) => f.slug === foundationSlug),
    [foundationSlug]
  );

  const [markdown, setMarkdown] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);

  useEffect(() => {
    if (!foundation) return;
    let cancelled = false;
    setMarkdown(null);
    setLoadError(null);
    foundationModules[foundation.moduleKey]()
      .then((md) => {
        if (!cancelled) setMarkdown(md);
      })
      .catch((e) => {
        if (!cancelled) setLoadError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [foundation?.moduleKey, retryNonce]);

  if (!foundation) return <Navigate to="/404" replace />;

  return (
    <div
      style={{
        padding: "var(--space-6) var(--space-4)",
        maxWidth: 800,
        margin: "0 auto",
      }}
    >
      <Link
        to="/foundations"
        style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)" }}
      >
        ← 返回基础概念
      </Link>
      <div
        style={{
          fontSize: "var(--fs-sm)",
          color: "var(--ink-muted)",
          marginTop: "var(--space-6)",
        }}
      >
        foundations / {foundation.slug}
      </div>
      <div
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: "var(--fs-md)",
          lineHeight: 1.7,
          marginTop: "var(--space-4)",
        }}
      >
        {loadError && (
          <div
            role="alert"
            style={{
              padding: "var(--space-4)",
              border: "1px solid var(--accent-warn)",
              borderRadius: "var(--radius-md)",
              display: "flex",
              gap: "var(--space-4)",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <span style={{ color: "var(--accent-warn)" }}>
              加载失败: {loadError}
            </span>
            <button
              type="button"
              onClick={() => setRetryNonce((n) => n + 1)}
              style={{
                padding: "var(--space-2) var(--space-4)",
                border: "1px solid var(--accent-warn)",
                borderRadius: "var(--radius-md)",
                background: "var(--bg-surface)",
                color: "var(--accent-warn)",
                cursor: "pointer",
              }}
            >
              重试
            </button>
          </div>
        )}
        {!markdown && !loadError && <p>加载中…</p>}
        {markdown && (
          <MarkdownRenderer
            markdown={markdown.replace(/^---[\s\S]*?---\n?/, "")}
            sourcePath={foundation.path}
          />
        )}
      </div>
    </div>
  );
}
