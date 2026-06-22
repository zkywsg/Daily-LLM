import { useParams, Link, Navigate } from "react-router";
import { Suspense, useEffect, useState } from "react";
import type { FamiliesData, FamilyId } from "../../types/family";
import familiesJson from "../../data/families.json";
import { familyColorVar } from "../../lib/colors";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { goldenSamples } from "./golden";
import styles from "./NodePage.module.css";

const data = familiesJson as unknown as FamiliesData;

// Glob all markdown files under repo root NN-xxx/ at build time.
// NodePage.tsx is at web/src/components/node/, so repo root is 4 levels up.
const markdownModules = import.meta.glob(
  "../../../../[0-9][0-9]-*/*.md",
  { query: "?raw", import: "default", eager: false }
) as Record<string, () => Promise<string>>;

function nodePathToModuleKey(nodePath: string): string {
  // nodePath: "01-cnn/05-resnet.md" → "../../../../01-cnn/05-resnet.md"
  return `../../../../${nodePath}`;
}

export function NodePage() {
  const { familyId, nodeSlug } = useParams<{
    familyId: FamilyId;
    nodeSlug: string;
  }>();

  const goldenKey = `${familyId}/${nodeSlug}`;
  const GoldenComponent = goldenSamples[goldenKey];

  if (GoldenComponent) {
    return (
      <Suspense
        fallback={
          <div style={{ padding: "var(--space-16)", textAlign: "center" }}>
            Loading golden sample…
          </div>
        }
      >
        <GoldenComponent />
      </Suspense>
    );
  }

  const family = data.families.find((f) => f.id === familyId);
  const node = family?.nodes.find(
    (n) => n.path.split("/").pop()?.replace(/\.md$/, "") === nodeSlug
  );

  const [markdown, setMarkdown] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);

  useEffect(() => {
    if (!node) return;
    let cancelled = false;
    setLoadError(null);
    setMarkdown(null);
    const key = nodePathToModuleKey(node.path);
    const loader = markdownModules[key];
    if (!loader) {
      setLoadError(
        `Markdown not found: ${node.path}. Available keys: ${Object.keys(
          markdownModules
        )
          .slice(0, 3)
          .join(", ")}...`
      );
      return;
    }
    loader()
      .then((md) => {
        if (!cancelled) setMarkdown(md);
      })
      .catch((e) => {
        if (!cancelled) setLoadError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [node?.path, retryNonce]);

  if (!family || !node) {
    return <Navigate to="/404" replace />;
  }

  const accent = familyColorVar(family.id);
  const body =
    markdown
      ?.replace(/^---[\s\S]*?---\n?/, "")
      // 剥离正文首个 H1 —— 页面 hero 已渲染节点标题,正文里的 `# Name (Year)` 会重复
      .replace(/^\s*#\s+[^\n]+\n+/, "") ?? "";

  // 同家族里的前后节点(family.nodes 已按 order 升序排好)
  const idx = family.nodes.findIndex((n) => n.path === node.path);
  const prev = idx > 0 ? family.nodes[idx - 1] : null;
  const next =
    idx >= 0 && idx < family.nodes.length - 1 ? family.nodes[idx + 1] : null;
  const slugOf = (n: typeof node) =>
    n.path.split("/").pop()!.replace(/\.md$/, "");

  return (
    <div className={styles.container}>
      <Link to={`/families/${family.id}`} className={styles.back}>
        ← 返回 {family.label}
      </Link>
      <div className={styles.meta}>
        <h1 style={{ fontSize: "var(--fs-3xl)", color: accent }}>
          {node.name} ({node.year})
        </h1>
        <div className={styles.metaLine}>
          作者: {node.authors.join(", ") || "—"}
        </div>
        <div className={styles.metaLine}>论文: {node.paper}</div>
      </div>
      <div className={styles.body}>
        {loadError && (
          <div className={styles.errorBox} role="alert">
            <p style={{ color: "var(--accent-warn)", margin: 0 }}>
              加载失败: {loadError}
            </p>
            <button
              type="button"
              className={styles.retryBtn}
              onClick={() => setRetryNonce((n) => n + 1)}
            >
              重试
            </button>
          </div>
        )}
        {!markdown && !loadError && <p>加载中…</p>}
        {markdown && <MarkdownRenderer markdown={body} sourcePath={node.path} />}
      </div>
      {(prev || next) && (
        <nav
          className={styles.prevNext}
          aria-label={`${family.label} 家族内节点导航`}
        >
          {prev ? (
            <Link
              to={`/families/${family.id}/${slugOf(prev)}`}
              className={styles.prevNextLink}
            >
              <span className={styles.prevNextDir}>← 上一篇</span>
              <span className={styles.prevNextName}>
                {prev.name} <span className={styles.prevNextYear}>({prev.year})</span>
              </span>
            </Link>
          ) : (
            <span aria-hidden="true" />
          )}
          {next ? (
            <Link
              to={`/families/${family.id}/${slugOf(next)}`}
              className={`${styles.prevNextLink} ${styles.prevNextRight}`}
            >
              <span className={styles.prevNextDir}>下一篇 →</span>
              <span className={styles.prevNextName}>
                {next.name} <span className={styles.prevNextYear}>({next.year})</span>
              </span>
            </Link>
          ) : (
            <span aria-hidden="true" />
          )}
        </nav>
      )}
    </div>
  );
}
