import { useParams, Link, Navigate } from "react-router";
import { useEffect, useState } from "react";
import type { FamiliesData, FamilyId } from "../../types/family";
import familiesJson from "../../data/families.json";
import { familyColorVar } from "../../lib/colors";
import { MarkdownRenderer } from "./MarkdownRenderer";
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
  const family = data.families.find((f) => f.id === familyId);
  const node = family?.nodes.find(
    (n) => n.path.split("/").pop()?.replace(/\.md$/, "") === nodeSlug
  );

  const [markdown, setMarkdown] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!node) return;
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
      .then(setMarkdown)
      .catch((e) => setLoadError(String(e)));
  }, [node?.path]);

  if (!family || !node) {
    return <Navigate to="/404" replace />;
  }

  const accent = familyColorVar(family.id);
  const body = markdown?.replace(/^---[\s\S]*?---\n?/, "") ?? "";

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
          <p style={{ color: "var(--accent-warn)" }}>加载失败: {loadError}</p>
        )}
        {!markdown && !loadError && <p>加载中…</p>}
        {markdown && <MarkdownRenderer markdown={body} />}
      </div>
    </div>
  );
}
