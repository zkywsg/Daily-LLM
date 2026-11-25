import { useEffect, useMemo, useState } from "react";
import GithubSlugger from "github-slugger";
import styles from "./NodeTOC.module.css";

interface NodeTOCProps {
  markdown: string;
}

interface TOCItem {
  level: 2 | 3;
  text: string;
  slug: string;
}

// 与 rehype-slug 行为对齐:都用 github-slugger,且按文档顺序重复名带 -1/-2 后缀
function extractToc(markdown: string): TOCItem[] {
  const items: TOCItem[] = [];
  const slugger = new GithubSlugger();
  // 跳过 ``` 代码块里的 # 行(避免把代码注释当标题)
  let inCode = false;
  for (const line of markdown.split("\n")) {
    if (line.startsWith("```")) {
      inCode = !inCode;
      continue;
    }
    if (inCode) continue;
    const m = line.match(/^(#{2,3})\s+(.+?)\s*$/);
    if (!m) continue;
    const level = (m[1].length as 2 | 3);
    // 去掉标题里的 markdown 强调 / 链接语法,只留可读文本
    const text = m[2]
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/\*([^*]+)\*/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .trim();
    items.push({ level, text, slug: slugger.slug(text) });
  }
  return items;
}

export function NodeTOC({ markdown }: NodeTOCProps) {
  const items = useMemo(() => extractToc(markdown), [markdown]);
  const [activeSlug, setActiveSlug] = useState<string | null>(
    items[0]?.slug ?? null
  );

  // IntersectionObserver:当前 viewport 顶端的标题高亮
  useEffect(() => {
    if (items.length === 0) return;
    const headings = items
      .map((i) => document.getElementById(i.slug))
      .filter((el): el is HTMLElement => el !== null);
    if (headings.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        // 找视口内最靠上的可见标题
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) {
          setActiveSlug(visible[0].target.id);
        }
      },
      { rootMargin: "0px 0px -70% 0px", threshold: 0 }
    );
    headings.forEach((h) => observer.observe(h));
    return () => observer.disconnect();
  }, [items]);

  if (items.length < 2) return null;

  return (
    <nav className={styles.toc} aria-label="本页目录">
      <div className={styles.tocTitle}>本页目录</div>
      <ul className={styles.tocList}>
        {items.map((it) => (
          <li
            key={it.slug}
            className={`${styles.tocItem} ${
              it.level === 3 ? styles.tocItemH3 : ""
            }`}
          >
            <a
              href={`#${it.slug}`}
              className={`${styles.tocLink} ${
                it.slug === activeSlug ? styles.tocLinkActive : ""
              }`}
            >
              {it.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
