import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router";
import type { FamiliesData, NodeData } from "../../types/family";
import familiesJson from "../../data/families.json";
import { familyColorVar } from "../../lib/colors";
import styles from "./SearchPalette.module.css";

const data = familiesJson as unknown as FamiliesData;
const allNodes: NodeData[] = data.families.flatMap((f) => f.nodes);

// 朴素打分:精确 > 前缀 > 包含 > 序内字符。返回 0 = 不匹配。
function scoreNode(node: NodeData, q: string): number {
  const lq = q.toLowerCase().trim();
  if (!lq) return 1;
  const name = node.name.toLowerCase();
  const idea = node.key_idea.toLowerCase();
  if (name === lq) return 1000;
  if (name.startsWith(lq)) return 500 + (50 - Math.min(50, name.length));
  if (name.includes(lq)) return 200;
  if (idea.includes(lq)) return 80;
  // 序内字符 fuzzy
  let i = 0;
  for (const ch of lq) {
    const found = name.indexOf(ch, i);
    if (found === -1) return 0;
    i = found + 1;
  }
  return 10;
}

function slugOf(node: NodeData) {
  return node.path.split("/").pop()!.replace(/\.md$/, "");
}

export function SearchPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  // 全局快捷键 ⌘K / Ctrl+K 开合, Escape 关
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isCmdK = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k";
      if (isCmdK) {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // 打开时聚焦、清空
  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIdx(0);
      // 等 portal 挂载后再聚焦
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const results = useMemo(() => {
    if (!query.trim()) return allNodes.slice(0, 8);
    return allNodes
      .map((n) => ({ node: n, score: scoreNode(n, query) }))
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
      .map((r) => r.node);
  }, [query]);

  // query 变化时重置高亮位置
  useEffect(() => {
    setActiveIdx(0);
  }, [query]);

  const go = (node: NodeData) => {
    navigate(`/families/${node.family}/${slugOf(node)}`);
    setOpen(false);
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => Math.min(results.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const pick = results[activeIdx];
      if (pick) go(pick);
    }
  };

  const trigger = (
    <button
      className={styles.trigger}
      onClick={() => setOpen(true)}
      aria-label="搜索节点(快捷键 ⌘K)"
    >
      搜索节点 <kbd>⌘K</kbd>
    </button>
  );

  const dialog =
    open &&
    createPortal(
      <div
        className={styles.overlay}
        onClick={(e) => {
          if (e.target === e.currentTarget) setOpen(false);
        }}
      >
        <div
          className={styles.dialog}
          role="dialog"
          aria-modal="true"
          aria-label="节点搜索"
        >
          <div className={styles.inputWrap}>
            <span className={styles.inputIcon} aria-hidden="true">⌕</span>
            <input
              ref={inputRef}
              className={styles.input}
              type="text"
              value={query}
              placeholder="搜索节点名 / 关键思想…"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKey}
            />
            <span className={styles.kbdHint}>ESC</span>
          </div>
          <div className={styles.list}>
            {results.length === 0 ? (
              <div className={styles.empty}>没有匹配的节点</div>
            ) : (
              results.map((n, i) => (
                <a
                  key={n.path}
                  className={`${styles.item} ${
                    i === activeIdx ? styles.itemActive : ""
                  }`}
                  href={`/families/${n.family}/${slugOf(n)}`}
                  onMouseEnter={() => setActiveIdx(i)}
                  onClick={(e) => {
                    e.preventDefault();
                    go(n);
                  }}
                >
                  <span
                    className={styles.dot}
                    style={{ background: familyColorVar(n.family) }}
                    aria-hidden="true"
                  />
                  <div className={styles.itemBody}>
                    <div className={styles.itemName}>{n.name}</div>
                    <div className={styles.itemMeta} title={n.key_idea}>
                      {n.family} · {n.key_idea}
                    </div>
                  </div>
                  <span className={styles.itemYear}>{n.year}</span>
                </a>
              ))
            )}
          </div>
          <div className={styles.footer}>
            <span>
              <kbd>↑</kbd>
              <kbd>↓</kbd> 选择
            </span>
            <span>
              <kbd>Enter</kbd> 打开
            </span>
            <span>
              <kbd>Esc</kbd> 关闭
            </span>
          </div>
        </div>
      </div>,
      document.body
    );

  return (
    <>
      {trigger}
      {dialog}
    </>
  );
}
