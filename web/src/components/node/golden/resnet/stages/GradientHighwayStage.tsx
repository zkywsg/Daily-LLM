import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { StackDepthSlider } from "../widgets/StackDepthSlider";
import { GradientHighwaySVG } from "../widgets/GradientHighwaySVG";
import { HighwayShortcutToggle } from "../widgets/HighwayShortcutToggle";
import styles from "./Stage.module.css";

interface Props {
  mechanismProse: string;
  stackDepth: number;
  onStackDepthChange: (d: number) => void;
  highwayShortcut: boolean;
  onHighwayShortcutChange: (b: boolean) => void;
}

export function GradientHighwayStage({
  mechanismProse,
  stackDepth,
  onStackDepthChange,
  highwayShortcut,
  onHighwayShortcutChange,
}: Props) {
  const [playKey, setPlayKey] = useState(0);
  return (
    <div className={styles.grid}>
      <div>
        <h2 style={{ fontSize: "var(--fs-2xl)", marginBottom: "var(--space-4)" }}>
          梯度高速公路
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {mechanismProse}
          </ReactMarkdown>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "var(--space-2)" }}>
          <StackDepthSlider value={stackDepth} onChange={onStackDepthChange} />
          <HighwayShortcutToggle value={highwayShortcut} onChange={onHighwayShortcutChange} />
          <button
            onClick={() => setPlayKey((k) => k + 1)}
            style={{
              padding: "var(--space-2) var(--space-4)",
              borderRadius: "var(--radius-full)",
              background: "var(--accent-link)",
              color: "var(--bg-surface)",
              fontSize: "var(--fs-sm)",
              fontWeight: 500,
              border: "none",
              cursor: "pointer",
              marginLeft: "var(--space-3)",
            }}
          >
            ▶ 播放梯度反传
          </button>
        </div>

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          滑动改变堆叠块数。切换"有/无 shortcut"对比深网梯度命运。
          点 ▶ 看红色（主路衰减）和蓝色（shortcut 恒粗）粒子同时反传——红球到达 input 时几乎消失，蓝球毫发无损。
        </p>
      </div>

      <div className={styles.stickyPanel}>
        <GradientHighwaySVG
          stackDepth={stackDepth}
          showShortcut={highwayShortcut}
          playKey={playKey}
        />
      </div>
    </div>
  );
}
