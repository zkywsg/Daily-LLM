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

        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center" }}>
          <StackDepthSlider value={stackDepth} onChange={onStackDepthChange} />
          <HighwayShortcutToggle value={highwayShortcut} onChange={onHighwayShortcutChange} />
        </div>

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          滑动改变堆叠块数。主路梯度按 0.85^N 衰减，shortcut 路径无衰减直达底层。
          切换"有/无 shortcut"对比深网梯度命运。
        </p>
      </div>

      <div className={styles.stickyPanel}>
        <GradientHighwaySVG stackDepth={stackDepth} showShortcut={highwayShortcut} />
      </div>
    </div>
  );
}
