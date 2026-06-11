import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { RESNET_SOURCE_PATH } from "../lib/prose";
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
          <MarkdownRenderer
            markdown={mechanismProse}
            sourcePath={RESNET_SOURCE_PATH}
          />
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
          Hover 某块看数学公式 + 具体衰减百分比。点 ▶ 看红蓝粒子同时反传。
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
