import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { BlockTypeToggle } from "../widgets/BlockTypeToggle";
import { BottleneckSVG } from "../widgets/BottleneckSVG";
import { ShortcutToggle } from "../widgets/ShortcutToggle";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  blockType: "basic" | "bottleneck";
  onBlockTypeChange: (t: "basic" | "bottleneck") => void;
  showShortcut: boolean;
  onShowShortcutChange: (b: boolean) => void;
}

export function ResidualBlockStage({
  intuitionProse,
  blockType,
  onBlockTypeChange,
  showShortcut,
  onShowShortcutChange,
}: Props) {
  return (
    <div className={styles.grid}>
      <div>
        <h2 style={{ fontSize: "var(--fs-2xl)", marginBottom: "var(--space-4)" }}>
          残差的直觉
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {intuitionProse}
          </ReactMarkdown>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center" }}>
          <BlockTypeToggle value={blockType} onChange={onBlockTypeChange} />
          <ShortcutToggle value={showShortcut} onChange={onShowShortcutChange} />
        </div>

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          切换两种 block 结构看参数量变化。Hover 任意层显示 tensor shape。
          切换 F(x) / F(x)+x 看 shortcut 弧线是否出现。
        </p>
      </div>

      <div className={styles.stickyPanel}>
        <BottleneckSVG blockType={blockType} showShortcut={showShortcut} />
      </div>
    </div>
  );
}
