import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { BlockTypeToggle } from "../widgets/BlockTypeToggle";
import { BottleneckSVG } from "../widgets/BottleneckSVG";

interface Props {
  intuitionProse: string;
  blockType: "basic" | "bottleneck";
  onBlockTypeChange: (t: "basic" | "bottleneck") => void;
}

export function ResidualBlockStage({
  intuitionProse,
  blockType,
  onBlockTypeChange,
}: Props) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "var(--space-8)",
        alignItems: "start",
      }}
    >
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

        <BlockTypeToggle value={blockType} onChange={onBlockTypeChange} />

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          切换两种 block 结构，看参数量与内部结构变化。Bottleneck 用 1×1 降维节省 ~75% 参数。
        </p>
      </div>

      <div style={{ position: "sticky", top: "var(--space-8)" }}>
        <BottleneckSVG blockType={blockType} />
      </div>
    </div>
  );
}
