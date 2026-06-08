import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { DepthSlider } from "../widgets/DepthSlider";
import { DegradationCurves } from "../widgets/DegradationCurves";

interface Props {
  beforeStuckOnProse: string;
  coreInsightProse: string;
  depth: number;
  onDepthChange: (d: number) => void;
}

export function DegradationStage({
  beforeStuckOnProse,
  coreInsightProse,
  depth,
  onDepthChange,
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
          之前卡在哪
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {beforeStuckOnProse}
          </ReactMarkdown>
        </div>

        <h2
          style={{
            fontSize: "var(--fs-2xl)",
            marginTop: "var(--space-8)",
            marginBottom: "var(--space-4)",
          }}
        >
          核心思想
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {coreInsightProse}
          </ReactMarkdown>
        </div>

        <DepthSlider value={depth} onChange={onDepthChange} />

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          滑动选择网络深度，右图实时变化。深度超过 30 层后，plain 网络会出现"先降后升"的退化形态。
        </p>
      </div>

      <div style={{ position: "sticky", top: "var(--space-8)" }}>
        <DegradationCurves depth={depth} />
      </div>
    </div>
  );
}
