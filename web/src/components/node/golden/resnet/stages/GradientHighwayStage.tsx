import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { StackDepthSlider } from "../widgets/StackDepthSlider";
import { GradientHighwaySVG } from "../widgets/GradientHighwaySVG";

interface Props {
  mechanismProse: string;
  stackDepth: number;
  onStackDepthChange: (d: number) => void;
}

export function GradientHighwayStage({
  mechanismProse,
  stackDepth,
  onStackDepthChange,
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

        <StackDepthSlider value={stackDepth} onChange={onStackDepthChange} />

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          滑动改变堆叠块数。主路梯度按 0.85^N 衰减，shortcut 路径无衰减直达底层。
        </p>
      </div>

      <div style={{ position: "sticky", top: "var(--space-8)" }}>
        <GradientHighwaySVG stackDepth={stackDepth} />
      </div>
    </div>
  );
}
