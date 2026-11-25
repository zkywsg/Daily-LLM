import { ZERO_SHOT_EXAMPLES } from "../lib/data";

interface Props {
  exampleIdx: number;
}

const W = 700;
const H = 320;

// Zero-shot 流程:
//   image → image_emb
//   "a photo of a {class}"(N 个候选)→ N 个 text_emb
//   image 跟每个 text 算 cos sim → softmax → 取最高
// 让 viewer 看见"无需 fine-tune,只换 prompt 就能换分类任务"。

export function ZeroShotClassifier({ exampleIdx }: Props) {
  const ex = ZERO_SHOT_EXAMPLES[exampleIdx];
  const sims = ex.candidates;
  const maxSim = Math.max(...sims.map((c) => c.sim));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CLIP zero-shot classification flow">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Zero-shot 推理:image vs N 个 "a photo of a &#123;class&#125;" 谁 cos sim 最高
      </text>

      {/* 左:image */}
      <rect x={30} y={50} width={120} height={120} rx={8} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} />
      <text x={90} y={130} textAnchor="middle" fontSize={60}>{ex.emoji}</text>
      <text x={90} y={185} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">输入 image</text>

      {/* 中:Vision encoder → image_emb */}
      <line x1={150} y1={110} x2={210} y2={110} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#zs-arr)" />
      <defs>
        <marker id="zs-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <rect x={210} y={90} width={100} height={40} rx={4} fill="#fce7f3" stroke="#ec4899" />
      <text x={260} y={115} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">Image Enc</text>
      <line x1={310} y1={110} x2={370} y2={110} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#zs-arr)" />

      {/* image_emb */}
      <rect x={370} y={90} width={80} height={40} rx={4} fill="#dbeafe" stroke="#3b82f6" />
      <text x={410} y={115} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">image_emb</text>

      {/* 右:候选 + sim 条 */}
      <text x={470} y={45} fontSize={11} fontWeight={600} fill="var(--ink-primary)">
        候选 prompt + cos sim:
      </text>
      {sims.map((c, k) => {
        const y = 60 + k * 26;
        const barW = (W - 510) * (c.sim / Math.max(0.01, maxSim));
        const isWinner = c.label === ex.trueLabel;
        return (
          <g key={k}>
            <text x={470} y={y + 10} fontSize={10} fill="var(--ink-secondary)">
              "{`a photo of a ${c.label}`}"
            </text>
            <rect
              x={470}
              y={y + 12}
              width={Math.max(2, barW)}
              height={8}
              rx={2}
              fill={isWinner ? "#10b981" : "#dbeafe"}
              opacity={isWinner ? 1 : 0.6}
            />
            <text x={475 + barW} y={y + 19} fontSize={9} fill="var(--ink-muted)">
              {(c.sim * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}

      {/* 结果 */}
      <text x={W / 2} y={H - 30} textAnchor="middle" fontSize={13} fontWeight={700} fill="#10b981">
        ✓ argmax = "{ex.trueLabel}" (真实标签)
      </text>
      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        模型从未见过这些类别 · 只换 prompt 就能换任务 → ImageNet zero-shot 76.2%
      </text>
    </svg>
  );
}
