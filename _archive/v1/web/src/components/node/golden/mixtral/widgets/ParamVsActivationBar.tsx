import { PARAM_COMPARE } from "../lib/data";

const W = 700;
const H = 220;

// Mixtral 8x7B:
//   总参数 46.7B(8 个 expert + 共享 attention)
//   每 token 激活 12.9B(2 个 expert + 共享)
//   等价容量约 Llama-2-70B
// 用三条横向条对比,标注比例。

function fmtB(n: number): string {
  return `${(n / 1e9).toFixed(1)}B`;
}

export function ParamVsActivationBar() {
  const max = PARAM_COMPARE.denseEquivalent;
  const px = (v: number) => (v / max) * (W - 280);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Mixtral params vs active params vs dense equivalent">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        总参数 vs 激活参数 vs 等价 dense 容量
      </text>

      {/* 总参数 */}
      <text x={20} y={62} fontSize={11} fontWeight={600} fill="var(--ink-primary)">总参数</text>
      <text x={20} y={76} fontSize={9} fill="var(--ink-muted)">8 expert × 5.6B + 共享</text>
      <rect x={170} y={50} width={Math.max(2, px(PARAM_COMPARE.totalParams))} height={24} rx={3} fill="#9ca3af" opacity={0.85} />
      <text x={180 + px(PARAM_COMPARE.totalParams)} y={66} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        {fmtB(PARAM_COMPARE.totalParams)}
      </text>

      {/* 激活参数 */}
      <text x={20} y={112} fontSize={11} fontWeight={600} fill="var(--ink-primary)">激活参数</text>
      <text x={20} y={126} fontSize={9} fill="var(--ink-muted)">每 token 2 expert + 共享</text>
      <rect x={170} y={100} width={Math.max(2, px(PARAM_COMPARE.activeParams))} height={24} rx={3} fill="#ec4899" opacity={0.85} />
      <text x={180 + px(PARAM_COMPARE.activeParams)} y={116} fontSize={11} fontWeight={700} fill="#831843">
        {fmtB(PARAM_COMPARE.activeParams)} · 仅 {(PARAM_COMPARE.activeParams / PARAM_COMPARE.totalParams * 100).toFixed(0)}%
      </text>

      {/* 等价 dense */}
      <text x={20} y={162} fontSize={11} fontWeight={600} fill="var(--ink-primary)">等价 dense 容量</text>
      <text x={20} y={176} fontSize={9} fill="var(--ink-muted)">质量大致媲美</text>
      <rect x={170} y={150} width={Math.max(2, px(PARAM_COMPARE.denseEquivalent))} height={24} rx={3} fill="#10b981" opacity={0.85} />
      <text x={180 + px(PARAM_COMPARE.denseEquivalent)} y={166} fontSize={11} fontWeight={700} fill="#065f46">
        {fmtB(PARAM_COMPARE.denseEquivalent)} (Llama-2-70B 类)
      </text>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        47B 总参数 = Llama-13B 算力 = 70B 质量 · 这是 MoE 的核心买卖
      </text>
    </svg>
  );
}
