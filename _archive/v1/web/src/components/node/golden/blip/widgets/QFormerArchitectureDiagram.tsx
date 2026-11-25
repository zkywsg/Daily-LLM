import { QFORMER_PARAMS } from "../lib/data";

const W = 700;
const H = 260;

export function QFormerArchitectureDiagram() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BLIP-2 Q-Former 桥接架构">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        冻结视觉编码器 + 冻结 LLM,只训练中间的 Q-Former
      </text>

      <g transform="translate(30, 50)">
        <rect x={0} y={0} width={150} height={60} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} rx={6} strokeDasharray="4 3" />
        <text x={75} y={26} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">❄ 冻结 ViT-G/14</text>
        <text x={75} y={44} textAnchor="middle" fontSize={9} fill="#1e40af">{QFORMER_PARAMS[0].paramsM}M 参数</text>

        <line x1={150} y1={30} x2={200} y2={30} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-qf)" />

        <rect x={200} y={-10} width={160} height={80} fill="#ecfdf5" stroke="#10b981" strokeWidth={2} rx={6} />
        <text x={280} y={10} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">Q-Former(可训练)</text>
        <text x={280} y={28} textAnchor="middle" fontSize={9} fill="#065f46">32 个 Query tokens</text>
        <text x={280} y={44} textAnchor="middle" fontSize={9} fill="#065f46">12 层 cross-attention</text>
        <text x={280} y={60} textAnchor="middle" fontSize={9} fontWeight={700} fill="#065f46">{QFORMER_PARAMS[1].paramsM}M 参数(1.5%)</text>

        <line x1={360} y1={30} x2={410} y2={30} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-qf)" />

        <rect x={410} y={0} width={160} height={60} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} rx={6} strokeDasharray="4 3" />
        <text x={490} y={26} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">❄ 冻结 Flan-T5 XXL</text>
        <text x={490} y={44} textAnchor="middle" fontSize={9} fill="#1e40af">{(QFORMER_PARAMS[2].paramsM / 1000).toFixed(0)}B 参数</text>
      </g>

      <text x={W / 2} y={170} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        总参数 12.1B,可训练部分只有 Q-Former 的 188M(1.5%)
      </text>
      <text x={W / 2} y={190} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        视觉和 LLM 都不动,32 个 Q tokens 通过 cross-attention "查询"视觉特征,提炼出图像-语言对齐的向量
      </text>

      <defs>
        <marker id="arrow-qf" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
