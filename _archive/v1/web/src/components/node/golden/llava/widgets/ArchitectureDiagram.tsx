const W = 700;
const H = 220;

export function ArchitectureDiagram() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LLaVA 极简架构:CLIP + projection + LLM">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        CLIP 当眼睛 + LLaMA 当脑子,只训一个 projection 桥
      </text>

      <rect x={30} y={60} width={150} height={60} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} rx={6} strokeDasharray="4 3" />
      <text x={105} y={86} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">❄ CLIP ViT-L/14</text>
      <text x={105} y={104} textAnchor="middle" fontSize={9} fill="#1e40af">冻结,256 patch tokens</text>

      <line x1={180} y1={90} x2={230} y2={90} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-llava)" />

      <rect x={230} y={70} width={140} height={40} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} rx={6} />
      <text x={300} y={94} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">Projection(可训练)</text>

      <line x1={370} y1={90} x2={420} y2={90} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-llava)" />

      <rect x={420} y={60} width={160} height={60} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} rx={6} strokeDasharray="4 3" />
      <text x={500} y={86} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">LLaMA / Vicuna</text>
      <text x={500} y={104} textAnchor="middle" fontSize={9} fill="#1e40af">Stage1 冻结,Stage2 全微调</text>

      <text x={W / 2} y={155} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        768 维 CLIP 特征 → linear projection → 4096/5120 维 LLM 空间,直接拼接到文本 token 前面
      </text>
      <text x={W / 2} y={175} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        没有 Q-Former,没有 Perceiver Resampler,没有 cross-attention — LLM 架构一行不改
      </text>

      <defs>
        <marker id="arrow-llava" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
