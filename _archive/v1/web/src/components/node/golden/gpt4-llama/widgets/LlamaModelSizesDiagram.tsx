import { LLAMA_MODELS } from "../lib/data";

const W = 700;
const H = 260;

interface Props {
  selectedIdx: number;
}

export function LlamaModelSizesDiagram({ selectedIdx }: Props) {
  const colW = 150;
  const gap = 20;
  const startX = (W - (LLAMA_MODELS.length * colW + (LLAMA_MODELS.length - 1) * gap)) / 2;
  const maxLayers = 80;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LLaMA-1 四档模型对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        LLaMA-1 四档 — 全部公开架构与训练细节
      </text>

      {LLAMA_MODELS.map((m, i) => {
        const x = startX + i * (colW + gap);
        const isSelected = i === selectedIdx;
        const barH = (m.layers / maxLayers) * 120;
        return (
          <g key={m.name} opacity={isSelected ? 1 : 0.4}>
            <text x={x + colW / 2} y={40} textAnchor="middle" fontSize={11} fontWeight={700} fill={isSelected ? "#1e40af" : "#6b7280"}>{m.name}</text>
            <rect x={x + colW / 2 - 24} y={170 - barH} width={48} height={barH} fill={isSelected ? "#dbeafe" : "#f3f4f6"} stroke={isSelected ? "#3b82f6" : "#9ca3af"} strokeWidth={1.6} rx={3} />
            <text x={x + colW / 2} y={170 - barH - 8} textAnchor="middle" fontSize={10} fontWeight={700} fill={isSelected ? "#1e40af" : "#6b7280"}>{m.layers} 层</text>
            <text x={x + colW / 2} y={188} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">d_model={m.dModel}</text>
            <text x={x + colW / 2} y={202} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">{m.tokensT}T token</text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {LLAMA_MODELS[selectedIdx].note}
      </text>
    </svg>
  );
}
