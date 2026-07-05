import { DEMO_TOKENS, staticMaskPositions, dynamicMaskPositions } from "../lib/data";

const W = 700;
const H = 260;

interface Props {
  mode: "static" | "dynamic";
}

const EPOCHS = [0, 1, 2];

export function DynamicMaskingDiagram({ mode }: Props) {
  const tokenW = 52;
  const startX = (W - DEMO_TOKENS.length * tokenW) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="静态 vs 动态 masking 演示">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "static" ? "静态 Masking(BERT)— 同一组位置重复出现" : "动态 Masking(RoBERTa)— 每个 epoch 重新选位置"}
      </text>

      {EPOCHS.map((epoch) => {
        const masks = mode === "static" ? staticMaskPositions(epoch) : dynamicMaskPositions(epoch);
        const y = 50 + epoch * 62;
        return (
          <g key={epoch}>
            <text x={startX - 12} y={y + 20} textAnchor="end" fontSize={10} fontWeight={700} fill="#6b7280">epoch {epoch + 1}</text>
            {DEMO_TOKENS.map((tok, i) => {
              const x = startX + i * tokenW;
              const masked = masks[i];
              return (
                <g key={i}>
                  <rect x={x} y={y} width={tokenW - 6} height={26} fill={masked ? "#fce7f3" : "var(--bg-surface)"} stroke={masked ? "#ec4899" : "var(--border)"} strokeWidth={masked ? 1.6 : 1} rx={3} />
                  <text x={x + (tokenW - 6) / 2} y={y + 17} textAnchor="middle" fontSize={9} fill={masked ? "#be185d" : "var(--ink-secondary)"} fontWeight={masked ? 700 : 400}>
                    {masked ? "[MASK]" : tok}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {mode === "static"
          ? "同一批被 mask 的位置在每个 epoch 都相同 — 模型反复看到同样的空缺,数据多样性受限"
          : "每个 epoch 重新随机选 15% 位置 mask — 同一句子在不同 epoch 呈现不同的预测任务"}
      </text>
    </svg>
  );
}
