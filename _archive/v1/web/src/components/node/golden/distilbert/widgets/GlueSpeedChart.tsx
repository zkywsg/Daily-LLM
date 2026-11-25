import { GLUE_TABLE, SPEED_TABLE } from "../lib/data";

const W = 700;
const H = 460;

export function GlueSpeedChart() {
  const PAD_L = 90;
  const rowH = 26;
  const maxScore = 100;
  const plotW = 380;

  const glueTop = 50;
  const speedTop = glueTop + GLUE_TABLE.length * rowH + 50;
  const maxLatency = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GLUE 任务成绩与推理速度对比,BERT-base vs DistilBERT">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        GLUE 9 任务成绩(节选) — DistilBERT 平均保留 97% 性能
      </text>

      {GLUE_TABLE.map((row, i) => {
        const y = glueTop + i * rowH;
        const isAvg = row.task.includes("平均");
        const bertW = (row.bert / maxScore) * plotW;
        const distilW = (row.distilbert / maxScore) * plotW;
        return (
          <g key={row.task}>
            <text x={PAD_L - 8} y={y + 14} textAnchor="end" fontSize={9} fontWeight={isAvg ? 700 : 400} fill="var(--ink-primary)">
              {row.task}
            </text>
            <rect x={PAD_L} y={y} width={bertW} height={9} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1} />
            <rect x={PAD_L} y={y + 10} width={distilW} height={9} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1} />
            <text x={PAD_L + Math.max(bertW, distilW) + 8} y={y + 14} fontSize={9} fill="var(--ink-secondary)">
              {row.bert} → {row.distilbert}
            </text>
          </g>
        );
      })}

      <g transform={`translate(${PAD_L}, ${glueTop - 20})`}>
        <rect x={0} y={-9} width={10} height={10} fill="#dbeafe" stroke="#3b82f6" />
        <text x={14} y={0} fontSize={9} fill="var(--ink-secondary)">BERT-base</text>
        <rect x={90} y={-9} width={10} height={10} fill="#fef3c7" stroke="#f59e0b" />
        <text x={104} y={0} fontSize={9} fill="var(--ink-secondary)">DistilBERT</text>
      </g>

      <text x={W / 2} y={speedTop - 20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        推理速度(V100,batch 1,序列 128)
      </text>

      {SPEED_TABLE.map((row, i) => {
        const y = speedTop + i * 46;
        const w = (row.latencyMs / maxLatency) * plotW;
        const color = row.model === "DistilBERT" ? "#10b981" : row.model === "BERT-base" ? "#3b82f6" : "#9ca3af";
        const bg = row.model === "DistilBERT" ? "#ecfdf5" : row.model === "BERT-base" ? "#dbeafe" : "#f3f4f6";
        return (
          <g key={row.model}>
            <text x={PAD_L - 8} y={y + 14} textAnchor="end" fontSize={10} fontWeight={700} fill={color}>
              {row.model}
            </text>
            <rect x={PAD_L} y={y} width={Math.max(w, 4)} height={20} fill={bg} stroke={color} strokeWidth={1.4} rx={3} />
            <text x={PAD_L + Math.max(w, 4) + 8} y={y + 14} fontSize={10} fontWeight={700} fill={color}>
              {row.latencyMs}ms · {row.paramsM}M 参数 · {row.tps}+ TPS
            </text>
          </g>
        );
      })}
    </svg>
  );
}
