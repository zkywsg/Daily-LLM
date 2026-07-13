import { GOOGLENET_STAGES } from "../lib/data";

const W = 700;
const H = 220;

const KIND_COLOR: Record<string, { bg: string; stroke: string }> = {
  stem: { bg: "#f3f4f6", stroke: "#9ca3af" },
  inception: { bg: "#fce7f3", stroke: "#ec4899" },
  pool: { bg: "#dbeafe", stroke: "#3b82f6" },
  aux: { bg: "#fef3c7", stroke: "#f59e0b" },
  output: { bg: "#ecfdf5", stroke: "#10b981" },
};

export function GoogLeNetPipelineDiagram() {
  const boxW = 46;
  const gap = 6;
  const totalW = GOOGLENET_STAGES.length * (boxW + gap) - gap;
  const startX = (W - totalW) / 2;
  const y = 90;
  const boxH = 44;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="GoogLeNet 整网:stem + 9 个 Inception block + GAP + 单层 FC,2 个 aux head 挂在 4a/4d"
    >
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        GoogLeNet 整网 — stem + 9 个 Inception block + GAP + FC
      </text>

      {GOOGLENET_STAGES.map((stage, i) => {
        const x = startX + i * (boxW + gap);
        const color = KIND_COLOR[stage.kind];
        return (
          <g key={`${stage.label}-${i}`}>
            <rect x={x} y={y} width={boxW} height={boxH} rx={4} fill={color.bg} stroke={color.stroke} strokeWidth={1.4} />
            <text x={x + boxW / 2} y={y + boxH / 2 + 4} textAnchor="middle" fontSize={9} fontWeight={700} fill="var(--ink-primary)">
              {stage.label}
            </text>
            {stage.note && (
              <>
                <line x1={x + boxW / 2} y1={y} x2={x + boxW / 2} y2={y - 14} stroke="#f59e0b" strokeWidth={1.2} />
                <foreignObject x={x - 30} y={y - 74} width={boxW + 60} height={60}>
                  <div
                    style={{
                      fontSize: 8.5,
                      lineHeight: 1.35,
                      color: "#f59e0b",
                      fontWeight: 700,
                      textAlign: "center",
                      fontFamily: "system-ui",
                    }}
                  >
                    {stage.note}
                  </div>
                </foreignObject>
              </>
            )}
          </g>
        );
      })}

      <text x={W / 2} y={y + boxH + 30} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        推理时 aux head 丢弃,只保留主干 + GAP + 单层 FC 输出 1000 类
      </text>
    </svg>
  );
}
