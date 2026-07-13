import { VGG_DEPTH_PROGRESSION } from "../lib/data";

const W = 700;
const H = 220;

// VGG-11 -> VGG-19:conv 层数逐代增加,fc 层数固定 3 层不变
export function VggDepthProgressionChart() {
  const n = VGG_DEPTH_PROGRESSION.length;
  const colW = 130;
  const gap = 24;
  const startX = 60;
  const baseY = 180;
  const maxTotal = Math.max(...VGG_DEPTH_PROGRESSION.map((s) => s.totalLayers));
  const scale = 130 / maxTotal;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="VGG-11 到 VGG-19 的深度演化,conv 层数递增、fc 层数固定"
    >
      <text x={W / 2} y={18} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        VGG-11 → VGG-19 — 深度作为单一变量递增
      </text>

      {VGG_DEPTH_PROGRESSION.map((step, i) => {
        const x = startX + i * (colW + gap);
        const convH = step.convLayers * scale;
        const fcH = step.fcLayers * scale;
        return (
          <g key={step.name}>
            <rect x={x} y={baseY - convH - fcH} width={colW} height={fcH} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} />
            <rect x={x} y={baseY - convH} width={colW} height={convH} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} />
            <text x={x + colW / 2} y={baseY - convH - fcH - 10} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {step.name}
            </text>
            <text x={x + colW / 2} y={baseY + 18} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
              conv {step.convLayers} + fc {step.fcLayers} = {step.totalLayers} 层
            </text>
          </g>
        );
      })}

      <g transform={`translate(${W - 170}, ${H - 30})`}>
        <rect x={0} y={-10} width={10} height={10} fill="#fce7f3" stroke="#ec4899" strokeWidth={1} />
        <text x={16} y={-1} fontSize={9} fill="var(--ink-muted)">conv(3×3 砖头堆叠)</text>
        <rect x={0} y={8} width={10} height={10} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1} />
        <text x={16} y={17} fontSize={9} fill="var(--ink-muted)">fc(固定 3 层)</text>
      </g>
    </svg>
  );
}
