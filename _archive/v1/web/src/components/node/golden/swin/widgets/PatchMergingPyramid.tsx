import { SWIN_STAGES } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  highlightIdx: number;
}

export function PatchMergingPyramid({ highlightIdx }: Props) {
  const maxSize = 140;
  const startY = 60;
  const gap = 60;

  let cx = 90;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Patch merging pyramid">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Patch Merging — 层级化产出多尺度特征图
      </text>

      {SWIN_STAGES.map((s, i) => {
        const size = maxSize * (s.resolution / 56) ** 0.5 + 20;
        const isHigh = i === highlightIdx || highlightIdx === -1;
        const x = cx;
        cx += size + gap;
        return (
          <g key={i} opacity={isHigh ? 1 : 0.35}>
            <rect x={x} y={startY + (maxSize - size) / 2} width={size} height={size}
                  fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={3} />
            <text x={x + size / 2} y={startY + maxSize / 2 - 4} textAnchor="middle" fontSize={11} fontWeight={700} fill="#831843">
              {s.resolution}×{s.resolution}
            </text>
            <text x={x + size / 2} y={startY + maxSize / 2 + 12} textAnchor="middle" fontSize={9} fill="#831843">
              {s.channels}ch
            </text>

            <text x={x + size / 2} y={startY + maxSize + 24} textAnchor="middle" fontSize={10} fontWeight={600} fill="#374151">{s.name}</text>
            <text x={x + size / 2} y={startY + maxSize + 40} textAnchor="middle" fontSize={9} fill="#6b7280">{s.blocks} blocks</text>

            {i < SWIN_STAGES.length - 1 && (
              <g>
                <line x1={x + size + 4} y1={startY + maxSize / 2} x2={x + size + gap - 4} y2={startY + maxSize / 2}
                      stroke="#f59e0b" strokeWidth={1.8} markerEnd="url(#pm-arr)" />
                <text x={x + size + gap / 2} y={startY + maxSize / 2 - 8} textAnchor="middle" fontSize={8} fill="#92400e">Patch</text>
                <text x={x + size + gap / 2} y={startY + maxSize / 2 + 14} textAnchor="middle" fontSize={8} fill="#92400e">Merging</text>
              </g>
            )}
          </g>
        );
      })}

      <defs>
        <marker id="pm-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b" />
        </marker>
      </defs>

      <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        每 stage:2× 减空间 · 2× 增 channel — 模仿 ResNet 金字塔,4 个尺度可直接接 FPN
      </text>
    </svg>
  );
}
