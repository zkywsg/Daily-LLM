import { VGG16_BLOCKS } from "../lib/data";

const W = 700;
const H = 260;

// VGG-16 的 5 个 conv block:通道沿深度翻倍,空间沿深度减半(224 -> 7)
export function VggBlockDiagram() {
  const n = VGG16_BLOCKS.length;
  const colW = 110;
  const gap = 16;
  const startX = 40;
  const baseY = 190;
  const maxChannels = Math.max(...VGG16_BLOCKS.map((b) => b.channels));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="VGG-16 五个 conv block 的通道翻倍与空间减半"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        VGG-16 — 同 block 内通道不变,跨 block 通道翻倍 + 空间减半
      </text>

      {VGG16_BLOCKS.map((b, i) => {
        const x = startX + i * (colW + gap);
        const barH = (b.channels / maxChannels) * 120;
        return (
          <g key={b.block}>
            <rect
              x={x}
              y={baseY - barH}
              width={colW}
              height={barH}
              fill="#fce7f3"
              stroke="#ec4899"
              strokeWidth={1.6}
              rx={3}
            />
            <text x={x + colW / 2} y={baseY - barH - 22} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              block {b.block}
            </text>
            <text x={x + colW / 2} y={baseY - barH - 8} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
              {b.channels} 通道 × {b.convRepeats} 层 3×3
            </text>
            <text x={x + colW / 2} y={baseY + 18} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
              → {b.spatialAfterPool}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        MaxPool/2 让空间 224→112→56→28→14→7,同时通道 64→128→256→512→512 翻倍再封顶
      </text>
    </svg>
  );
}
