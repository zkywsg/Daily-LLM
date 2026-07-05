import {
  GENERATOR_LAYERS_DCGAN,
  DISCRIMINATOR_LAYERS_DCGAN,
  GENERATOR_LAYERS_MLP,
  DISCRIMINATOR_LAYERS_MLP,
} from "../lib/data";

const W = 700;
const H = 360;

interface Props {
  mode: "mlp" | "dcgan";
}

export function ArchitectureDiagram({ mode }: Props) {
  const isDcgan = mode === "dcgan";

  const genLayers = isDcgan ? GENERATOR_LAYERS_DCGAN : GENERATOR_LAYERS_MLP;
  const disLayers = isDcgan ? DISCRIMINATOR_LAYERS_DCGAN : DISCRIMINATOR_LAYERS_MLP;

  const colW = 130;
  const gap = 20;
  const startX = 40;
  const rowY1 = 70;
  const rowY2 = 220;

  const maxDim = isDcgan ? 5 : 5;

  const barHeight = (i: number) => {
    // 越靠近图像端(两侧)方块越大,模拟空间尺寸增长/缩小
    const ratio = isDcgan ? (i + 1) / maxDim : 0.6;
    return 18 + ratio * 34;
  };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={isDcgan ? "DCGAN 全卷积 G/D 结构" : "原版 GAN 的 MLP G/D 结构"}
    >
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {isDcgan ? "DCGAN — 全卷积 Generator / Discriminator" : "原版 GAN — 全连接(MLP)Generator / Discriminator"}
      </text>

      <text x={startX} y={rowY1 - 20} fontSize={11} fontWeight={700} fill="#3b82f6">Generator</text>
      <g>
        {genLayers.map((layer, i) => {
          const x = startX + i * (colW + gap);
          const h = barHeight(i);
          const label = "channels" in layer ? `${layer.size}` : `${(layer as { units: number }).units}`;
          const sub = "channels" in layer ? `${layer.channels}ch` : "units";
          return (
            <g key={layer.label}>
              <rect
                x={x}
                y={rowY1 - h / 2}
                width={colW}
                height={h}
                fill="#dbeafe"
                stroke="#3b82f6"
                strokeWidth={1.4}
                rx={4}
              />
              <text x={x + colW / 2} y={rowY1 - h / 2 - 8} textAnchor="middle" fontSize={9} fill="var(--ink-secondary)">
                {layer.label}
              </text>
              <text x={x + colW / 2} y={rowY1 + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
                {label}
              </text>
              <text x={x + colW / 2} y={rowY1 + h / 2 + 14} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">
                {sub}
              </text>
              {i < genLayers.length - 1 && (
                <line
                  x1={x + colW}
                  y1={rowY1}
                  x2={x + colW + gap}
                  y2={rowY1}
                  stroke="var(--border)"
                  strokeWidth={1.5}
                  markerEnd="url(#arrow-dcgan)"
                />
              )}
            </g>
          );
        })}
      </g>

      <text x={startX} y={rowY2 - 20} fontSize={11} fontWeight={700} fill="#ec4899">Discriminator</text>
      <g>
        {disLayers.map((layer, i) => {
          const x = startX + i * (colW + gap);
          const h = barHeight(disLayers.length - 1 - i);
          const label = "channels" in layer ? `${layer.size}` : `${(layer as { units: number }).units}`;
          const sub = "channels" in layer ? `${layer.channels}ch` : "units";
          return (
            <g key={layer.label}>
              <rect
                x={x}
                y={rowY2 - h / 2}
                width={colW}
                height={h}
                fill="#fce7f3"
                stroke="#ec4899"
                strokeWidth={1.4}
                rx={4}
              />
              <text x={x + colW / 2} y={rowY2 - h / 2 - 8} textAnchor="middle" fontSize={9} fill="var(--ink-secondary)">
                {layer.label}
              </text>
              <text x={x + colW / 2} y={rowY2 + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
                {label}
              </text>
              <text x={x + colW / 2} y={rowY2 + h / 2 + 14} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">
                {sub}
              </text>
              {i < disLayers.length - 1 && (
                <line
                  x1={x + colW}
                  y1={rowY2}
                  x2={x + colW + gap}
                  y2={rowY2}
                  stroke="var(--border)"
                  strokeWidth={1.5}
                  markerEnd="url(#arrow-dcgan)"
                />
              )}
            </g>
          );
        })}
      </g>

      <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {isDcgan
          ? "strided conv / transposed conv 全程在空间网格上操作,无 fc reshape"
          : "fc 层把 z reshape 成图像 / 把图像 flatten 成 scalar,破坏空间结构"}
      </text>

      <defs>
        <marker id="arrow-dcgan" markerWidth={8} markerHeight={8} refX={6} refY={4} orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="var(--border)" />
        </marker>
      </defs>
    </svg>
  );
}
