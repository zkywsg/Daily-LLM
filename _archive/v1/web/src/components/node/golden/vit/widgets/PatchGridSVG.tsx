import { TOY_IMAGE } from "../lib/data";

interface Props {
  patchSize: number;
  hoverIdx: number | null;
  onHoverIdxChange: (i: number | null) => void;
}

const W = 700;
const H = 320;

// 14×14 玩具 image 切成 N×N 个 patch,viewer hover 单个 patch 看高亮 + 索引。
// 左边显示原 image,右边把同样的 patches 展开成一行(token sequence)。
export function PatchGridSVG({ patchSize, hoverIdx, onHoverIdxChange }: Props) {
  const N = TOY_IMAGE.length;
  const numAxis = N / patchSize;
  const numPatches = numAxis * numAxis;

  // 左侧:原 image
  const leftSize = 240;
  const leftPad = 30;
  const pixSize = leftSize / N;
  const patchPixW = patchSize * pixSize;

  // 右侧:token 序列
  const rightStartX = leftPad + leftSize + 60;
  const rightW = W - rightStartX - 30;
  const tokSize = Math.min(rightW / numPatches - 4, 30);

  const cellFill = (v: number) => `hsl(0, 0%, ${v * 100}%)`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Image split into ${numPatches} patches`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        Patch 化:14×14 image → {numPatches} 个 {patchSize}×{patchSize} patch token
      </text>

      {/* 左侧 image */}
      <g transform={`translate(${leftPad}, 50)`}>
        <text x={leftSize / 2} y={-8} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">输入 image (14×14)</text>
        {TOY_IMAGE.map((row, r) =>
          row.map((v, c) => (
            <rect
              key={`p-${r}-${c}`}
              x={c * pixSize}
              y={r * pixSize}
              width={pixSize}
              height={pixSize}
              fill={cellFill(v)}
            />
          ))
        )}
        {/* patch 网格 + hover 高亮 */}
        {Array.from({ length: numAxis * numAxis }, (_, i) => {
          const pr = Math.floor(i / numAxis);
          const pc = i % numAxis;
          const x = pc * patchPixW;
          const y = pr * patchPixW;
          const isHover = hoverIdx === i;
          return (
            <g
              key={`grid-${i}`}
              onMouseEnter={() => onHoverIdxChange(i)}
              onMouseLeave={() => onHoverIdxChange(null)}
              style={{ cursor: "pointer" }}
            >
              <rect
                x={x}
                y={y}
                width={patchPixW}
                height={patchPixW}
                fill={isHover ? "rgba(236, 72, 153, 0.3)" : "transparent"}
                stroke={isHover ? "#ec4899" : "#d1d5db"}
                strokeWidth={isHover ? 2 : 0.6}
              />
              {isHover && (
                <text x={x + patchPixW / 2} y={y + patchPixW / 2 + 4} textAnchor="middle" fontSize={11} fontWeight={700} fill="#ec4899">
                  {i}
                </text>
              )}
            </g>
          );
        })}
      </g>

      {/* 箭头 */}
      <text x={leftPad + leftSize + 30} y={H / 2} textAnchor="middle" fontSize={24} fill="var(--ink-muted)">
        →
      </text>
      <text x={leftPad + leftSize + 30} y={H / 2 + 16} textAnchor="middle" fontSize={9} fontStyle="italic" fill="var(--ink-muted)">
        flatten
      </text>

      {/* 右侧:token 序列 */}
      <g transform={`translate(${rightStartX}, 50)`}>
        <text x={rightW / 2} y={-8} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
          token 序列(展开成 1D)
        </text>
        {Array.from({ length: numPatches }, (_, i) => {
          const isHover = hoverIdx === i;
          const col = i % Math.min(numPatches, 8);
          const row = Math.floor(i / 8);
          return (
            <g key={`tok-${i}`} transform={`translate(${col * (tokSize + 4)}, ${row * (tokSize + 4)})`}>
              <rect
                width={tokSize}
                height={tokSize}
                rx={3}
                fill={isHover ? "#fce7f3" : "#fef3c7"}
                stroke={isHover ? "#ec4899" : "#f59e0b"}
                strokeWidth={isHover ? 2 : 1}
              />
              <text x={tokSize / 2} y={tokSize / 2 + 3} textAnchor="middle" fontSize={9} fontWeight={isHover ? 700 : 500} fill="#1f2937">
                t{i}
              </text>
            </g>
          );
        })}
      </g>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        每个 patch 经 flatten + linear projection → 1 个 token 向量,送入标准 Transformer
      </text>
    </svg>
  );
}
