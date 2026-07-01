import { DEMO_SRC } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  cDim: number; // 展示压缩维度大小的直觉(数值越小瓶颈越明显)
}

function Arrow({ x1, y1, x2, y2, color, id }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string }) {
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function EncoderCompressPipeline({ cDim }: Props) {
  const cRadius = 20 + cDim * 0.15; // 300..1000 → 65..170 半径映射范围裁剪
  const clampedR = Math.min(Math.max(cRadius, 25), 55);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Encoder compression to fixed vector c">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Encoder — 任意长输入压成固定向量 c(维度 = {cDim})
      </text>

      {/* tokens */}
      {DEMO_SRC.map((tok, i) => (
        <g key={i}>
          <rect x={40 + i * 90} y={70} width={70} height={30} rx={4} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} />
          <text x={75 + i * 90} y={90} textAnchor="middle" fontSize={12} fontWeight={600} fill="#1f2937">{tok}</text>
          <line x1={75 + i * 90} y1={100} x2={75 + i * 90} y2={130} stroke="#3b82f6" strokeWidth={1.2} />
          <rect x={45 + i * 90} y={130} width={60} height={26} rx={3} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.2} />
          <text x={75 + i * 90} y={148} textAnchor="middle" fontSize={10} fill="#831843">h_{i + 1}</text>
          {i < DEMO_SRC.length - 1 && (
            <Arrow x1={105 + i * 90} y1={143} x2={135 + i * 90} y2={143} color="#ec4899" id={`enc-a-${i}`} />
          )}
        </g>
      ))}

      {/* 压缩箭头到 c */}
      <Arrow x1={75 + (DEMO_SRC.length - 1) * 90} y1={156} x2={W / 2} y2={210} color="#f59e0b" id="enc-comp" />

      {/* context vector c */}
      <circle cx={W / 2} cy={230} r={clampedR} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} />
      <text x={W / 2} y={225} textAnchor="middle" fontSize={13} fontWeight={700} fill="#92400e">c</text>
      <text x={W / 2} y={240} textAnchor="middle" fontSize={9} fill="#92400e">{cDim} 维</text>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        c = h_T(encoder 最后一时刻隐状态)· 所有词序/句法/语义都压进这一个固定维度向量
      </text>
    </svg>
  );
}
