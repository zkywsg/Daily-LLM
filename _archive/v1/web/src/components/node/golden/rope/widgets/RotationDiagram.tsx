import { rotate2D } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  m: number; // 位置 m
  theta: number; // 频率角度(度)
}

export function RotationDiagram({ m, theta }: Props) {
  const cx1 = 180, cy1 = 180, r = 120;
  const cx2 = 520, cy2 = 180;

  const thetaRad = (theta * Math.PI) / 180;
  const angleQ = m * thetaRad;

  // base vector q
  const qx0 = r, qy0 = 0;
  const [qx, qy] = rotate2D(qx0, qy0, angleQ);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="RoPE 2D rotation diagram">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        位置 m 直接对应几何旋转角度 — q 被旋转 m·θ
      </text>

      {/* left: single vector rotation */}
      <circle cx={cx1} cy={cy1} r={r} fill="none" stroke="#e5e7eb" strokeWidth={1} />
      <line x1={cx1 - r} y1={cy1} x2={cx1 + r} y2={cy1} stroke="#e5e7eb" strokeWidth={1} />
      <line x1={cx1} y1={cy1 - r} x2={cx1} y2={cy1 + r} stroke="#e5e7eb" strokeWidth={1} />

      {/* original q (dashed) */}
      <line x1={cx1} y1={cy1} x2={cx1 + qx0} y2={cy1 - qy0} stroke="#9ca3af" strokeWidth={1.5} strokeDasharray="4 3" />
      <text x={cx1 + qx0 + 8} y={cy1 - qy0 + 4} fontSize={10} fill="#9ca3af">q (m=0)</text>

      {/* rotated q */}
      <line x1={cx1} y1={cy1} x2={cx1 + qx} y2={cy1 - qy} stroke="#ec4899" strokeWidth={2.5} markerEnd="url(#rot-arr)" />
      <text x={cx1 + qx + 10} y={cy1 - qy} fontSize={11} fontWeight={700} fill="#ec4899">R_m·q</text>

      <defs>
        <marker id="rot-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ec4899" />
        </marker>
      </defs>

      {/* angle arc */}
      <path d={`M ${cx1 + 40} ${cy1} A 40 40 0 ${angleQ > Math.PI ? 1 : 0} 0 ${cx1 + 40 * Math.cos(angleQ)} ${cy1 - 40 * Math.sin(angleQ)}`}
            fill="none" stroke="#f59e0b" strokeWidth={1.5} />
      <text x={cx1 + 55} y={cy1 - 10} fontSize={10} fontWeight={600} fill="#92400e">m·θ = {(angleQ * 180 / Math.PI).toFixed(0)}°</text>

      <text x={cx1} y={cy1 + r + 30} textAnchor="middle" fontSize={11} fill="#374151">m = {m}</text>

      {/* right: group law illustration */}
      <text x={cx2} y={60} textAnchor="middle" fontSize={12} fontWeight={700} fill="#374151">
        群论性质
      </text>
      <rect x={cx2 - 130} y={80} width={260} height={70} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.2} opacity={0.6} />
      <text x={cx2} y={108} textAnchor="middle" fontSize={12} fontFamily="ui-monospace, monospace" fill="#1f2937">
        R_m^T R_n = R_{"{n-m}"}
      </text>
      <text x={cx2} y={130} textAnchor="middle" fontSize={11} fontFamily="ui-monospace, monospace" fill="#1f2937">
        ⟨R_m q, R_n k⟩ = q^T R_{"{n-m}"} k
      </text>

      <text x={cx2} y={180} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">
        位置 m、n 消失,只剩 n − m
      </text>
      <text x={cx2} y={200} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#9ca3af">
        几何结构保证,不需要模型学习
      </text>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        拖动 m 看向量转多少度 · 拖动 θ 看旋转快慢
      </text>
    </svg>
  );
}
