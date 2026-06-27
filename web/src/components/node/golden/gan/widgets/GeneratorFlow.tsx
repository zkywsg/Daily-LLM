const W = 700;
const H = 220;

// G 流程图:z ~ 𝒩(0, I) → G(z) → fake image。
// 强调"无显式概率密度,只学采样器"这一点 —— 跟 VAE/Normalizing Flow 区分。

function Box({
  x, y, w, h, fill, stroke, label, sub,
}: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={10} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  const id = `arr-${x1}-${y1}-${x2}-${y2}`;
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1.5} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function GeneratorFlow() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GAN Generator flow">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        Generator:z ~ 𝒩(0, I) → MLP / DeConv → 假图 x̂
      </text>

      {/* z 噪声 */}
      <Box x={30} y={60} w={120} h={60} fill="#fef3c7" stroke="#f59e0b" label="z ∈ ℝ¹⁰⁰" sub="标准高斯噪声" />
      <Arrow x1={150} y1={90} x2={220} y2={90} />

      {/* G */}
      <Box x={220} y={60} w={200} h={60} fill="#fce7f3" stroke="#ec4899" label="Generator G_θ" sub="MLP / DeConv stack" />
      <Arrow x1={420} y1={90} x2={490} y2={90} />

      {/* x̂ fake */}
      <Box x={490} y={60} w={180} h={60} fill="#ecfdf5" stroke="#10b981" label="x̂ = G(z)" sub="假图(看起来像真图)" />

      <text x={W / 2} y={170} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-secondary)">
        关键:G 不学 p_data 的显式密度,只学一个采样器
      </text>
      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        VAE 是 p(x|z) 显式密度 · Normalizing Flow 是可逆变换 · GAN 是 implicit 采样器
      </text>
    </svg>
  );
}
