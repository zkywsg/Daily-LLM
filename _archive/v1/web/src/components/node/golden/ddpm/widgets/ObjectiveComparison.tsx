const W = 700;
const H = 300;

// 把 L_vlb (variational lower bound) 和 L_simple 的对比画成两栏:
// L_vlb 是逐项 KL 加权和(数学严谨但梯度抖),
// L_simple 把所有 t 的权重都拍平到 1(直接 MSE on ε)。
// 用 box+arrow 把化简过程拍成"DDPM 论文式的卡通"。
// 配色:input 黄、compute 粉、output 绿、data 蓝。

const COL = {
  vlb: { fill: "#dbeafe", stroke: "#3b82f6" },
  step: { fill: "#fef3c7", stroke: "#f59e0b" },
  simple: { fill: "#ecfdf5", stroke: "#10b981" },
};

function Box({
  x,
  y,
  w,
  h,
  fill,
  stroke,
  label,
  sub,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  stroke: string;
  label: string;
  sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="#1f2937">
        {label}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={10} fill="#6b7280">
          {sub}
        </text>
      )}
    </g>
  );
}

export function ObjectiveComparison() {
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="L_vlb vs L_simple"
    >
      {/* 左栏:L_vlb */}
      <text x={170} y={22} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        L_vlb (variational lower bound)
      </text>
      <Box x={20} y={40} w={130} h={48} {...COL.vlb} label="D_KL(q ‖ p_T)" sub="L_T(无参数)" />
      <Box x={20} y={100} w={130} h={48} {...COL.vlb} label="Σ_t L_t" sub="t∈[1,T-1] KL" />
      <Box x={20} y={160} w={130} h={48} {...COL.vlb} label="−log p_θ(x_0 | x_1)" sub="L_0 重建" />

      <text x={170} y={240} textAnchor="middle" fontSize={11} fontStyle="italic" fill="#dc2626">
        理论严谨,但每步权重不同 →
      </text>
      <text x={170} y={256} textAnchor="middle" fontSize={11} fontStyle="italic" fill="#dc2626">
        梯度方差大、收敛慢
      </text>

      {/* 箭头 → 化简 */}
      <line x1={310} y1={150} x2={400} y2={150} stroke="#9ca3af" strokeWidth={2} markerEnd="url(#obj-arr)" />
      <text x={355} y={140} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-secondary)">
        ε 重参数化
      </text>
      <text x={355} y={170} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-secondary)">
        + 拍平 t 权重
      </text>
      <defs>
        <marker id="obj-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>

      {/* 右栏:L_simple */}
      <text x={540} y={22} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        L_simple (Ho et al. 2020)
      </text>
      <Box
        x={420}
        y={70}
        w={240}
        h={80}
        {...COL.simple}
        label="E_{t, x_0, ε} ‖ ε − ε_θ(x_t, t) ‖²"
        sub="均匀采样 t,只算 MSE on ε"
      />

      <text x={540} y={180} textAnchor="middle" fontSize={11} fontStyle="italic" fill="#10b981">
        实测训练更稳、生成质量更好
      </text>
      <text x={540} y={196} textAnchor="middle" fontSize={11} fontStyle="italic" fill="#10b981">
        虽然只是 ELBO 的"加权变体"
      </text>

      {/* 底部注脚 */}
      <text x={W / 2} y={290} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        Ho 2020 的关键洞见:严格优化下界不如均匀加权简单目标
      </text>
    </svg>
  );
}
