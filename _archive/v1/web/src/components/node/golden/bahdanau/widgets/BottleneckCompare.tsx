const W = 700;
const H = 380;

interface Props {
  highlight: "seq2seq" | "bahdanau" | "both";
}

const SRC = ["I", "love", "cats"];
const TGT = ["我", "爱", "猫"];

function Tile({ x, y, w, h, fill, stroke, label, sub, opacity = 1 }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; opacity?: number;
}) {
  return (
    <g opacity={opacity}>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
    </g>
  );
}

export function BottleneckCompare({ highlight }: Props) {
  const dimSeq = highlight === "bahdanau" ? 0.35 : 1;
  const dimBah = highlight === "seq2seq" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Seq2Seq fixed c vs Bahdanau dynamic context">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Seq2Seq 固定 c vs Bahdanau 动态 c_t — 信息瓶颈如何被解开
      </text>

      {/* === Seq2Seq 上半 === */}
      <g opacity={dimSeq}>
        <text x={20} y={50} fontSize={12} fontWeight={700} fill="#831843">Seq2Seq · 整句压成 1 个 c</text>
        {/* encoder tiles */}
        {SRC.map((w, i) => (
          <g key={i}>
            <Tile x={40 + i * 70} y={62} w={56} h={28} fill="#dbeafe" stroke="#3b82f6" label={w} />
            <line x1={68 + i * 70} y1={90} x2={68 + i * 70} y2={106} stroke="#3b82f6" strokeWidth={1.4} />
            <Tile x={40 + i * 70} y={106} w={56} h={26} fill="#fce7f3" stroke="#ec4899" label={`h${i + 1}`} />
            {i < SRC.length - 1 && (
              <line x1={96 + i * 70} y1={119} x2={110 + i * 70} y2={119} stroke="#ec4899" strokeWidth={1.4} />
            )}
          </g>
        ))}
        {/* 漏斗压缩成 c */}
        <line x1={250} y1={119} x2={310} y2={119} stroke="#ec4899" strokeWidth={1.4} />
        <Tile x={310} y={102} w={70} h={36} fill="#fef3c7" stroke="#f59e0b" label="c" sub="单一 d 维" />
        <text x={345} y={155} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#831843">瓶颈!</text>

        {/* decoder 每一步都连同一个 c */}
        {TGT.map((w, i) => (
          <g key={i}>
            <line x1={380} y1={120} x2={410 + i * 80} y2={88} stroke="#f59e0b" strokeWidth={1} strokeDasharray="3 3" />
            <Tile x={400 + i * 80} y={70} w={50} h={28} fill="#ecfdf5" stroke="#10b981" label={w} />
          </g>
        ))}
        <text x={500} y={42} fontSize={10} fontStyle="italic" fill="#6b7280">译"我"/"爱"/"猫" 都只能看同一个 c</text>
      </g>

      {/* 分隔 */}
      <line x1={20} y1={180} x2={W - 20} y2={180} stroke="#e5e7eb" />

      {/* === Bahdanau 下半 === */}
      <g opacity={dimBah}>
        <text x={20} y={205} fontSize={12} fontWeight={700} fill="#065f46">Bahdanau · 保留所有 h_i + 动态加权</text>

        {/* encoder tiles */}
        {SRC.map((w, i) => (
          <g key={i}>
            <Tile x={40 + i * 70} y={220} w={56} h={28} fill="#dbeafe" stroke="#3b82f6" label={w} />
            <line x1={68 + i * 70} y1={248} x2={68 + i * 70} y2={264} stroke="#3b82f6" strokeWidth={1.4} />
            <Tile x={40 + i * 70} y={264} w={56} h={26} fill="#fce7f3" stroke="#ec4899" label={`h${i + 1}`} />
          </g>
        ))}
        {/* BiRNN 反向连接 */}
        <text x={150} y={310} fontSize={9} fontStyle="italic" fill="#6b7280">↔ BiGRU 双向</text>

        {/* decoder 每一步连不同 alpha */}
        {TGT.map((w, ti) => {
          const cx = 400 + ti * 80;
          // 加权连线 — 高亮对角元素
          return (
            <g key={ti}>
              {SRC.map((_, si) => {
                const isMatch = ti === si;
                const op = isMatch ? 0.9 : 0.12;
                const stroke = isMatch ? "#10b981" : "#9ca3af";
                return (
                  <line key={si} x1={68 + si * 70} y1={290} x2={cx + 25} y2={232} stroke={stroke} strokeWidth={isMatch ? 1.6 : 0.9} opacity={op} />
                );
              })}
              <Tile x={cx} y={214} w={50} h={28} fill="#ecfdf5" stroke="#10b981" label={w} />
              <text x={cx + 25} y={209} textAnchor="middle" fontSize={9} fontWeight={600} fill="#065f46">c_{ti + 1}</text>
            </g>
          );
        })}

        <text x={500} y={205} fontSize={10} fontStyle="italic" fill="#065f46">译"我"看 I,译"爱"看 love,译"猫"看 cats</text>
      </g>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        信息载体从 1 个 d 维向量 → T 个 d 维向量的集合 · 长句质量回到与短句平行
      </text>
    </svg>
  );
}
