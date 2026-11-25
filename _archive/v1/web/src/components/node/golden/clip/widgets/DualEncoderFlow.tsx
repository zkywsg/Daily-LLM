const W = 700;
const H = 280;

// 双塔结构卡通:
//   image → ViT/ResNet → image_emb
//   text  → Transformer → text_emb
// 两个 encoder 完全独立,最后用 linear projection 把维度对齐到同一 d_emb (512)。
// 配色:输入黄,encoder 粉,投影后蓝(共享空间)。

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

export function DualEncoderFlow() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CLIP dual encoder architecture">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        Dual Encoder:image 塔 + text 塔 独立编码,投影到 d_emb=512 共享空间
      </text>

      {/* 上路:image */}
      <Box x={30} y={60} w={80} h={50} fill="#fef3c7" stroke="#f59e0b" label="🐱" sub="image" />
      <Arrow x1={110} y1={85} x2={170} y2={85} />
      <Box x={170} y={60} w={140} h={50} fill="#fce7f3" stroke="#ec4899" label="Vision Encoder" sub="ViT-L/14 or RN50" />
      <Arrow x1={310} y1={85} x2={370} y2={85} />
      <Box x={370} y={60} w={100} h={50} fill="#fce7f3" stroke="#ec4899" label="Linear" sub="proj 768→512" />
      <Arrow x1={470} y1={85} x2={530} y2={85} />
      <Box x={530} y={60} w={140} h={50} fill="#dbeafe" stroke="#3b82f6" label="image_emb" sub="ℝ⁵¹²" />

      {/* 下路:text */}
      <Box x={30} y={170} w={80} h={50} fill="#fef3c7" stroke="#f59e0b" label="caption" sub='"a photo of a cat"' />
      <Arrow x1={110} y1={195} x2={170} y2={195} />
      <Box x={170} y={170} w={140} h={50} fill="#fce7f3" stroke="#ec4899" label="Text Encoder" sub="12-layer Transformer" />
      <Arrow x1={310} y1={195} x2={370} y2={195} />
      <Box x={370} y={170} w={100} h={50} fill="#fce7f3" stroke="#ec4899" label="Linear" sub="proj 512→512" />
      <Arrow x1={470} y1={195} x2={530} y2={195} />
      <Box x={530} y={170} w={140} h={50} fill="#dbeafe" stroke="#3b82f6" label="text_emb" sub="ℝ⁵¹²" />

      {/* 注脚 */}
      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        两个塔不共享参数 · 投影后 L2-normalize → 后续 cos sim 退化成点积
      </text>
    </svg>
  );
}
