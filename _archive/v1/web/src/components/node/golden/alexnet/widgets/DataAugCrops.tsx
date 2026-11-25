const W = 700;
const H = 320;

interface Props {
  cropX: number;     // 0..32, position of 224 crop within 256
  cropY: number;
  flipped: boolean;
  colorJitter: number; // 0..1
}

// 左边:256×256 原图(纯色块代表),叠加 224×224 红框
// 右边:把 crop 出的样本展示成增强后样子
export function DataAugCrops({ cropX, cropY, flipped, colorJitter }: Props) {
  const orig_x = 40;
  const orig_y = 60;
  const orig_size = 220;
  const scale = orig_size / 256;
  const crop_box_x = orig_x + cropX * scale;
  const crop_box_y = orig_y + cropY * scale;
  const crop_box_w = 224 * scale;

  const sample_x = 400;
  const sample_y = 60;
  const sample_size = 220;

  // 颜色扰动:base 是青绿(R=120, G=180, B=140)平均,jitter 推向红/蓝
  const baseR = 120, baseG = 180, baseB = 140;
  const dR = (colorJitter - 0.5) * 80;
  const dB = -(colorJitter - 0.5) * 80;
  const fill = `rgb(${baseR + dR}, ${baseG}, ${baseB + dB})`;

  // 模拟"图像主体":中央一个不同颜色的椭圆
  const subjFill = `rgb(${200 + dR}, ${100}, ${80 + dB})`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Data augmentation crop and color jitter">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        数据增强 — 256×256 随机 224 crop · 水平翻转 · PCA 颜色扰动
      </text>

      {/* 原图 (256x256 缩放) */}
      <text x={orig_x + orig_size / 2} y={orig_y - 8} textAnchor="middle" fontSize={11} fontWeight={600} fill="#6b7280">原图 256×256</text>
      <rect x={orig_x} y={orig_y} width={orig_size} height={orig_size} fill="rgb(120,180,140)" />
      <ellipse cx={orig_x + orig_size / 2} cy={orig_y + orig_size / 2 + 10} rx={48} ry={36} fill="rgb(200,100,80)" />
      <circle cx={orig_x + orig_size / 2 - 15} cy={orig_y + orig_size / 2 - 5} r={6} fill="#1f2937" />
      <circle cx={orig_x + orig_size / 2 + 15} cy={orig_y + orig_size / 2 - 5} r={6} fill="#1f2937" />

      {/* crop 红框 */}
      <rect x={crop_box_x} y={crop_box_y} width={crop_box_w} height={crop_box_w} fill="none" stroke="#ec4899" strokeWidth={2.5} strokeDasharray="6 4" />
      <text x={crop_box_x + crop_box_w / 2} y={crop_box_y - 4} textAnchor="middle" fontSize={9} fontWeight={700} fill="#ec4899">crop 224</text>

      {/* arrow → */}
      <line x1={orig_x + orig_size + 14} y1={orig_y + orig_size / 2} x2={sample_x - 14} y2={sample_y + sample_size / 2} stroke="#9ca3af" strokeWidth={1.8} markerEnd="url(#aug-arr)" />
      <defs>
        <marker id="aug-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>

      {/* 增强后样本 */}
      <text x={sample_x + sample_size / 2} y={sample_y - 8} textAnchor="middle" fontSize={11} fontWeight={600} fill="#6b7280">增强后 224×224</text>
      <g transform={flipped ? `translate(${sample_x * 2 + sample_size}, 0) scale(-1, 1)` : ""}>
        <rect x={sample_x} y={sample_y} width={sample_size} height={sample_size} fill={fill} />
        {/* 模拟"主体"的位置,根据 cropX/Y 偏移 */}
        <ellipse cx={sample_x + sample_size / 2 + (16 - cropX) * 1.2} cy={sample_y + sample_size / 2 + 10 + (16 - cropY) * 1.2} rx={50} ry={38} fill={subjFill} />
        <circle cx={sample_x + sample_size / 2 - 15 + (16 - cropX) * 1.2} cy={sample_y + sample_size / 2 - 5 + (16 - cropY) * 1.2} r={6} fill="#1f2937" />
        <circle cx={sample_x + sample_size / 2 + 15 + (16 - cropX) * 1.2} cy={sample_y + sample_size / 2 - 5 + (16 - cropY) * 1.2} r={6} fill="#1f2937" />
      </g>

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        32² × 2 = 2048 种变体 / 张图 — 把 120 万张训练集"放大"到 25 亿样本
      </text>
    </svg>
  );
}
