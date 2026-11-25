const W = 700;
const H = 280;

// 对比 CNN 和 ViT 的归纳偏置:
//   CNN: 卷积 = locality + translation invariance(强 prior,小数据下省事)
//   ViT: 全 attention = 无 prior(大数据下能学到更通用表示)
// 让 viewer 理解为什么"小数据 CNN 强、大数据 ViT 反超"。

function Card({
  x, y, w, h, fill, stroke, title, bullets,
}: {
  x: number; y: number; w: number; h: number;
  fill: string; stroke: string; title: string; bullets: string[];
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + 22} textAnchor="middle" fontSize={13} fontWeight={700} fill="#1f2937">{title}</text>
      {bullets.map((b, i) => (
        <text key={i} x={x + 14} y={y + 50 + i * 18} fontSize={10} fill="#374151">
          • {b}
        </text>
      ))}
    </g>
  );
}

export function InductiveBiasCompare() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CNN vs ViT 归纳偏置对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        归纳偏置:CNN 内置先验 vs ViT 让数据自己学
      </text>

      <Card
        x={30}
        y={40}
        w={310}
        h={200}
        fill="#fef3c7"
        stroke="#f59e0b"
        title="CNN(强归纳偏置)"
        bullets={[
          "卷积核 → locality (邻近像素相关)",
          "权重共享 → translation invariance",
          "层次特征 → 从纹理到形状到物体",
          "→ 小数据集就能 work,数据不足时强",
          "→ 但天花板低,大数据增益递减",
        ]}
      />
      <Card
        x={360}
        y={40}
        w={310}
        h={200}
        fill="#dbeafe"
        stroke="#3b82f6"
        title="ViT(无视觉先验)"
        bullets={[
          "Self-attention 全连接 → 任意 patch 都可关注",
          "无 locality / 无 translation invariance",
          "纯靠数据学 \"哪些 patch 相关\"",
          "→ 小数据下不如 CNN,被 prior 压一头",
          "→ 大数据下持续吃饱,反超 CNN",
        ]}
      />

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        \"Inductive bias trap\":先验加得越多 → 小数据快 / 大数据卡天花板
      </text>
    </svg>
  );
}
