const W = 700;
const H = 200;

// 演示 contrastive loss 的两个对称方向:
//   image→text softmax 行方向 (对每张图,在 N 个 caption 里挑对的)
//   text→image softmax 列方向 (对每段文本,在 N 个图里挑对的)
// 最终 loss = (L_i2t + L_t2i) / 2

export function DualSoftmax() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Contrastive loss: two symmetric softmax directions">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Contrastive loss 双向对称
      </text>

      {/* 行方向 */}
      <g transform="translate(40, 50)">
        <text x={150} y={0} textAnchor="middle" fontSize={11} fontWeight={600} fill="#ec4899">image → text</text>
        <text x={150} y={16} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">行 softmax:每张图在 N caption 里挑对的</text>
        {Array.from({ length: 4 }, (_, j) => (
          <g key={j}>
            <rect
              x={50 + j * 50}
              y={28}
              width={42}
              height={32}
              rx={3}
              fill={j === 1 ? "#ec4899" : "#fce7f3"}
              opacity={j === 1 ? 1 : 0.4}
              stroke="#ec4899"
              strokeWidth={1}
            />
            <text x={71 + j * 50} y={48} textAnchor="middle" fontSize={11} fontWeight={j === 1 ? 700 : 500} fill={j === 1 ? "#fff" : "#831843"}>
              {j === 1 ? "0.74" : "0.09"}
            </text>
          </g>
        ))}
        <text x={150} y={78} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-secondary)">
          → CE loss = −log 0.74
        </text>
      </g>

      {/* 列方向 */}
      <g transform="translate(380, 50)">
        <text x={140} y={0} textAnchor="middle" fontSize={11} fontWeight={600} fill="#3b82f6">text → image</text>
        <text x={140} y={16} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">列 softmax:每段文本在 N image 里挑对的</text>
        <g transform="translate(80, 28)">
          {Array.from({ length: 4 }, (_, i) => (
            <g key={i}>
              <rect
                x={0}
                y={i * 16}
                width={120}
                height={14}
                rx={3}
                fill={i === 1 ? "#3b82f6" : "#dbeafe"}
                opacity={i === 1 ? 1 : 0.4}
                stroke="#3b82f6"
                strokeWidth={1}
              />
              <text x={60} y={i * 16 + 11} textAnchor="middle" fontSize={10} fontWeight={i === 1 ? 700 : 500} fill={i === 1 ? "#fff" : "#1e3a8a"}>
                {i === 1 ? "0.68" : "0.11"}
              </text>
            </g>
          ))}
        </g>
      </g>

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fontFamily="ui-monospace" fill="var(--ink-primary)">
        L_clip = (CE_i2t + CE_t2i) / 2
      </text>
    </svg>
  );
}
