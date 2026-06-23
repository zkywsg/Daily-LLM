interface Props {
  dModel: number;
  numHeads: number;
}

const W = 700;
const H = 240;

// d_model 整段 → h 个 d_k 切片
// 通过对比"一个大注意力 vs h 个小注意力"让 viewer 看到 Multi-Head 的本质:
// 多个低维子空间各看一种关系,然后 concat。
export function HeadSplitSVG({ dModel, numHeads }: Props) {
  const dk = Math.floor(dModel / numHeads);
  const sliceW = (W - 40) / numHeads;
  const palette = [
    "#fce7f3",
    "#dbeafe",
    "#ecfdf5",
    "#fef3c7",
    "#fed7e2",
    "#bfdbfe",
    "#bbf7d0",
    "#fde68a",
    "#f5d0fe",
    "#a5f3fc",
  ];
  const heads = Array.from({ length: numHeads });
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={`Multi-Head split: d_model ${dModel} → ${numHeads} 个 head × dk ${dk}`}
    >
      {/* d_model 整段(粉色 compute) */}
      <rect
        x={20}
        y={30}
        width={W - 40}
        height={42}
        rx={6}
        fill="#fce7f3"
        stroke="#ec4899"
        strokeWidth={1.5}
      />
      <text
        x={W / 2}
        y={56}
        textAnchor="middle"
        fontSize={14}
        fontWeight={600}
        fill="#1f2937"
      >
        Q / K / V projection · d_model = {dModel}
      </text>

      {/* 分裂箭头 */}
      {heads.map((_, h) => {
        const cx = 20 + sliceW * (h + 0.5);
        return (
          <line
            key={`arr-${h}`}
            x1={cx}
            y1={75}
            x2={cx}
            y2={115}
            stroke="#9ca3af"
            strokeWidth={1.5}
            markerEnd="url(#multi-arr)"
          />
        );
      })}
      <defs>
        <marker
          id="multi-arr"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>

      {/* h 个 head 子空间 */}
      {heads.map((_, h) => (
        <g key={`head-${h}`}>
          <rect
            x={20 + sliceW * h + 3}
            y={120}
            width={sliceW - 6}
            height={48}
            rx={5}
            fill={palette[h % palette.length]}
            stroke="#6b7280"
            strokeWidth={1}
          />
          <text
            x={20 + sliceW * (h + 0.5)}
            y={140}
            textAnchor="middle"
            fontSize={11}
            fontWeight={600}
            fill="#1f2937"
          >
            head {h + 1}
          </text>
          <text
            x={20 + sliceW * (h + 0.5)}
            y={156}
            textAnchor="middle"
            fontSize={10}
            fill="#6b7280"
          >
            dk={dk}
          </text>
        </g>
      ))}

      {/* concat 回 d_model */}
      {heads.map((_, h) => {
        const cx = 20 + sliceW * (h + 0.5);
        return (
          <line
            key={`out-${h}`}
            x1={cx}
            y1={170}
            x2={cx}
            y2={195}
            stroke="#9ca3af"
            strokeWidth={1.5}
            markerEnd="url(#multi-arr)"
          />
        );
      })}
      <rect
        x={20}
        y={200}
        width={W - 40}
        height={32}
        rx={5}
        fill="#ecfdf5"
        stroke="#10b981"
        strokeWidth={1.5}
      />
      <text
        x={W / 2}
        y={221}
        textAnchor="middle"
        fontSize={12}
        fontWeight={600}
        fill="#1f2937"
      >
        concat → W_O → output · d_model = {dk * numHeads}
        {dk * numHeads !== dModel && (
          <tspan fill="#dc2626"> (注意:dk×h 不等于 d_model,选 head 数让它整除)</tspan>
        )}
      </text>
    </svg>
  );
}
