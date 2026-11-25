interface Props {
  scaled: boolean;
  highlightStep: 0 | 1 | 2 | 3;
}

// Scaled Dot-Product Attention 流程:
//   Q (n×dk)   K (n×dk)   V (n×dv)
//      └── matmul ──┘
//        QKᵀ (n×n)
//           ↓ /√dk
//        scaled scores
//           ↓ softmax
//        weights (n×n)
//           ↓ · V
//        output (n×dv)
//
// 用 highlightStep 高亮当前 viewer 在看哪一步,跟外面 view 选择同步。

const W = 700;
const H = 320;

// 配色:input 黄、compute 粉、output 绿、data 蓝
const COL = {
  input: { fill: "#fef3c7", stroke: "#f59e0b" },
  compute: { fill: "#fce7f3", stroke: "#ec4899" },
  output: { fill: "#ecfdf5", stroke: "#10b981" },
  data: { fill: "#dbeafe", stroke: "#3b82f6" },
};

function Box({
  x,
  y,
  w,
  h,
  fill,
  stroke,
  label,
  sublabel,
  highlight,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  stroke: string;
  label: string;
  sublabel?: string;
  highlight?: boolean;
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={6}
        fill={fill}
        stroke={stroke}
        strokeWidth={highlight ? 3 : 1.5}
        opacity={highlight === false ? 0.5 : 1}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 - 2}
        textAnchor="middle"
        fontSize={13}
        fontWeight={600}
        fill="#1f2937"
      >
        {label}
      </text>
      {sublabel && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 14}
          textAnchor="middle"
          fontSize={10}
          fill="#6b7280"
        >
          {sublabel}
        </text>
      )}
    </g>
  );
}

function Arrow({
  x1,
  y1,
  x2,
  y2,
  label,
  highlight,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label?: string;
  highlight?: boolean;
}) {
  const color = highlight ? "#ec4899" : "#9ca3af";
  return (
    <g>
      <defs>
        <marker
          id={`arr-${x1}-${y1}-${x2}-${y2}`}
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={color}
        strokeWidth={highlight ? 2.5 : 1.5}
        markerEnd={`url(#arr-${x1}-${y1}-${x2}-${y2})`}
      />
      {label && (
        <text
          x={(x1 + x2) / 2}
          y={(y1 + y2) / 2 - 6}
          textAnchor="middle"
          fontSize={11}
          fontStyle="italic"
          fill={highlight ? "#ec4899" : "#6b7280"}
        >
          {label}
        </text>
      )}
    </g>
  );
}

export function QKVPipeline({ scaled, highlightStep }: Props) {
  // Steps:
  //   1 = QKᵀ (raw scores)
  //   2 = / √dk
  //   3 = softmax → weights
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="Scaled dot-product attention 流程图"
    >
      <Box
        x={20}
        y={40}
        w={90}
        h={50}
        {...COL.input}
        label="Q"
        sublabel="n × dk"
      />
      <Box
        x={20}
        y={135}
        w={90}
        h={50}
        {...COL.input}
        label="K"
        sublabel="n × dk"
      />
      <Box
        x={20}
        y={230}
        w={90}
        h={50}
        {...COL.input}
        label="V"
        sublabel="n × dv"
      />

      {/* Step 1: matmul */}
      <Arrow x1={110} y1={65} x2={195} y2={140} highlight={highlightStep === 1} />
      <Arrow x1={110} y1={160} x2={195} y2={155} highlight={highlightStep === 1} />
      <Box
        x={200}
        y={130}
        w={120}
        h={60}
        {...COL.compute}
        label="Q · Kᵀ"
        sublabel="n × n scores"
        highlight={highlightStep === 1}
      />

      {/* Step 2: scale */}
      <Arrow
        x1={320}
        y1={160}
        x2={400}
        y2={160}
        label={scaled ? "÷ √dk" : "(skip)"}
        highlight={highlightStep === 2}
      />
      <Box
        x={400}
        y={130}
        w={120}
        h={60}
        {...COL.compute}
        label={scaled ? "scaled scores" : "raw scores"}
        sublabel="n × n"
        highlight={highlightStep === 2}
      />

      {/* Step 3: softmax */}
      <Arrow
        x1={520}
        y1={160}
        x2={600}
        y2={160}
        label="softmax"
        highlight={highlightStep === 3}
      />
      <Box
        x={600}
        y={130}
        w={80}
        h={60}
        {...COL.data}
        label="weights"
        sublabel="行和=1"
        highlight={highlightStep === 3}
      />

      {/* Output: weights · V */}
      <Arrow x1={640} y1={190} x2={400} y2={260} />
      <Arrow x1={110} y1={255} x2={395} y2={262} />
      <Box
        x={400}
        y={250}
        w={120}
        h={50}
        {...COL.output}
        label="output"
        sublabel="n × dv"
      />
    </svg>
  );
}
