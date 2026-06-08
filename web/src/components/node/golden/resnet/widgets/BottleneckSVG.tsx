import { BASIC_BLOCK_PARAMS, BOTTLENECK_PARAMS } from "../lib/curves";

interface Props {
  blockType: "basic" | "bottleneck";
  width?: number;
  height?: number;
}

// 元素尺寸常量
const RECT_H = 36;
const RECT_Y = 182;
const ROW_CENTER = 200;
const CIRCLE_R = 16;
const W_INPUT_OUTPUT = 50;
const W_CONV = 70;
const W_RELU = 50;
const GAP = 8;
const LEFT_MARGIN = 14;

interface Layer {
  label: string;
  sub: string;
  w: number;
}

const BASIC_LAYERS: Layer[] = [
  { label: "Conv 3×3", sub: "64", w: W_CONV },
  { label: "ReLU", sub: "", w: W_RELU },
  { label: "Conv 3×3", sub: "64", w: W_CONV },
];

const BOTTLENECK_LAYERS: Layer[] = [
  { label: "Conv 1×1 ↓", sub: "64", w: W_CONV },
  { label: "ReLU", sub: "", w: W_RELU },
  { label: "Conv 3×3", sub: "64", w: W_CONV },
  { label: "ReLU", sub: "", w: W_RELU },
  { label: "Conv 1×1 ↑", sub: "256", w: W_CONV },
];

export function BottleneckSVG({
  blockType,
  width = 600,
  height = 360,
}: Props) {
  const isBasic = blockType === "basic";
  const params = isBasic ? BASIC_BLOCK_PARAMS : BOTTLENECK_PARAMS;
  const layers = isBasic ? BASIC_LAYERS : BOTTLENECK_LAYERS;

  // 计算各元素 x 位置（流式布局）
  let cursor = LEFT_MARGIN;
  const inputX = cursor;
  cursor += W_INPUT_OUTPUT + GAP;

  const layerXs: number[] = [];
  for (const L of layers) {
    layerXs.push(cursor);
    cursor += L.w + GAP;
  }

  const addCX = cursor + CIRCLE_R;
  cursor += CIRCLE_R * 2 + GAP;

  const outputX = cursor;
  cursor += W_INPUT_OUTPUT;

  const contentRight = cursor + LEFT_MARGIN;
  // 如果默认 width 不够，自动扩展 viewBox
  const vbWidth = Math.max(width, contentRight);

  // shortcut 弧：从 input 顶部跨到 ⊕ 顶部
  const shortcutStartX = inputX + W_INPUT_OUTPUT / 2;
  const shortcutEndX = addCX;
  const shortcutPath = `M ${shortcutStartX} ${RECT_Y} C ${shortcutStartX} 80, ${shortcutEndX} 80, ${shortcutEndX} ${RECT_Y + 4}`;

  return (
    <svg
      viewBox={`0 0 ${vbWidth} ${height}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label={`${isBasic ? "BasicBlock" : "Bottleneck"} 残差块结构`}
    >
      {/* 标题 */}
      <text
        x={vbWidth / 2}
        y={30}
        textAnchor="middle"
        fontSize={16}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        {isBasic ? "BasicBlock" : "Bottleneck"}
      </text>

      {/* shortcut 弧线（先画，避免被遮挡） */}
      <path d={shortcutPath} stroke="#2563eb" strokeWidth={3} fill="none" />
      <text
        x={(shortcutStartX + shortcutEndX) / 2}
        y={72}
        textAnchor="middle"
        fontSize={13}
        fontStyle="italic"
        fontWeight={600}
        fill="#1e40af"
      >
        identity
      </text>

      {/* 主路前向箭头（在主路元素下方） */}
      <g stroke="#9d174d" strokeWidth={1.2} fill="none">
        {/* input → first layer */}
        <line
          x1={inputX + W_INPUT_OUTPUT}
          y1={ROW_CENTER}
          x2={layerXs[0]}
          y2={ROW_CENTER}
        />
        {/* between layers */}
        {layers.slice(0, -1).map((L, i) => (
          <line
            key={i}
            x1={layerXs[i] + L.w}
            y1={ROW_CENTER}
            x2={layerXs[i + 1]}
            y2={ROW_CENTER}
          />
        ))}
        {/* last layer → ⊕ */}
        <line
          x1={layerXs[layers.length - 1] + layers[layers.length - 1].w}
          y1={ROW_CENTER}
          x2={addCX - CIRCLE_R}
          y2={ROW_CENTER}
        />
        {/* ⊕ → output */}
        <line
          x1={addCX + CIRCLE_R}
          y1={ROW_CENTER}
          x2={outputX}
          y2={ROW_CENTER}
        />
      </g>

      {/* Input */}
      <rect
        x={inputX}
        y={RECT_Y}
        width={W_INPUT_OUTPUT}
        height={RECT_H}
        rx={6}
        fill="#fef3c7"
        stroke="#d97706"
        strokeWidth={1.5}
      />
      <text
        x={inputX + W_INPUT_OUTPUT / 2}
        y={ROW_CENTER + 5}
        textAnchor="middle"
        fontSize={14}
        fill="#92400e"
      >
        x
      </text>

      {/* 主路层 */}
      {layers.map((L, i) => (
        <g key={i}>
          <rect
            x={layerXs[i]}
            y={RECT_Y}
            width={L.w}
            height={RECT_H}
            rx={6}
            fill="#fce7f3"
            stroke="#db2777"
            strokeWidth={1.5}
          />
          <text
            x={layerXs[i] + L.w / 2}
            y={L.sub ? 197 : ROW_CENTER + 4}
            textAnchor="middle"
            fontSize={12}
            fill="#9d174d"
          >
            {L.label}
          </text>
          {L.sub && (
            <text
              x={layerXs[i] + L.w / 2}
              y={211}
              textAnchor="middle"
              fontSize={10}
              fill="#9d174d"
            >
              {L.sub}
            </text>
          )}
        </g>
      ))}

      {/* ⊕ */}
      <circle
        cx={addCX}
        cy={ROW_CENTER}
        r={CIRCLE_R}
        fill="#fef3c7"
        stroke="#d97706"
        strokeWidth={2}
      />
      <text
        x={addCX}
        y={ROW_CENTER + 6}
        textAnchor="middle"
        fontSize={18}
        fill="#92400e"
        fontWeight={700}
      >
        ⊕
      </text>

      {/* Output */}
      <rect
        x={outputX}
        y={RECT_Y}
        width={W_INPUT_OUTPUT}
        height={RECT_H}
        rx={6}
        fill="#fef3c7"
        stroke="#d97706"
        strokeWidth={1.5}
      />
      <text
        x={outputX + W_INPUT_OUTPUT / 2}
        y={ROW_CENTER + 5}
        textAnchor="middle"
        fontSize={14}
        fill="#92400e"
      >
        y
      </text>

      {/* F(x)+x 标签：放在 ⊕ 下方，避免与右侧 output 冲突 */}
      <text
        x={addCX}
        y={ROW_CENTER + 42}
        textAnchor="middle"
        fontSize={13}
        fontStyle="italic"
        fill="var(--ink-primary)"
      >
        F(x) + x
      </text>

      {/* 参数量 */}
      <g transform={`translate(${vbWidth / 2}, 290)`}>
        <text
          textAnchor="middle"
          fontSize={13}
          fill="var(--ink-secondary)"
        >
          参数量
        </text>
        <text
          y={26}
          textAnchor="middle"
          fontSize={22}
          fontWeight={600}
          fill="var(--ink-primary)"
        >
          {params.toLocaleString()}
        </text>
      </g>
    </svg>
  );
}
