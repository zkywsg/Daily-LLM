import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface LayerSpec {
  id: string;
  label: string;
  spatial: number;
  channels: number;
  semantic: string;
  baseFill: string;
  lightFill: string;
  darkFill: string;
  stroke: string;
}

const LAYERS: LayerSpec[] = [
  { id: "input", label: "Input", spatial: 224, channels: 3, semantic: "RGB 像素",
    baseFill: "#fef3c7", lightFill: "#fffbeb", darkFill: "#fde68a", stroke: "#d97706" },
  { id: "conv1", label: "conv1 7×7 /2", spatial: 112, channels: 64, semantic: "边缘",
    baseFill: "#fce7f3", lightFill: "#fdf2f8", darkFill: "#f9a8d4", stroke: "#db2777" },
  { id: "pool1", label: "MaxPool /2", spatial: 56, channels: 64, semantic: "边缘",
    baseFill: "#fce7f3", lightFill: "#fdf2f8", darkFill: "#f9a8d4", stroke: "#db2777" },
  { id: "stage1", label: "Stage1 ×3", spatial: 56, channels: 256, semantic: "纹理",
    baseFill: "#fbcfe8", lightFill: "#fce7f3", darkFill: "#f9a8d4", stroke: "#db2777" },
  { id: "stage2", label: "Stage2 ×4", spatial: 28, channels: 512, semantic: "部件",
    baseFill: "#f9a8d4", lightFill: "#fbcfe8", darkFill: "#f472b6", stroke: "#db2777" },
  { id: "stage3", label: "Stage3 ×6", spatial: 14, channels: 1024, semantic: "物体",
    baseFill: "#f472b6", lightFill: "#f9a8d4", darkFill: "#ec4899", stroke: "#db2777" },
  { id: "stage4", label: "Stage4 ×3", spatial: 7, channels: 2048, semantic: "概念",
    baseFill: "#ec4899", lightFill: "#f472b6", darkFill: "#db2777", stroke: "#9d174d" },
  { id: "gap", label: "GAP", spatial: 1, channels: 2048, semantic: "压缩向量",
    baseFill: "#ecfdf5", lightFill: "#f0fdf4", darkFill: "#d1fae5", stroke: "#059669" },
  { id: "fc", label: "FC 1000", spatial: 1, channels: 1000, semantic: "类别概率",
    baseFill: "#ecfdf5", lightFill: "#f0fdf4", darkFill: "#d1fae5", stroke: "#059669" },
];

const COS30 = 0.866;
const SIN30 = 0.5;
const BASE_Y = 240;
const GAP_X = 30;
const START_X = 40;
const FLOW_LINE_COUNT = 8;

interface CuboidGeom {
  layer: LayerSpec;
  x: number;
  y: number;
  W: number;
  H: number;
  D: number;
  centerX: number;
}

function computeGeom(): CuboidGeom[] {
  const geoms: CuboidGeom[] = [];
  let cursor = START_X;
  for (const layer of LAYERS) {
    const W = Math.max(6, Math.sqrt(layer.spatial) * 5);
    const H = Math.log2(layer.channels + 1) * 5 + 6;
    const D = W;
    const dx = D * COS30;
    const y = BASE_Y - H;
    geoms.push({
      layer,
      x: cursor,
      y,
      W,
      H,
      D,
      centerX: cursor + (W + dx) / 2,
    });
    cursor += W + dx + GAP_X;
  }
  return geoms;
}

interface CuboidProps extends CuboidGeom {
  isHovered: boolean;
  onHoverStart: () => void;
  onHoverEnd: () => void;
}

function Cuboid({ layer, x, y, W, H, D, isHovered, onHoverStart, onHoverEnd }: CuboidProps) {
  const dx = D * COS30;
  const dy = -D * SIN30;
  return (
    <motion.g
      onMouseEnter={onHoverStart}
      onMouseLeave={onHoverEnd}
      animate={{ scale: isHovered ? 1.05 : 1 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      style={{ cursor: "help", transformOrigin: `${x + (W + dx) / 2}px ${BASE_Y - H / 2}px` }}
    >
      <rect x={x} y={y} width={W} height={H} fill={layer.baseFill} stroke={layer.stroke} strokeWidth={1} />
      <polygon
        points={`${x},${y} ${x + W},${y} ${x + W + dx},${y + dy} ${x + dx},${y + dy}`}
        fill={layer.lightFill}
        stroke={layer.stroke}
        strokeWidth={1}
      />
      <polygon
        points={`${x + W},${y} ${x + W},${y + H} ${x + W + dx},${y + H + dy} ${x + W + dx},${y + dy}`}
        fill={layer.darkFill}
        stroke={layer.stroke}
        strokeWidth={1}
      />
    </motion.g>
  );
}

interface FlowLineProps {
  source: CuboidGeom;
  target: CuboidGeom;
  gradientId: string;
}

function FlowLines({ source, target, gradientId }: FlowLineProps) {
  const sourceX = source.x + source.W;
  const targetX = target.x;
  const dxBetween = targetX - sourceX;

  const lines: string[] = [];
  for (let i = 0; i < FLOW_LINE_COUNT; i++) {
    const t = i / (FLOW_LINE_COUNT - 1);
    const sy = source.y + source.H * t;
    const ty = target.y + target.H * t;
    const cx1 = sourceX + dxBetween * 0.4;
    const cx2 = targetX - dxBetween * 0.4;
    lines.push(`M ${sourceX} ${sy} C ${cx1} ${sy}, ${cx2} ${ty}, ${targetX} ${ty}`);
  }

  return (
    <g>
      {lines.map((d, i) => (
        <path
          key={i}
          d={d}
          stroke={`url(#${gradientId})`}
          strokeWidth={1.2}
          fill="none"
          strokeDasharray="5 7"
          className="flow-line"
        />
      ))}
    </g>
  );
}

export function ResNet50Overview() {
  const geoms = computeGeom();
  const [hoverId, setHoverId] = useState<string | null>(null);
  const hoveredGeom = hoverId ? geoms.find((g) => g.layer.id === hoverId) : null;

  return (
    <div>
      <h3 style={{ fontSize: "var(--fs-lg)", margin: "0 0 var(--space-3)" }}>
        ResNet-50 整体架构
      </h3>
      <svg
        viewBox="0 0 1000 340"
        style={{ maxWidth: "100%", height: "auto", display: "block" }}
        role="img"
        aria-label="ResNet-50 整体架构总览，9 层立方体 + 流动连接线"
      >
        <defs>
          {/* Per-pair linear gradients for flow lines */}
          {geoms.slice(0, -1).map((source, i) => {
            const target = geoms[i + 1];
            const gradientId = `flow-grad-${source.layer.id}-${target.layer.id}`;
            return (
              <linearGradient
                key={gradientId}
                id={gradientId}
                x1={source.x + source.W}
                x2={target.x}
                y1={0}
                y2={0}
                gradientUnits="userSpaceOnUse"
              >
                <stop offset="0%" stopColor={source.layer.stroke} stopOpacity={0.7} />
                <stop offset="100%" stopColor={target.layer.stroke} stopOpacity={0.7} />
              </linearGradient>
            );
          })}
        </defs>

        {/* Inline keyframes for the flowing dash animation */}
        <style>
          {`
            .flow-line {
              animation: flowDash 1.2s linear infinite;
            }
            @keyframes flowDash {
              to { stroke-dashoffset: -12; }
            }
          `}
        </style>

        {/* Flow lines between adjacent cuboids (rendered behind cuboids) */}
        {geoms.slice(0, -1).map((source, i) => {
          const target = geoms[i + 1];
          const gradientId = `flow-grad-${source.layer.id}-${target.layer.id}`;
          return (
            <FlowLines
              key={`flow-${source.layer.id}`}
              source={source}
              target={target}
              gradientId={gradientId}
            />
          );
        })}

        {/* Cuboids with stagger entrance */}
        {geoms.map((g, i) => (
          <motion.g
            key={g.layer.id}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.12, duration: 0.4, ease: "easeOut" }}
          >
            <Cuboid
              {...g}
              isHovered={hoverId === g.layer.id}
              onHoverStart={() => setHoverId(g.layer.id)}
              onHoverEnd={() => setHoverId(null)}
            />
            {/* Layer label */}
            <text
              x={g.centerX}
              y={BASE_Y + 18}
              textAnchor="middle"
              fontSize={10}
              fill="var(--ink-primary)"
              fontWeight={500}
            >
              {g.layer.label}
            </text>
            {/* Spatial × channels */}
            <text
              x={g.centerX}
              y={BASE_Y + 32}
              textAnchor="middle"
              fontSize={9}
              fill="var(--ink-muted)"
            >
              {g.layer.spatial > 1
                ? `${g.layer.spatial}² × ${g.layer.channels}`
                : `${g.layer.channels}`}
            </text>
            {/* Semantic label */}
            <text
              x={g.centerX}
              y={BASE_Y + 48}
              textAnchor="middle"
              fontSize={10}
              fill={g.layer.stroke}
              fontWeight={600}
            >
              {g.layer.semantic}
            </text>
          </motion.g>
        ))}

        {/* Hover tooltip */}
        <AnimatePresence>
          {hoveredGeom && (
            <motion.g
              key={`tooltip-${hoveredGeom.layer.id}`}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              <rect
                x={Math.max(0, Math.min(hoveredGeom.centerX - 70, 860))}
                y={20}
                width={140}
                height={48}
                rx={4}
                fill="var(--bg-surface)"
                stroke="var(--border)"
                strokeWidth={1}
              />
              <text
                x={Math.max(0, Math.min(hoveredGeom.centerX - 70, 860)) + 70}
                y={38}
                textAnchor="middle"
                fontSize={11}
                fill="var(--ink-secondary)"
              >
                {hoveredGeom.layer.label}
              </text>
              <text
                x={Math.max(0, Math.min(hoveredGeom.centerX - 70, 860)) + 70}
                y={56}
                textAnchor="middle"
                fontSize={11}
                fontFamily="var(--font-mono)"
                fill="var(--ink-primary)"
              >
                [B, {hoveredGeom.layer.channels}, {hoveredGeom.layer.spatial}, {hoveredGeom.layer.spatial}]
              </text>
            </motion.g>
          )}
        </AnimatePresence>
      </svg>
      <p
        style={{
          fontSize: "var(--fs-sm)",
          color: "var(--ink-muted)",
          marginTop: "var(--space-3)",
          textAlign: "center",
        }}
      >
        每个立方体代表网络的一个阶段：宽度 = 空间维（H×W），高度 = 通道数（对数尺度）。
        连接线持续流动表示数据传递；底部标签是该层"看到"的特征性质（来自 Zeiler & Fergus 2014 经典可视化）。
      </p>
    </div>
  );
}
