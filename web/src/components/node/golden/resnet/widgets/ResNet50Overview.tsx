import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface LayerSpec {
  id: string;
  label: string;
  spatial: number;
  channels: number;
  semantic: string;
  baseFill: string;
  stroke: string;
}

const LAYERS: LayerSpec[] = [
  { id: "input", label: "Input", spatial: 224, channels: 3, semantic: "RGB 像素",
    baseFill: "#fef3c7", stroke: "#d97706" },
  { id: "conv1", label: "conv1 7×7 /2", spatial: 112, channels: 64, semantic: "边缘",
    baseFill: "#fce7f3", stroke: "#db2777" },
  { id: "pool1", label: "MaxPool /2", spatial: 56, channels: 64, semantic: "边缘",
    baseFill: "#fce7f3", stroke: "#db2777" },
  { id: "stage1", label: "Stage1 ×3", spatial: 56, channels: 256, semantic: "纹理",
    baseFill: "#fbcfe8", stroke: "#db2777" },
  { id: "stage2", label: "Stage2 ×4", spatial: 28, channels: 512, semantic: "部件",
    baseFill: "#f9a8d4", stroke: "#db2777" },
  { id: "stage3", label: "Stage3 ×6", spatial: 14, channels: 1024, semantic: "物体",
    baseFill: "#f472b6", stroke: "#db2777" },
  { id: "stage4", label: "Stage4 ×3", spatial: 7, channels: 2048, semantic: "概念",
    baseFill: "#ec4899", stroke: "#9d174d" },
  { id: "gap", label: "GAP", spatial: 1, channels: 2048, semantic: "压缩向量",
    baseFill: "#ecfdf5", stroke: "#059669" },
  { id: "fc", label: "FC 1000", spatial: 1, channels: 1000, semantic: "类别概率",
    baseFill: "#ecfdf5", stroke: "#059669" },
];

const BASE_Y = 200;
const START_X = 40;
const GAP_X = 36;
const PLANE_OFFSET = 4; // 每个后置平面相对前平面的偏移（向右上）

interface LayerGeom {
  layer: LayerSpec;
  x: number;
  frontSize: number;
  stackCount: number;
  centerX: number;
  stackWidth: number;
}

function computeGeoms(): LayerGeom[] {
  const geoms: LayerGeom[] = [];
  let cursor = START_X;
  for (const layer of LAYERS) {
    const frontSize = Math.max(16, Math.sqrt(layer.spatial) * 4.5);
    const stackCount = Math.min(5, Math.max(1, Math.floor(Math.log2(layer.channels))));
    const stackWidth = frontSize + (stackCount - 1) * PLANE_OFFSET;
    const x = cursor;
    geoms.push({
      layer,
      x,
      frontSize,
      stackCount,
      centerX: x + frontSize / 2,
      stackWidth,
    });
    cursor += stackWidth + GAP_X;
  }
  return geoms;
}

interface LayerStackProps extends LayerGeom {
  isHovered: boolean;
  onHoverStart: () => void;
  onHoverEnd: () => void;
  onToggle: () => void;
}

function LayerStack({
  layer,
  x,
  frontSize,
  stackCount,
  isHovered,
  onHoverStart,
  onHoverEnd,
  onToggle,
}: LayerStackProps) {
  const frontY = BASE_Y - frontSize;

  // 从 back-most (i=0) 到 front-most (i=stackCount-1) 渲染
  const planes: { px: number; py: number; opacity: number; isFront: boolean }[] = [];
  for (let i = 0; i < stackCount; i++) {
    const offsetFromFront = (stackCount - 1 - i) * PLANE_OFFSET;
    const px = x + offsetFromFront;
    const py = frontY - offsetFromFront;
    const opacity =
      stackCount === 1 ? 1 : 0.22 + 0.78 * (i / (stackCount - 1));
    planes.push({ px, py, opacity, isFront: i === stackCount - 1 });
  }

  return (
    <motion.g
      onMouseEnter={onHoverStart}
      onMouseLeave={onHoverEnd}
      onClick={onToggle}
      animate={{ scale: isHovered ? 1.05 : 1 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      style={{
        cursor: "pointer",
        transformOrigin: `${x + frontSize / 2}px ${frontY + frontSize / 2}px`,
      }}
    >
      {planes.map((p, i) => (
        <rect
          key={i}
          x={p.px}
          y={p.py}
          width={frontSize}
          height={frontSize}
          rx={3}
          fill={layer.baseFill}
          stroke={layer.stroke}
          strokeWidth={p.isFront ? 1.2 : 0.6}
          opacity={p.opacity}
        />
      ))}
    </motion.g>
  );
}

export function ResNet50Overview() {
  const geoms = computeGeoms();
  const [hoverId, setHoverId] = useState<string | null>(null);
  const hoveredGeom = hoverId
    ? geoms.find((g) => g.layer.id === hoverId)
    : null;

  const last = geoms[geoms.length - 1];
  const totalWidth = Math.max(1000, last.x + last.stackWidth + 40);

  return (
    <div>
      <h3 style={{ fontSize: "var(--fs-lg)", margin: "0 0 var(--space-3)" }}>
        ResNet-50 整体架构
      </h3>
      <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
      <svg
        viewBox={`0 0 ${totalWidth} 280`}
        style={{
          width: "100%",
          minWidth: 760,
          height: "auto",
          display: "block",
        }}
        role="img"
        aria-label="ResNet-50 整体架构总览，9 层层叠平面"
      >
        {/* 极简连接线：前面方块右-中 → 下一层左-中 */}
        {geoms.slice(0, -1).map((source, i) => {
          const target = geoms[i + 1];
          const x1 = source.x + source.frontSize;
          const y1 = BASE_Y - source.frontSize / 2;
          const x2 = target.x;
          const y2 = BASE_Y - target.frontSize / 2;
          return (
            <line
              key={`conn-${source.layer.id}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="var(--ink-muted)"
              strokeWidth={1}
              opacity={0.45}
            />
          );
        })}

        {/* 9 层层叠平面 + stagger 入场 */}
        {geoms.map((g, i) => (
          <motion.g
            key={g.layer.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1, duration: 0.4, ease: "easeOut" }}
          >
            <LayerStack
              {...g}
              isHovered={hoverId === g.layer.id}
              onHoverStart={() => setHoverId(g.layer.id)}
              onHoverEnd={() => setHoverId(null)}
              onToggle={() =>
                setHoverId((cur) => (cur === g.layer.id ? null : g.layer.id))
              }
            />
            {/* layer label */}
            <text
              x={g.centerX}
              y={BASE_Y + 24}
              textAnchor="middle"
              fontSize={13}
              fill="var(--ink-primary)"
              fontWeight={500}
            >
              {g.layer.label}
            </text>
            {/* spatial × channels */}
            <text
              x={g.centerX}
              y={BASE_Y + 42}
              textAnchor="middle"
              fontSize={11}
              fill="var(--ink-muted)"
            >
              {g.layer.spatial > 1
                ? `${g.layer.spatial}² × ${g.layer.channels}`
                : `${g.layer.channels}`}
            </text>
            {/* semantic label */}
            <text
              x={g.centerX}
              y={BASE_Y + 62}
              textAnchor="middle"
              fontSize={13}
              fill={g.layer.stroke}
              fontWeight={600}
            >
              {g.layer.semantic}
            </text>
          </motion.g>
        ))}

        {/* hover tooltip */}
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
                x={Math.max(0, Math.min(hoveredGeom.centerX - 70, totalWidth - 140))}
                y={8}
                width={140}
                height={50}
                rx={4}
                fill="var(--bg-surface)"
                stroke="var(--border)"
                strokeWidth={1}
                filter="drop-shadow(0 2px 8px rgba(0,0,0,0.06))"
              />
              <text
                x={Math.max(0, Math.min(hoveredGeom.centerX - 70, totalWidth - 140)) + 70}
                y={27}
                textAnchor="middle"
                fontSize={12}
                fill="var(--ink-secondary)"
              >
                {hoveredGeom.layer.label}
              </text>
              <text
                x={Math.max(0, Math.min(hoveredGeom.centerX - 70, totalWidth - 140)) + 70}
                y={45}
                textAnchor="middle"
                fontSize={12}
                fontFamily="var(--font-mono)"
                fill="var(--ink-primary)"
              >
                [B, {hoveredGeom.layer.channels}, {hoveredGeom.layer.spatial}, {hoveredGeom.layer.spatial}]
              </text>
            </motion.g>
          )}
        </AnimatePresence>
      </svg>
      </div>
      <p
        style={{
          fontSize: "var(--fs-sm)",
          color: "var(--ink-muted)",
          marginTop: "var(--space-3)",
          textAlign: "center",
        }}
      >
        每层是一叠平面：前面方块边长 = 空间维 (H×W) 真实缩放；后面叠层数代表通道深度（log 尺度）。
        底部标签是该层"看到"的特征性质（参考 Zeiler &amp; Fergus 2014 可视化）。
      </p>
    </div>
  );
}
