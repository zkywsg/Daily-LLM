import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface LayerSpec {
  id: string;
  label: string;
  spatial: number;
  channels: number;
  baseFill: string;
  lightFill: string;
  darkFill: string;
  stroke: string;
}

const LAYERS: LayerSpec[] = [
  {
    id: "input",
    label: "Input",
    spatial: 224,
    channels: 3,
    baseFill: "#fef3c7",
    lightFill: "#fffbeb",
    darkFill: "#fde68a",
    stroke: "#d97706",
  },
  {
    id: "conv1",
    label: "conv1 7×7 /2",
    spatial: 112,
    channels: 64,
    baseFill: "#fce7f3",
    lightFill: "#fdf2f8",
    darkFill: "#f9a8d4",
    stroke: "#db2777",
  },
  {
    id: "pool1",
    label: "MaxPool /2",
    spatial: 56,
    channels: 64,
    baseFill: "#fce7f3",
    lightFill: "#fdf2f8",
    darkFill: "#f9a8d4",
    stroke: "#db2777",
  },
  {
    id: "stage1",
    label: "Stage1 ×3",
    spatial: 56,
    channels: 256,
    baseFill: "#fbcfe8",
    lightFill: "#fce7f3",
    darkFill: "#f9a8d4",
    stroke: "#db2777",
  },
  {
    id: "stage2",
    label: "Stage2 ×4",
    spatial: 28,
    channels: 512,
    baseFill: "#f9a8d4",
    lightFill: "#fbcfe8",
    darkFill: "#f472b6",
    stroke: "#db2777",
  },
  {
    id: "stage3",
    label: "Stage3 ×6",
    spatial: 14,
    channels: 1024,
    baseFill: "#f472b6",
    lightFill: "#f9a8d4",
    darkFill: "#ec4899",
    stroke: "#db2777",
  },
  {
    id: "stage4",
    label: "Stage4 ×3",
    spatial: 7,
    channels: 2048,
    baseFill: "#ec4899",
    lightFill: "#f472b6",
    darkFill: "#db2777",
    stroke: "#9d174d",
  },
  {
    id: "gap",
    label: "GAP",
    spatial: 1,
    channels: 2048,
    baseFill: "#ecfdf5",
    lightFill: "#f0fdf4",
    darkFill: "#d1fae5",
    stroke: "#059669",
  },
  {
    id: "fc",
    label: "FC 1000",
    spatial: 1,
    channels: 1000,
    baseFill: "#ecfdf5",
    lightFill: "#f0fdf4",
    darkFill: "#d1fae5",
    stroke: "#059669",
  },
];

const COS30 = 0.866;
const SIN30 = 0.5;
const BASE_Y = 240;
const GAP_X = 30;
const START_X = 40;

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
  pulse: boolean;
}

function Cuboid({
  layer,
  x,
  y,
  W,
  H,
  D,
  isHovered,
  onHoverStart,
  onHoverEnd,
  pulse,
}: CuboidProps) {
  const dx = D * COS30;
  const dy = -D * SIN30;
  return (
    <motion.g
      onMouseEnter={onHoverStart}
      onMouseLeave={onHoverEnd}
      animate={{
        scale: isHovered ? 1.05 : pulse ? 1.08 : 1,
      }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      style={{
        cursor: "help",
        transformOrigin: `${x + (W + dx) / 2}px ${BASE_Y - H / 2}px`,
      }}
    >
      <rect
        x={x}
        y={y}
        width={W}
        height={H}
        fill={layer.baseFill}
        stroke={layer.stroke}
        strokeWidth={1}
      />
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

export function ResNet50Overview() {
  const geoms = computeGeom();
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [playStep, setPlayStep] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);

  const hoveredGeom = hoverId ? geoms.find((g) => g.layer.id === hoverId) : null;

  async function playFlow() {
    if (playing) return;
    setPlaying(true);
    for (let i = 0; i < geoms.length; i++) {
      setPlayStep(i);
      await new Promise((res) => setTimeout(res, 400));
    }
    setPlayStep(null);
    setPlaying(false);
  }

  const particlePos = playStep !== null ? geoms[playStep].centerX : null;

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--space-3)",
          marginBottom: "var(--space-3)",
          flexWrap: "wrap",
        }}
      >
        <h3 style={{ fontSize: "var(--fs-lg)", margin: 0 }}>
          ResNet-50 整体架构
        </h3>
        <button
          onClick={playFlow}
          disabled={playing}
          style={{
            padding: "var(--space-1) var(--space-3)",
            borderRadius: "var(--radius-full)",
            background: playing ? "var(--bg-subtle)" : "var(--accent-link)",
            color: playing ? "var(--ink-muted)" : "var(--bg-surface)",
            fontSize: "var(--fs-sm)",
            fontWeight: 500,
            border: "none",
            cursor: playing ? "default" : "pointer",
          }}
        >
          {playing ? "▶ 播放中…" : "▶ 播放数据流"}
        </button>
      </div>
      <svg
        viewBox="0 0 1000 320"
        style={{ maxWidth: "100%", height: "auto", display: "block" }}
        role="img"
        aria-label="ResNet-50 整体架构总览，9 层立方体"
      >
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
              pulse={playStep === i}
            />
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
          </motion.g>
        ))}

        <AnimatePresence>
          {particlePos !== null && (
            <motion.circle
              key={`particle-${playStep}`}
              cx={particlePos}
              cy={BASE_Y - 80}
              r={6}
              fill="#2563eb"
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              style={{ filter: "drop-shadow(0 0 6px #3b82f6)" }}
            />
          )}
        </AnimatePresence>

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
    </div>
  );
}
