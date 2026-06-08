import { motion } from "framer-motion";
import { Link } from "react-router";
import type { NodeData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { fadeUp, duration, ease } from "../../lib/motion";

interface NodeHoverCardProps {
  node: NodeData;
  x: number;
  y: number;
}

export function NodeHoverCard({ node, x, y }: NodeHoverCardProps) {
  const nodeSlug = node.path.split("/").pop()!.replace(/\.md$/, "");
  return (
    <motion.div
      variants={fadeUp}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={{ duration: duration.fast, ease: ease.out as unknown as number[] }}
      style={{
        position: "absolute",
        left: x,
        top: y,
        zIndex: 10,
        padding: "var(--space-4)",
        background: "var(--bg-surface)",
        border: `2px solid ${familyColorVar(node.family)}`,
        borderRadius: "var(--radius-md)",
        boxShadow: "var(--shadow-lg)",
        minWidth: 240,
        maxWidth: 320,
        pointerEvents: "none",
      }}
    >
      <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        {node.year} · {node.family}
      </div>
      <div
        style={{
          fontSize: "var(--fs-lg)",
          fontWeight: 600,
          marginTop: "var(--space-1)",
        }}
      >
        {node.name}
      </div>
      <p
        style={{
          fontSize: "var(--fs-sm)",
          color: "var(--ink-secondary)",
          margin: "var(--space-3) 0",
          lineHeight: 1.5,
        }}
      >
        {node.key_idea}
      </p>
      <div style={{ display: "flex", gap: "var(--space-3)", pointerEvents: "auto" }}>
        <Link
          to={`/families/${node.family}`}
          style={{ fontSize: "var(--fs-sm)" }}
        >
          → 进入家族
        </Link>
        <Link
          to={`/families/${node.family}/${nodeSlug}`}
          style={{ fontSize: "var(--fs-sm)" }}
        >
          → 节点详情
        </Link>
      </div>
    </motion.div>
  );
}
