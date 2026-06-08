import { motion } from "framer-motion";
import { Link } from "react-router";
import type { FamiliesData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { staggerContainer, fadeUp, duration, ease } from "../../lib/motion";

interface FamilyGridViewProps {
  data: FamiliesData;
}

export function FamilyGridView({ data }: FamilyGridViewProps) {
  return (
    <motion.div
      variants={staggerContainer}
      initial="initial"
      animate="animate"
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
        gap: "var(--space-6)",
      }}
    >
      {data.families.map((f) => (
        <motion.div
          key={f.id}
          variants={fadeUp}
          transition={{ duration: duration.base, ease: ease.out as unknown as number[] }}
        >
          <Link
            to={`/families/${f.id}`}
            style={{
              display: "block",
              padding: "var(--space-6)",
              background: "var(--bg-surface)",
              border: "1px solid var(--border)",
              borderTop: `4px solid ${familyColorVar(f.id)}`,
              borderRadius: "var(--radius-md)",
              transition: `transform var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out)`,
              color: "var(--ink-primary)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-4px)";
              e.currentTarget.style.boxShadow = "var(--shadow-md)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "";
              e.currentTarget.style.boxShadow = "";
            }}
          >
            <div
              style={{
                fontSize: "var(--fs-sm)",
                color: "var(--ink-muted)",
                marginBottom: "var(--space-2)",
              }}
            >
              {f.id}
            </div>
            <h3
              style={{
                fontSize: "var(--fs-lg)",
                marginBottom: "var(--space-3)",
              }}
            >
              {f.label}
            </h3>
            <p
              style={{
                fontSize: "var(--fs-sm)",
                color: "var(--ink-secondary)",
                lineHeight: 1.5,
                marginBottom: "var(--space-4)",
              }}
            >
              {f.blurb}
            </p>
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
              }}
            >
              {f.nodes.length > 0
                ? `${f.nodes.length} 节点 · ${f.yearRange?.[0]}–${f.yearRange?.[1]}`
                : "待补充"}
            </div>
          </Link>
        </motion.div>
      ))}
    </motion.div>
  );
}
