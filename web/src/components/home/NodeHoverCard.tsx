import { motion } from "framer-motion";
import { Link } from "react-router";
import type { NodeData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { popScale, duration, ease } from "../../lib/motion";
import { getMiniArch } from "../mini-arches/getMiniArch";
import styles from "./NodeHoverCard.module.css";

interface NodeHoverCardProps {
  node: NodeData;
  x: number;
  y: number;
}

export function NodeHoverCard({ node, x, y }: NodeHoverCardProps) {
  const nodeSlug = node.path.split("/").pop()!.replace(/\.md$/, "");
  const MiniArch = getMiniArch(node.path);

  return (
    <motion.div
      className={styles.card}
      variants={popScale}
      initial="initial"
      animate={{ opacity: 1, scale: 1, x, y }}
      exit="exit"
      transition={{ duration: duration.fast, ease: ease.out }}
      style={{
        left: 0,
        top: 0,
        transformOrigin: "top left",
        borderColor: familyColorVar(node.family),
      }}
    >
      {MiniArch && (
        <div className={styles.archThumb} aria-hidden="true">
          <MiniArch width={260} height={70} />
        </div>
      )}
      <div className={styles.year}>
        {node.year} · {node.family}
      </div>
      <div className={styles.name}>{node.name}</div>
      <p className={styles.idea}>{node.key_idea}</p>
      <div className={styles.actions}>
        <Link to={`/families/${node.family}`} className={styles.actionLink}>
          → 进入家族
        </Link>
        <Link
          to={`/families/${node.family}/${nodeSlug}`}
          className={styles.actionLink}
        >
          → 节点详情
        </Link>
      </div>
    </motion.div>
  );
}
