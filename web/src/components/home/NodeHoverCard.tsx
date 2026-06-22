import { useLayoutEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router";
import type { NodeData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { fadeUp, duration, ease } from "../../lib/motion";
import { getMiniArch } from "../mini-arches/getMiniArch";
import styles from "./NodeHoverCard.module.css";

interface NodeHoverCardProps {
  node: NodeData;
  x: number;
  y: number;
}

const VIEWPORT_MARGIN = 8;

export function NodeHoverCard({ node, x, y }: NodeHoverCardProps) {
  const nodeSlug = node.path.split("/").pop()!.replace(/\.md$/, "");
  const MiniArch = getMiniArch(node.path);
  const cardRef = useRef<HTMLDivElement>(null);
  const [clamped, setClamped] = useState({ x, y });

  // 测量后把卡 clamp 进视口,避免轴左 / 顶边的节点把卡推出屏幕
  useLayoutEffect(() => {
    const card = cardRef.current;
    if (!card) {
      setClamped({ x, y });
      return;
    }
    const rect = card.getBoundingClientRect();
    const docW = document.documentElement.clientWidth + window.scrollX;
    const docH = document.documentElement.clientHeight + window.scrollY;
    const maxX = docW - rect.width - VIEWPORT_MARGIN;
    const maxY = docH - rect.height - VIEWPORT_MARGIN;
    setClamped({
      x: Math.min(Math.max(VIEWPORT_MARGIN + window.scrollX, x), maxX),
      y: Math.min(Math.max(VIEWPORT_MARGIN + window.scrollY, y), maxY),
    });
  }, [x, y, node.path]);

  return (
    <motion.div
      ref={cardRef}
      className={styles.card}
      variants={fadeUp}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={{ duration: duration.fast, ease: ease.out }}
      style={{
        left: clamped.x,
        top: clamped.y,
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
