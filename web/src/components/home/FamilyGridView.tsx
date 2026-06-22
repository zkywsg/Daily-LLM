import { motion } from "framer-motion";
import { Link } from "react-router";
import type { FamiliesData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { staggerContainer, fadeUp, duration, ease } from "../../lib/motion";
import styles from "./FamilyGridView.module.css";

interface FamilyGridViewProps {
  data: FamiliesData;
}

export function FamilyGridView({ data }: FamilyGridViewProps) {
  return (
    <motion.div
      className={styles.grid}
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      {data.families.map((f) => (
        <motion.div
          key={f.id}
          variants={fadeUp}
          transition={{ duration: duration.base, ease: ease.out }}
        >
          <Link
            to={`/families/${f.id}`}
            className={styles.card}
            style={{ borderTopColor: familyColorVar(f.id) }}
          >
            <div className={styles.cardId}>{f.id}</div>
            <h3 className={styles.cardTitle}>{f.label}</h3>
            <p className={styles.cardBlurb}>{f.blurb}</p>
            <div className={styles.cardMeta}>
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
