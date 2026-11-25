import { motion } from "framer-motion";
import { Link } from "react-router";
import type { FamiliesData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { staggerContainer, fadeUp, duration, ease } from "../../lib/motion";
import { getMiniArch } from "../mini-arches/getMiniArch";
import { FAMILY_HERO } from "./familyHero";
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
      {data.families.map((f) => {
        const heroPath = FAMILY_HERO[f.id];
        const HeroArch = heroPath ? getMiniArch(heroPath) : null;
        const heroNode = heroPath
          ? f.nodes.find((n) => n.path === heroPath)
          : null;
        return (
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
              {HeroArch && (
                <div
                  className={styles.heroArch}
                  aria-hidden="true"
                  title={heroNode ? `代表作:${heroNode.name}` : undefined}
                >
                  <HeroArch width={220} height={70} />
                </div>
              )}
              <div className={styles.cardId}>{f.id}</div>
              <h3 className={styles.cardTitle}>{f.label}</h3>
              <p className={styles.cardBlurb}>{f.blurb}</p>
              <div className={styles.cardMeta}>
                {f.nodes.length > 0
                  ? `${f.nodes.length} 节点 · ${f.yearRange?.[0]}–${f.yearRange?.[1]}`
                  : "待补充"}
                {heroNode && (
                  <span className={styles.cardHeroLabel}>
                    {" "}· 代表作 {heroNode.name}
                  </span>
                )}
              </div>
            </Link>
          </motion.div>
        );
      })}
    </motion.div>
  );
}
