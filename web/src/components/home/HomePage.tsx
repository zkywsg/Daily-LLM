import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { FamiliesData } from "../../types/family";
import familiesJson from "../../data/families.json";
import { TimeAxisView } from "./TimeAxisView";
import { FamilyGridView } from "./FamilyGridView";
import { fadeIn, duration, ease } from "../../lib/motion";
import styles from "./HomePage.module.css";

const data = familiesJson as unknown as FamiliesData;

type Mode = "time" | "family";

export function HomePage() {
  const [mode, setMode] = useState<Mode>("time");

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>被逼出来的历史</h1>
        <p className={styles.subtitle}>
          深度学习与大模型 · 1998–2025 · {data.families.length} 家族
        </p>
        <div className={styles.toggle}>
          <button
            className={`${styles.toggleBtn} ${
              mode === "time" ? styles.toggleBtnActive : ""
            }`}
            onClick={() => setMode("time")}
          >
            按时间
          </button>
          <button
            className={`${styles.toggleBtn} ${
              mode === "family" ? styles.toggleBtnActive : ""
            }`}
            onClick={() => setMode("family")}
          >
            按家族
          </button>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={mode}
          variants={fadeIn}
          initial="initial"
          animate="animate"
          exit="exit"
          transition={{ duration: duration.base, ease: ease.out as unknown as number[] }}
        >
          {mode === "time" ? (
            <TimeAxisView data={data} />
          ) : (
            <FamilyGridView data={data} />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
