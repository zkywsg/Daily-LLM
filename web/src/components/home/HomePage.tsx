import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { FamiliesData } from "../../types/family";
import familiesJson from "../../data/families.json";
import { TimeAxisView } from "./TimeAxisView";
import { FamilyGridView } from "./FamilyGridView";
import { fadeIn, duration, ease } from "../../lib/motion";
import styles from "./HomePage.module.css";

const data = familiesJson as unknown as FamiliesData;

type Mode = "time" | "family";
const MODE_STORAGE_KEY = "daily-llm.homeMode";

// 从数据推导年份范围和节点总数,避免硬编码漂移
const allYears = data.families.flatMap((f) => f.nodes.map((n) => n.year));
const minYear = allYears.length > 0 ? Math.min(...allYears) : 0;
const maxYear = allYears.length > 0 ? Math.max(...allYears) : 0;
const totalNodes = data.families.reduce((sum, f) => sum + f.nodes.length, 0);

function readStoredMode(): Mode {
  try {
    const v = window.localStorage.getItem(MODE_STORAGE_KEY);
    return v === "family" ? "family" : "time";
  } catch {
    // SSR / 受限环境 / 老 jsdom 没 localStorage 时回退默认
    return "time";
  }
}

export function HomePage() {
  const [mode, setMode] = useState<Mode>(readStoredMode);

  // mode 变化时落 localStorage,刷新后还能回到上次选项
  useEffect(() => {
    try {
      window.localStorage.setItem(MODE_STORAGE_KEY, mode);
    } catch {
      // localStorage 不可用(隐私模式 / iframe 等)— 静默忽略
    }
  }, [mode]);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>深度学习与大模型演化路径</h1>
        <p className={styles.subtitle}>
          {minYear}–{maxYear} · {data.families.length} 家族 · {totalNodes} 节点
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
