import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DDPM_SOURCE_PATH } from "../lib/prose";
import type { Schedule } from "../lib/math";
import { NoiseStrip } from "../widgets/NoiseStrip";
import { ScheduleCurve } from "../widgets/ScheduleCurve";
import { ForwardControls } from "../widgets/ForwardControls";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function ForwardProcessStage({
  intuitionProse,
  mechanism1Prose,
}: Props) {
  const [T, setT] = useState(1000);
  const [t, setT_t] = useState(500);
  const [schedule, setSchedule] = useState<Schedule>("linear");

  // T 改变时 clamp t,避免越界
  const handleTChange = (newT: number) => {
    setT(newT);
    if (t >= newT) setT_t(newT - 1);
  };

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Forward Process — 固定的高斯加噪 Markov 链
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        从干净 x_0 出发,每步加一点高斯噪声,T 步之后变成纯噪声。
        关键是这一过程无参数 —— 选定 β 序列后,任意 x_t 都能从
        x_0 一步采样,无需走 Markov 链。
      </p>

      <NoiseStrip T={T} t={t} schedule={schedule} />
      <p className={styles.caption}>
        ↑ 拖时间 slider 看一维信号被噪声覆盖的全过程。粉色实线 = 当前 x_t,
        虚线 = 原始 x_0,背景色块表示像素值大小。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer
              markdown={intuitionProse}
              sourcePath={DDPM_SOURCE_PATH}
            />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer
              markdown={mechanism1Prose}
              sourcePath={DDPM_SOURCE_PATH}
            />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ScheduleCurve T={T} t={t} schedule={schedule} />
          <p className={styles.caption}>
            β_t 是每步加的方差;ᾱ_t 是累积后保留信号的比例;SNR(t) 是
            信噪比(log 刻度,右轴)。试试切 cosine vs linear 看尾部
            ᾱ 的差异 —— cosine 让靠近 T 的步保留更多结构,后来的图像
            扩散模型都用它。
          </p>
          <div style={{ marginTop: "var(--space-4)" }}>
            <ForwardControls
              T={T}
              onTChange={handleTChange}
              t={t}
              onTimeChange={setT_t}
              schedule={schedule}
              onScheduleChange={setSchedule}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
