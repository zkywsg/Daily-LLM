import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LORA_SOURCE_PATH } from "../lib/prose";
import { InitDiagramSVG } from "../widgets/InitDiagramSVG";
import { TrainStepSlider } from "../widgets/TrainStepSlider";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function InitAndParamAccountStage({ mechanism2Prose }: Props) {
  const [step, setStep] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:B=0 + A Kaiming 初始化
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        如果 B 和 A 都随机初始化,ΔW = BA 在 step 0 就不为 0,模型行为
        立刻变形 —— 工业上不能接受。LoRA 的做法是 B = 0、A = Kaiming:
        ΔW(step 0) = 0,挂上 adapter 那一刻模型行为完全等于 W₀,
        然后训练让 B 渐进偏离 0。
      </p>

      <InitDiagramSVG step={step} />
      <p className={styles.caption}>
        ↑ 把 slider 拖到 0 看初始化:B 全 0、ΔW=0,接上 LoRA 不影响推理。
        往右拖看 B 渐填充、ΔW 渐生效 —— 训练过程就是这条路径。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={LORA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <TrainStepSlider step={step} onChange={setStep} />
        </div>
      </div>
    </div>
  );
}
