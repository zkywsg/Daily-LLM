import { Link } from "react-router";

import gpt4LlamaMarkdown from "../../../../../../07-gpt-scaling/05-gpt4-llama.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GPT4_LLAMA_SOURCE_PATH } from "./lib/prose";
import { Gpt4Stage } from "./stages/Gpt4Stage";
import { LlamaStage } from "./stages/LlamaStage";
import { RecipeStage } from "./stages/RecipeStage";
import { GapNarrowingChart } from "./widgets/GapNarrowingChart";
import styles from "./NodePageGPT4LLaMA.module.css";

const prose = extractProse(gpt4LlamaMarkdown);

export default function NodePageGPT4LLaMA() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/07-gpt-scaling" className={styles.back}>
          ← 返回 GPT Scaling
        </Link>
        <h1 className={styles.title}>GPT-4 / LLaMA (2023)</h1>
        <div className={styles.metaLine}>
          作者:OpenAI · Hugo Touvron · Thibaut Lavril · Gautier Izacard · Xavier Martinet · Marie-Anne Lachaux et al.(Meta)
        </div>
        <div className={styles.metaLine}>
          论文:GPT-4 Technical Report / LLaMA: Open and Efficient Foundation Language Models
        </div>
        <p className={styles.keyIdea}>
          GPT-4 把 LLM 推到万亿级 + 多模态闭源;LLaMA 给社区第一个工业级
          开源基础模型;现代 LLM 配方(Pre-RMSNorm + RoPE + GQA + SwiGLU)
          在两者上同时定型 — 2023 是 LLM 的"配方定型年"
        </p>
      </section>

      <section className={styles.stage}>
        <Gpt4Stage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <LlamaStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <RecipeStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GPT4_LLAMA_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <GapNarrowingChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.aftermath} sourcePath={GPT4_LLAMA_SOURCE_PATH} />
          </div>
        </div>
      </section>
    </div>
  );
}
