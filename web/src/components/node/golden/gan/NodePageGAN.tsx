import { Link } from "react-router";

import ganMarkdown from "../../../../../../04-gan/01-gan.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GAN_SOURCE_PATH } from "./lib/prose";
import { GeneratorStage } from "./stages/GeneratorStage";
import { DiscriminatorStage } from "./stages/DiscriminatorStage";
import { MinimaxStage } from "./stages/MinimaxStage";
import styles from "./NodePageGAN.module.css";

const prose = extractProse(ganMarkdown);

export default function NodePageGAN() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/04-gan" className={styles.back}>
          ← 返回 GAN 生成对抗
        </Link>
        <h1 className={styles.title}>GAN (2014)</h1>
        <div className={styles.metaLine}>
          作者:Ian Goodfellow · Jean Pouget-Abadie · Mehdi Mirza · ... · Yoshua Bengio · Université de Montréal
        </div>
        <div className={styles.metaLine}>
          论文:Generative Adversarial Networks
        </div>
        <p className={styles.keyIdea}>
          让 G 和 D 互相博弈 — G 学造假、D 学辨真,
          均衡时 G 输出和真分布完全一致 — 把"生成是否真实"这件事显化成可微目标
        </p>
      </section>

      <section className={styles.stage}>
        <GeneratorStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DiscriminatorStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <MinimaxStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GAN_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GAN_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GAN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
