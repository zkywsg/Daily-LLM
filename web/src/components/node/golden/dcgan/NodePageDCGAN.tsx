import { Link } from "react-router";

import dcganMarkdown from "../../../../../../04-gan/02-dcgan.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DCGAN_SOURCE_PATH } from "./lib/prose";
import { AllConvStage } from "./stages/AllConvStage";
import { StabilizersStage } from "./stages/StabilizersStage";
import { GuidelineStage } from "./stages/GuidelineStage";
import { Cifar10CompareChart } from "./widgets/Cifar10CompareChart";
import styles from "./NodePageDCGAN.module.css";

const prose = extractProse(dcganMarkdown);

export default function NodePageDCGAN() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/04-gan" className={styles.back}>
          ← 返回 GAN
        </Link>
        <h1 className={styles.title}>DCGAN (2015)</h1>
        <div className={styles.metaLine}>
          作者:Alec Radford · Luke Metz · Soumith Chintala
        </div>
        <div className={styles.metaLine}>
          论文:Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
        </div>
        <p className={styles.keyIdea}>
          把 CNN 完整移植到 GAN——用 strided conv 替代 pooling、加 BatchNorm、
          Generator 用 transpose conv 上采样、去全连接层;首次给出可复现的
          GAN 训练工程方案,生成 64×64 卧室 / 人脸图像
        </p>
      </section>

      <section className={styles.stage}>
        <AllConvStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <StabilizersStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <GuidelineStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <Cifar10CompareChart />
          <MarkdownRenderer markdown={prose.performance} sourcePath={DCGAN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DCGAN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DCGAN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
