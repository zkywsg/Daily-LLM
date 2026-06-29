import { Link } from "react-router";

import alexnetMarkdown from "../../../../../../01-cnn/02-alexnet.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, ALEXNET_SOURCE_PATH } from "./lib/prose";
import { ReluStage } from "./stages/ReluStage";
import { DropoutStage } from "./stages/DropoutStage";
import { GpuAugStage } from "./stages/GpuAugStage";
import { ImageNetTimeline } from "./widgets/ImageNetTimeline";
import styles from "./NodePageAlexNet.module.css";

const prose = extractProse(alexnetMarkdown);

export default function NodePageAlexNet() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/01-cnn" className={styles.back}>
          ← 返回 CNN 卷积神经网络
        </Link>
        <h1 className={styles.title}>AlexNet (2012)</h1>
        <div className={styles.metaLine}>
          作者:Alex Krizhevsky · Ilya Sutskever · Geoffrey Hinton · University of Toronto
        </div>
        <div className={styles.metaLine}>
          论文:ImageNet Classification with Deep Convolutional Neural Networks
        </div>
        <p className={styles.keyIdea}>
          ReLU + Dropout + 双 GPU + 数据增强,把 1989 年就存在的 CNN 第一次推到 ImageNet 1000 类
          规模 — Top-5 错误率 15.3% 比第二名领先 10 个百分点,深度学习时代的起点
        </p>
      </section>

      <section className={styles.stage}>
        <ReluStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DropoutStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <GpuAugStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>ImageNet 历年错误率</h2>
          <ImageNetTimeline />
          <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)", lineHeight: 1.5 }}>
            ↑ AlexNet 把 Top-5 从 25.8% 干到 16.4%(一夜降 10 个点);
            3 年后 ResNet-152 跨过人类基线(5.1%)。手工特征时代戛然而止。
          </p>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={ALEXNET_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={ALEXNET_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={ALEXNET_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
