import { useState } from "react";
import { Link } from "react-router";

import resnetMarkdown from "../../../../../../01-cnn/05-resnet.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, RESNET_SOURCE_PATH } from "./lib/prose";
import { DegradationStage } from "./stages/DegradationStage";
import { ResidualBlockStage } from "./stages/ResidualBlockStage";
import { GradientHighwayStage } from "./stages/GradientHighwayStage";
import styles from "./NodePageResNet.module.css";

const prose = extractProse(resnetMarkdown);

export default function NodePageResNet() {
  const [depth, setDepth] = useState(56);
  const [blockType, setBlockType] = useState<"basic" | "bottleneck">(
    "bottleneck"
  );
  const [showShortcut, setShowShortcut] = useState(true);
  const [stackDepth, setStackDepth] = useState(6);
  const [highwayShortcut, setHighwayShortcut] = useState(true);

  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/01-cnn" className={styles.back}>
          ← 返回 CNN 卷积神经网络
        </Link>
        <h1 className={styles.title}>ResNet (2015)</h1>
        <div className={styles.metaLine}>
          作者：Kaiming He · Xiangyu Zhang · Shaoqing Ren · Jian Sun
        </div>
        <div className={styles.metaLine}>
          论文：Deep Residual Learning for Image Recognition
        </div>
        <p className={styles.keyIdea}>
          用 shortcut 让网络只学残差修正而不是从零重建映射，把 152 层稳定训练变成可能
        </p>
      </section>

      <section className={styles.stage}>
        <DegradationStage
          previousWorkProse={prose.previousWork}
          coreInsightProse={prose.coreInsight}
          depth={depth}
          onDepthChange={setDepth}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ResidualBlockStage
          intuitionProse={prose.intuition}
          blockType={blockType}
          onBlockTypeChange={setBlockType}
          showShortcut={showShortcut}
          onShowShortcutChange={setShowShortcut}
        />
      </section>

      <section className={styles.stage}>
        <GradientHighwayStage
          mechanismProse={prose.mechanism}
          stackDepth={stackDepth}
          onStackDepthChange={setStackDepth}
          highwayShortcut={highwayShortcut}
          onHighwayShortcutChange={setHighwayShortcut}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer
            markdown={prose.trainingDetails}
            sourcePath={RESNET_SOURCE_PATH}
          />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer
            markdown={prose.keyCode}
            sourcePath={RESNET_SOURCE_PATH}
          />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer
            markdown={prose.aftermath}
            sourcePath={RESNET_SOURCE_PATH}
          />
        </div>
      </section>
    </div>
  );
}
