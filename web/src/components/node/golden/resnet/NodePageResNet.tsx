import { useState } from "react";
import { Link } from "react-router";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github.css";

import resnetMarkdown from "../../../../../../01-cnn/05-resnet.md?raw";
import { extractProse } from "./lib/prose";
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
  const [stackDepth, setStackDepth] = useState(6);

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
          beforeStuckOnProse={prose.beforeStuckOn}
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
        />
      </section>

      <section className={styles.stage}>
        <GradientHighwayStage
          mechanismProse={prose.mechanism}
          stackDepth={stackDepth}
          onStackDepthChange={setStackDepth}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeHighlight, rehypeKatex]}
          >
            {prose.trainingDetails}
          </ReactMarkdown>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeHighlight, rehypeKatex]}
          >
            {prose.keyCode}
          </ReactMarkdown>
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {prose.aftermath}
          </ReactMarkdown>
        </div>
      </section>
    </div>
  );
}
