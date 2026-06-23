import { Link } from "react-router";

import transformerMarkdown from "../../../../../../05-transformer/01-transformer.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, TRANSFORMER_SOURCE_PATH } from "./lib/prose";
import { ScaledDotProductStage } from "./stages/ScaledDotProductStage";
import { MultiHeadStage } from "./stages/MultiHeadStage";
import { PositionEncodingStage } from "./stages/PositionEncodingStage";
import styles from "./NodePageTransformer.module.css";

const prose = extractProse(transformerMarkdown);

export default function NodePageTransformer() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/05-transformer" className={styles.back}>
          ← 返回 Transformer 架构
        </Link>
        <h1 className={styles.title}>Transformer (2017)</h1>
        <div className={styles.metaLine}>
          作者:Vaswani et al. · Google Brain / Google Research
        </div>
        <div className={styles.metaLine}>
          论文:Attention Is All You Need
        </div>
        <p className={styles.keyIdea}>
          扔掉循环 + 卷积,把 token 之间的关系全交给 Q/K/V
          自注意力,一次矩阵乘解决依赖,大模型时代由此开局
        </p>
      </section>

      <section className={styles.stage}>
        <ScaledDotProductStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <MultiHeadStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PositionEncodingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        {prose.encoderDecoder && (
          <div className={styles.footerSection}>
            <h2>完整 encoder / decoder</h2>
            <MarkdownRenderer
              markdown={prose.encoderDecoder}
              sourcePath={TRANSFORMER_SOURCE_PATH}
            />
          </div>
        )}
        {prose.postLN && (
          <div className={styles.footerSection}>
            <h2>Post-LN 原版细节</h2>
            <MarkdownRenderer
              markdown={prose.postLN}
              sourcePath={TRANSFORMER_SOURCE_PATH}
            />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer
            markdown={prose.trainingDetails}
            sourcePath={TRANSFORMER_SOURCE_PATH}
          />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer
            markdown={prose.keyCode}
            sourcePath={TRANSFORMER_SOURCE_PATH}
          />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer
            markdown={prose.aftermath}
            sourcePath={TRANSFORMER_SOURCE_PATH}
          />
        </div>
      </section>
    </div>
  );
}
