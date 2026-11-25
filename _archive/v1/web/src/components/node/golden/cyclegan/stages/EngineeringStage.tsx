import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CYCLEGAN_SOURCE_PATH } from "../lib/prose";
import { ENGINEERING_STABILIZERS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function EngineeringStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:PatchGAN + LSGAN + ResNet G — 让 4 网络同时训能跑通
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        CycleGAN 同时训 G、F、D_X、D_Y 四个网络,工程复杂度高。论文用三个工程稳定剂让训练真正能收敛:
        70×70 局部判别的 PatchGAN、用 MSE 替代 BCE 的 LSGAN loss、以及 9 个 residual block 的 ResNet Generator。
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16, marginTop: "var(--space-6)" }}>
        {ENGINEERING_STABILIZERS.map((row) => (
          <div
            key={row.name}
            style={{
              padding: "var(--space-4)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
            }}
          >
            <div style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: "var(--ink-primary)", marginBottom: 6 }}>
              {row.name}
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-secondary)", lineHeight: 1.6 }}>
              {row.role}
            </div>
          </div>
        ))}
      </div>
      <p className={styles.caption}>
        ↑ 三个稳定剂各自解决"4 网络同时训"里的一类工程问题:D 太重、梯度饱和、G 生成质量差。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={CYCLEGAN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={CYCLEGAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任一件会怎样
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.8, color: "var(--ink-secondary)" }}>
              <div>· 只有双向 G/D → 模式塌缩</div>
              <div>· 只有 cycle loss + 单边 G → F(G(x)) 无定义</div>
              <div>· 没有工程稳定剂 → 4 网络训练崩溃</div>
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              三件套合起来才让"无配对图像翻译"从概念变成可工程化的方法。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
