import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LDM_SOURCE_PATH } from "../lib/prose";
import { LatentDiffusionPipeline } from "../widgets/LatentDiffusionPipeline";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

const TOTAL_STEPS = 50;

export function LatentDiffusionStage({ mechanism2Prose }: Props) {
  const [step, setStep] = useState(25);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Latent Diffusion — 在 latent 上跑标准 DDPM
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Stage 2 完全复用 DDPM 的全套数学:ε-prediction 参数化 / linear schedule /
        DDIM 快采样 / CFG guidance。唯一改动是 U-Net 输入从 3×512×512 → 4×64×64,
        显存从 14G → 2G。LDM 没发明新 diffusion 理论 — 它发明的是搬动。
      </p>

      <LatentDiffusionPipeline step={step} totalSteps={TOTAL_STEPS} />
      <p className={styles.caption}>
        ↑ 拖 slider 看 50 步 DDIM 采样:从 z_T 纯噪声开始,每步 U-Net 预测 ε 并去掉,
        到 z_0 时已是清晰 latent。整条 diffusion 全程在 latent 空间,
        VAE.decode 只在最后一步把 latent 还原回像素。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={LDM_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: 4 }}>
              <span>采样步数</span><strong>{step} / {TOTAL_STEPS}</strong>
            </label>
            <input type="range" min={0} max={TOTAL_STEPS} step={1} value={step} onChange={(e) => setStep(parseInt(e.target.value))} style={{ width: "100%" }} />
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 8, lineHeight: 1.5 }}>
              <p style={{ margin: "0 0 6px" }}>
                <strong>训练 timestep T = 1000</strong>,推理用 DDIM 跳采样到 50 步就够 — 比 DDPM 快 20×。
              </p>
              <p style={{ margin: "0 0 6px" }}>
                <strong>* 0.18215 magic number</strong> — SD VAE 的 latent 标准化常数,把 latent 尺度对齐到 noise schedule。
              </p>
              <p style={{ margin: 0 }}>
                <strong>训练 loss</strong>: MSE(ε_pred, ε_true) — 跟 DDPM 字面相同,只是输入是 latent。
              </p>
            </div>
          </div>

          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4, fontWeight: 600 }}>训练循环 — 完全和 DDPM 同构</div>
            <pre style={{ fontSize: 10, lineHeight: 1.4, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`z_0 = vae.encode(x) * 0.18215
t   = randint(0, 1000)
ε   = randn_like(z_0)
z_t = scheduler.add_noise(z_0, ε, t)
ε̂   = unet(z_t, t, text_emb)
loss = mse(ε̂, ε)`}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
