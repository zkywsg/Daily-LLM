import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ALEXNET_SOURCE_PATH } from "../lib/prose";
import { DualGpuSplit } from "../widgets/DualGpuSplit";
import { DataAugCrops } from "../widgets/DataAugCrops";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function GpuAugStage({ mechanism3Prose, synergyProse }: Props) {
  const [cropX, setCropX] = useState(16);
  const [cropY, setCropY] = useState(16);
  const [flipped, setFlipped] = useState(false);
  const [colorJitter, setColorJitter] = useState(0.5);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:GPU + 数据增强 — 让训练 5 天跑完
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        60M 参数 × 1.2M 图 × 90 epoch 在 CPU 上要几个月。Krizhevsky 把通道维
        劈成两半放到两块 GTX 580(各 3 GB)上,加自己写的 CUDA kernel
        把训练压到 5 天;同时用 random crop / 水平翻转 / PCA 颜色扰动
        把每张图扩展成 ~2048 个变体,有效训练集 ≈ 25 亿样本。
      </p>

      <DualGpuSplit />
      <p className={styles.caption}>
        ↑ 上下两条 lane = 两块 GPU。大部分层各 GPU 独立(粉/蓝);
        只在 conv3 / fc 跨 GPU 通信(黄)— 是 "group convolution" 的雏形。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={ALEXNET_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={ALEXNET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <DataAugCrops cropX={cropX} cropY={cropY} flipped={flipped} colorJitter={colorJitter} />
          <p className={styles.caption}>
            ↑ 拖 slider 看 224 crop 在 256 原图里随机滑动,加水平翻转 + 颜色扰动。
            每张图理论上有 32² × 2 = 2048 种变体。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: 4 }}>
              <span>crop X</span><strong>{cropX}</strong>
            </label>
            <input type="range" min={0} max={32} step={1} value={cropX} onChange={(e) => setCropX(parseInt(e.target.value))} style={{ width: "100%" }} />
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "6px 0 4px" }}>
              <span>crop Y</span><strong>{cropY}</strong>
            </label>
            <input type="range" min={0} max={32} step={1} value={cropY} onChange={(e) => setCropY(parseInt(e.target.value))} style={{ width: "100%" }} />
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "6px 0 4px" }}>
              <span>PCA 颜色扰动</span><strong>{((colorJitter - 0.5) * 2).toFixed(2)}</strong>
            </label>
            <input type="range" min={0} max={1} step={0.05} value={colorJitter} onChange={(e) => setColorJitter(parseFloat(e.target.value))} style={{ width: "100%" }} />
            <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8, fontSize: "var(--fs-sm)", color: "var(--ink-secondary)" }}>
              <input type="checkbox" checked={flipped} onChange={(e) => setFlipped(e.target.checked)} />
              <span>水平翻转</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}
