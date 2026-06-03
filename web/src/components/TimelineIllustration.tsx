import type * as React from "react";

/**
 * 按年份渲染的简笔示意图（SVG + CSS 动画）。
 * 目标：在「发生了什么」叙事旁边，给一个 5 秒就能看懂的结构动图。
 * 暂未覆盖的年份返回 null（content-card 自然不会留白）。
 */

type TimelineIllustrationProps = {
  year: string;
};

export function TimelineIllustration({ year }: TimelineIllustrationProps) {
  const renderer = ILLUSTRATIONS[year];
  if (!renderer) return null;

  return (
    <figure className="illustration" aria-label={renderer.caption}>
      {renderer.render()}
      <figcaption>{renderer.caption}</figcaption>
    </figure>
  );
}

type Renderer = { caption: string; render: () => React.ReactElement };

const ILLUSTRATIONS: Record<string, Renderer> = {
  "2012": {
    caption: "AlexNet 结构示意：像素 → 卷积/池化层层抽象 → 全连接 → 分类概率",
    render: () => <CnnDiagram />,
  },
  "2014": {
    caption: "GAN 对抗示意：生成器与判别器互相博弈，把噪声推向真实样本分布",
    render: () => <GanDiagram />,
  },
  "2015": {
    caption: "ResNet 残差连接：恒等捷径让梯度直接跨越多层，深网终于训得动",
    render: () => <ResNetDiagram />,
  },
  "2017": {
    caption: "Transformer 注意力：每个 token 对所有其他 token 计算权重，并行而非递归",
    render: () => <AttentionDiagram />,
  },
  "2020": {
    caption: "Scaling Law：参数量级跃迁后，少样本/零样本能力随规模浮现",
    render: () => <ScalingDiagram />,
  },
  "2021": {
    caption: "CLIP 双塔：图像与文本编码到同一空间，匹配靠余弦相似度",
    render: () => <ClipDiagram />,
  },
};

/* =====================================================================
 * 2012 — CNN
 * ===================================================================== */
function CnnDiagram() {
  return (
    <svg viewBox="0 0 720 180" role="img" className="illustration__svg">
      {/* 输入图像格子 */}
      <g transform="translate(20,30)">
        <text x="0" y="-8" className="illustration__label">
          input 224²
        </text>
        {Array.from({ length: 6 }).map((_, r) =>
          Array.from({ length: 6 }).map((_, c) => (
            <rect
              key={`p-${r}-${c}`}
              x={c * 14}
              y={r * 14}
              width="12"
              height="12"
              rx="1.5"
              className="illustration__pixel"
              style={{ animationDelay: `${(r + c) * 70}ms` }}
            />
          )),
        )}
      </g>

      {/* 卷积特征图：3 组方块叠加，逐渐变小 */}
      {[
        { x: 160, size: 70, n: 3, label: "conv₁ + pool" },
        { x: 290, size: 54, n: 5, label: "conv₃ + pool" },
        { x: 410, size: 36, n: 7, label: "conv₅ + pool" },
      ].map((stage, i) => (
        <g key={`stage-${i}`} transform={`translate(${stage.x},40)`}>
          <text x="0" y="-8" className="illustration__label">
            {stage.label}
          </text>
          {Array.from({ length: stage.n }).map((_, k) => (
            <rect
              key={`s-${i}-${k}`}
              x={k * 5}
              y={k * 5}
              width={stage.size}
              height={stage.size}
              rx="3"
              className="illustration__featuremap"
              style={{
                animationDelay: `${i * 180 + k * 60}ms`,
              }}
            />
          ))}
        </g>
      ))}

      {/* 全连接层 */}
      <g transform="translate(530,40)">
        <text x="0" y="-8" className="illustration__label">
          FC
        </text>
        {Array.from({ length: 8 }).map((_, i) => (
          <circle
            key={`fc-${i}`}
            cx="12"
            cy={i * 14 + 8}
            r="4"
            className="illustration__neuron"
            style={{ animationDelay: `${600 + i * 50}ms` }}
          />
        ))}
      </g>

      {/* 输出 softmax */}
      <g transform="translate(620,40)">
        <text x="0" y="-8" className="illustration__label">
          softmax
        </text>
        {["cat", "dog", "car", "…"].map((label, i) => (
          <g key={label}>
            <rect
              x="0"
              y={i * 22 + 4}
              width={i === 0 ? 70 : 30 + i * 8}
              height="14"
              rx="2"
              className="illustration__bar"
              style={{ animationDelay: `${900 + i * 80}ms` }}
            />
            <text
              x="74"
              y={i * 22 + 14}
              className="illustration__label illustration__label--inline"
            >
              {label}
            </text>
          </g>
        ))}
      </g>

      {/* 流向箭头（4 段虚线，循环涌动） */}
      {[100, 245, 370, 500].map((x, i) => (
        <line
          key={`arrow-${i}`}
          x1={x}
          y1="90"
          x2={x + 22}
          y2="90"
          className="illustration__flow"
          style={{ animationDelay: `${i * 250}ms` }}
        />
      ))}
    </svg>
  );
}

/* =====================================================================
 * 2014 — GAN
 * ===================================================================== */
function GanDiagram() {
  return (
    <svg viewBox="0 0 720 180" role="img" className="illustration__svg">
      <g transform="translate(40,40)">
        <text x="0" y="-10" className="illustration__label">
          noise z
        </text>
        {Array.from({ length: 9 }).map((_, i) => (
          <circle
            key={i}
            cx={(i % 3) * 20 + 10}
            cy={Math.floor(i / 3) * 20 + 10}
            r="5"
            className="illustration__neuron illustration__neuron--noise"
            style={{ animationDelay: `${i * 80}ms` }}
          />
        ))}
      </g>

      {/* Generator block */}
      <g transform="translate(160,30)">
        <rect width="120" height="100" rx="10" className="illustration__block" />
        <text x="60" y="58" className="illustration__block-label" textAnchor="middle">
          Generator
        </text>
      </g>

      {/* fake samples */}
      <g transform="translate(310,55)">
        <text x="0" y="-12" className="illustration__label">
          fake
        </text>
        {Array.from({ length: 3 }).map((_, i) => (
          <rect
            key={i}
            x="0"
            y={i * 22}
            width="60"
            height="16"
            rx="3"
            className="illustration__bar illustration__bar--fake"
            style={{ animationDelay: `${300 + i * 100}ms` }}
          />
        ))}
      </g>

      {/* real samples */}
      <g transform="translate(310,2)">
        <text x="0" y="-2" className="illustration__label">
          real
        </text>
        {Array.from({ length: 1 }).map((_, i) => (
          <rect
            key={i}
            x="0"
            y={i * 22}
            width="60"
            height="16"
            rx="3"
            className="illustration__bar illustration__bar--real"
          />
        ))}
      </g>

      {/* Discriminator */}
      <g transform="translate(420,30)">
        <rect width="140" height="100" rx="10" className="illustration__block illustration__block--alt" />
        <text x="70" y="58" className="illustration__block-label" textAnchor="middle">
          Discriminator
        </text>
      </g>

      {/* verdict */}
      <g transform="translate(590,68)">
        <circle r="20" className="illustration__verdict" />
        <text className="illustration__label" textAnchor="middle" y="5">
          real?
        </text>
      </g>

      {/* feedback loop */}
      <path
        d="M 590 100 Q 380 170 220 130"
        className="illustration__feedback"
        fill="none"
      />
      <text x="380" y="168" className="illustration__label" textAnchor="middle">
        gradient ←
      </text>
    </svg>
  );
}

/* =====================================================================
 * 2015 — ResNet skip connection
 * ===================================================================== */
function ResNetDiagram() {
  return (
    <svg viewBox="0 0 720 180" role="img" className="illustration__svg">
      <line x1="40" y1="100" x2="680" y2="100" className="illustration__rail" />
      {[
        { x: 90, label: "x" },
        { x: 240, label: "F₁(x)" },
        { x: 380, label: "F₂" },
        { x: 520, label: "F₃" },
        { x: 660, label: "y" },
      ].map((n, i) => (
        <g key={i} transform={`translate(${n.x},100)`}>
          <circle r="18" className="illustration__neuron illustration__neuron--big" />
          <text textAnchor="middle" y="5" className="illustration__block-label">
            {n.label}
          </text>
        </g>
      ))}
      {/* skip connections (arcs above the rail) */}
      {[
        { x1: 90, x2: 380 },
        { x1: 240, x2: 520 },
        { x1: 380, x2: 660 },
      ].map((arc, i) => (
        <path
          key={i}
          d={`M ${arc.x1} 82 Q ${(arc.x1 + arc.x2) / 2} 10 ${arc.x2} 82`}
          className="illustration__skip"
          style={{ animationDelay: `${i * 220}ms` }}
          fill="none"
        />
      ))}
      <text x="360" y="170" className="illustration__label" textAnchor="middle">
        H(x) = F(x) + x  · 恒等捷径让梯度跨层流动
      </text>
    </svg>
  );
}

/* =====================================================================
 * 2017 — Transformer attention
 * ===================================================================== */
function AttentionDiagram() {
  // 5x5 attention matrix with varying intensity
  const matrix = [
    [0.9, 0.1, 0.05, 0.02, 0.0],
    [0.3, 0.7, 0.2, 0.05, 0.02],
    [0.1, 0.4, 0.6, 0.3, 0.05],
    [0.05, 0.2, 0.5, 0.7, 0.3],
    [0.0, 0.05, 0.3, 0.5, 0.85],
  ];
  const tokens = ["The", "cat", "sat", "on", "mat"];

  return (
    <svg viewBox="0 0 720 200" role="img" className="illustration__svg">
      <g transform="translate(60,30)">
        {/* matrix cells */}
        {matrix.map((row, r) =>
          row.map((v, c) => (
            <rect
              key={`${r}-${c}`}
              x={c * 28}
              y={r * 28}
              width="26"
              height="26"
              rx="3"
              className="illustration__attn-cell"
              style={{
                opacity: 0.15 + v * 0.85,
                animationDelay: `${(r * 5 + c) * 40}ms`,
              }}
            />
          )),
        )}
        {/* column labels (Q) */}
        {tokens.map((t, i) => (
          <text key={`c-${i}`} x={i * 28 + 13} y={-8} className="illustration__label" textAnchor="middle">
            {t}
          </text>
        ))}
        {/* row labels (K) */}
        {tokens.map((t, i) => (
          <text key={`r-${i}`} x={-10} y={i * 28 + 18} className="illustration__label" textAnchor="end">
            {t}
          </text>
        ))}
      </g>

      {/* multi-head fanned-out hint */}
      <g transform="translate(260,40)">
        <text x="0" y="-10" className="illustration__label">
          multi-head: 8 个子空间并行
        </text>
        {Array.from({ length: 8 }).map((_, i) => (
          <rect
            key={i}
            x={i * 18}
            y={i * 6}
            width="60"
            height="18"
            rx="3"
            className="illustration__bar illustration__bar--head"
            style={{ animationDelay: `${i * 80}ms` }}
          />
        ))}
      </g>

      <text x="360" y="190" className="illustration__label" textAnchor="middle">
        每个 token 对所有 token 计算权重 —— 没有 RNN，序列长度不再卡脖子
      </text>
    </svg>
  );
}

/* =====================================================================
 * 2020 — Scaling law curve
 * ===================================================================== */
function ScalingDiagram() {
  // log-log power-law-ish curve
  const points = Array.from({ length: 30 }, (_, i) => {
    const x = 30 + i * 20;
    const y = 150 - Math.pow(i / 30, 0.55) * 110;
    return [x, y] as const;
  });
  const path = points.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ");

  return (
    <svg viewBox="0 0 720 180" role="img" className="illustration__svg">
      {/* axes */}
      <line x1="30" y1="150" x2="630" y2="150" className="illustration__rail" />
      <line x1="30" y1="20" x2="30" y2="150" className="illustration__rail" />
      <text x="630" y="170" className="illustration__label" textAnchor="end">
        参数量 →
      </text>
      <text x="20" y="20" className="illustration__label" textAnchor="end">
        ↑ 能力
      </text>

      {/* dots showing model sizes */}
      {[
        { x: 80, y: 138, l: "117M" },
        { x: 180, y: 110, l: "1.5B" },
        { x: 360, y: 70, l: "13B" },
        { x: 560, y: 38, l: "175B" },
      ].map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r="6" className="illustration__neuron illustration__neuron--big" />
          <text x={p.x + 12} y={p.y + 4} className="illustration__label">
            {p.l}
          </text>
        </g>
      ))}

      <path d={path} className="illustration__curve" fill="none" />

      {/* emergent ability hint */}
      <g transform="translate(420,30)">
        <rect width="180" height="40" rx="8" className="illustration__block" />
        <text x="90" y="25" className="illustration__block-label" textAnchor="middle">
          few-shot 涌现
        </text>
      </g>

      <text x="360" y="178" className="illustration__label" textAnchor="middle">
        参数/数据/算力按幂律共增 → 模型不只是变好，是出现新能力
      </text>
    </svg>
  );
}

/* =====================================================================
 * 2021 — CLIP dual encoder
 * ===================================================================== */
function ClipDiagram() {
  return (
    <svg viewBox="0 0 720 180" role="img" className="illustration__svg">
      {/* image tower */}
      <g transform="translate(40,20)">
        <text x="0" y="-6" className="illustration__label">
          image
        </text>
        <rect width="50" height="50" rx="6" className="illustration__pixel illustration__pixel--big" />
        <line x1="55" y1="25" x2="105" y2="25" className="illustration__flow" />
        <rect x="110" width="100" height="50" rx="8" className="illustration__block" />
        <text x="160" y="30" className="illustration__block-label" textAnchor="middle">
          ViT
        </text>
      </g>

      {/* text tower */}
      <g transform="translate(40,110)">
        <text x="0" y="-6" className="illustration__label">
          “a photo of a cat”
        </text>
        <rect width="50" height="40" rx="6" className="illustration__bar illustration__bar--real" />
        <line x1="55" y1="20" x2="105" y2="20" className="illustration__flow" />
        <rect x="110" width="100" height="40" rx="8" className="illustration__block illustration__block--alt" />
        <text x="160" y="25" className="illustration__block-label" textAnchor="middle">
          Text Transformer
        </text>
      </g>

      {/* shared space */}
      <g transform="translate(380,60)">
        <ellipse cx="100" cy="40" rx="120" ry="50" className="illustration__space" />
        <text x="100" y="-10" className="illustration__label" textAnchor="middle">
          shared embedding space
        </text>
        <circle cx="60" cy="35" r="6" className="illustration__neuron" />
        <circle cx="140" cy="50" r="6" className="illustration__neuron illustration__neuron--noise" />
        <line x1="60" y1="35" x2="140" y2="50" className="illustration__sim" />
        <text x="100" y="80" className="illustration__label" textAnchor="middle">
          cos sim ↑
        </text>
      </g>
    </svg>
  );
}
