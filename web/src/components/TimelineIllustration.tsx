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
    caption:
      "Transformer 编码器一个完整 block：token → 嵌入 + 位置编码 → Q/K/V 投影 → 多头注意力 → Add&Norm → FFN → Add&Norm；右上 ×6 表示这个 block 串联六次",
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
 * 2017 — Transformer encoder (端到端完整 pipeline)
 *
 * 视图结构（从上到下）：
 *   ① Token 输入 → ② Embedding 行 → ③ + 位置编码 sin/cos
 *   ④ Encoder Block（带 ×6 徽标），内部：
 *      a. Multi-Head Self-Attention
 *           · X 同时投影 W_Q / W_K / W_V → Q, K, V
 *           · 每头独立算 Q·Kᵀ/√d → softmax → @V
 *           · 8 头并行（一头展开 + 七头堆叠）
 *           · Concat → W_O 输出投影
 *      b. Add & Norm 1（带从 X 跨过 MHA 的残差弧线）
 *      c. Feed-Forward Network: Linear d→4d → GELU → Linear 4d→d
 *      d. Add & Norm 2
 *   ⑤ 上下文化输出 5 个向量（颜色更饱和）
 * ===================================================================== */
function AttentionDiagram() {
  const tokens = ["The", "cat", "sat", "on", "mat"];
  // 5 token 的 x 中心位置（左右居中布局）
  const tokenXs = [120, 240, 360, 480, 600];
  const tokenW = 56;

  // 主头展开的 5x5 attention 矩阵
  const mainHead = [
    [0.92, 0.04, 0.02, 0.01, 0.01],
    [0.18, 0.62, 0.14, 0.04, 0.02],
    [0.08, 0.30, 0.46, 0.13, 0.03],
    [0.02, 0.06, 0.18, 0.58, 0.16],
    [0.01, 0.02, 0.06, 0.22, 0.69],
  ];

  return (
    <svg
      viewBox="0 0 1100 760"
      role="img"
      className="illustration__svg illustration__svg--tall"
    >
      <defs>
        <marker
          id="arrow-head"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" className="illustration__arrow-head" />
        </marker>
      </defs>

      {/* ============================================================ */}
      {/* ① 输入 tokens                                                 */}
      {/* ============================================================ */}
      <text x="30" y="30" className="illustration__label illustration__label--strong">
        ① 输入 token 序列
      </text>
      {tokens.map((t, i) => (
        <g
          key={`tok-${i}`}
          transform={`translate(${tokenXs[i] - tokenW / 2}, 44)`}
          style={{ animationDelay: `${i * 90}ms` }}
          className="illustration__appear"
        >
          <rect width={tokenW} height="30" rx="6" className="illustration__block illustration__block--alt" />
          <text x={tokenW / 2} y="20" className="illustration__block-label" textAnchor="middle">
            {t}
          </text>
        </g>
      ))}

      {/* token → embedding 箭头 */}
      {tokenXs.map((x, i) => (
        <line
          key={`a-${i}`}
          x1={x}
          y1="78"
          x2={x}
          y2="108"
          className="illustration__arrow"
          markerEnd="url(#arrow-head)"
        />
      ))}

      {/* ============================================================ */}
      {/* ② Embedding 行（每 token 一个 d 维向量，以一组小竖条表示）    */}
      {/* ============================================================ */}
      <text x="30" y="124" className="illustration__label illustration__label--strong">
        ② 嵌入 (d=512)
      </text>
      {tokenXs.map((x, i) =>
        Array.from({ length: 8 }).map((_, k) => (
          <rect
            key={`e-${i}-${k}`}
            x={x - 24 + k * 6}
            y="116"
            width="4"
            height="28"
            rx="1"
            className="illustration__pixel"
            style={{ animationDelay: `${200 + i * 60 + k * 20}ms` }}
          />
        )),
      )}

      {/* ============================================================ */}
      {/* ③ ⊕ 位置编码 sin/cos                                         */}
      {/* ============================================================ */}
      <text x="30" y="170" className="illustration__label illustration__label--strong">
        ③ + 位置编码 (sin/cos)
      </text>
      <PositionalWave x={90} y={172} width={620} />
      <text x="730" y="180" className="illustration__label">
        让序列顺序进入向量空间
      </text>

      {/* 注入箭头 */}
      <line x1="360" y1="200" x2="360" y2="232" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      {/* ============================================================ */}
      {/* ④ Encoder Block 外框 + ×6 徽标                                */}
      {/* ============================================================ */}
      <g>
        <rect
          x="40"
          y="232"
          width="1020"
          height="468"
          rx="14"
          className="illustration__group"
        />
        <g className="illustration__badge">
          <rect x="940" y="222" width="92" height="26" rx="13" />
          <text x="986" y="240" textAnchor="middle">
            × 6 blocks
          </text>
        </g>
        <text x="56" y="252" className="illustration__label illustration__label--strong">
          ④ Encoder Block · 内部完整流程
        </text>
      </g>

      {/* -- 4a. Multi-Head Self-Attention 外框 -------------------------- */}
      <rect
        x="58"
        y="262"
        width="984"
        height="232"
        rx="10"
        className="illustration__group illustration__group--inner"
      />
      <text x="72" y="280" className="illustration__label illustration__label--strong">
        a. Multi-Head Self-Attention
      </text>

      {/* X 输入矩阵（5 行） */}
      <g transform="translate(72, 292)">
        <text x="0" y="-2" className="illustration__label">X (输入)</text>
        {Array.from({ length: 5 }).map((_, r) => (
          <rect
            key={`x-${r}`}
            x="0"
            y={r * 14 + 6}
            width="56"
            height="11"
            rx="2"
            className="illustration__featuremap"
            style={{ animationDelay: `${400 + r * 50}ms` }}
          />
        ))}
      </g>

      {/* X 同时分三路 → W_Q / W_K / W_V 投影 → Q / K / V */}
      <g transform="translate(140, 296)">
        {/* 三条分流线（彩色） */}
        <path d="M 0 30 C 30 30, 50 0, 80 0" className="illustration__branch illustration__branch--q" fill="none" markerEnd="url(#arrow-head)" />
        <path d="M 0 30 L 80 30" className="illustration__branch illustration__branch--k" fill="none" markerEnd="url(#arrow-head)" />
        <path d="M 0 30 C 30 30, 50 60, 80 60" className="illustration__branch illustration__branch--v" fill="none" markerEnd="url(#arrow-head)" />

        {/* W_Q W_K W_V 投影矩阵 */}
        {(["W_Q", "W_K", "W_V"] as const).map((lbl, i) => (
          <g key={lbl} transform={`translate(82, ${i * 30 - 8})`}>
            <rect width="42" height="20" rx="4" className={`illustration__proj illustration__proj--${["q", "k", "v"][i]}`} />
            <text x="21" y="14" textAnchor="middle" className="illustration__block-label illustration__block-label--small">
              {lbl}
            </text>
          </g>
        ))}

        {/* Q/K/V 输出 */}
        {(["Q", "K", "V"] as const).map((lbl, i) => (
          <g key={lbl} transform={`translate(134, ${i * 30 - 8})`}>
            <line x1="0" y1="10" x2="20" y2="10" className="illustration__arrow" markerEnd="url(#arrow-head)" />
            <text x="30" y="14" className="illustration__label illustration__label--inline">{lbl}</text>
          </g>
        ))}
      </g>

      {/* 单头注意力详细计算（QKᵀ/√d → softmax → @V） */}
      <g transform="translate(360, 296)">
        <text x="0" y="-4" className="illustration__label">
          每头：scaled dot-product attention
        </text>

        {/* Q·Kᵀ 矩阵 */}
        <g transform="translate(0, 4)">
          <text x="36" y="-2" className="illustration__label illustration__label--small" textAnchor="middle">Q·Kᵀ / √d</text>
          {mainHead.map((row, r) =>
            row.map((_, c) => (
              <rect
                key={`qk-${r}-${c}`}
                x={c * 14}
                y={r * 14 + 4}
                width="12"
                height="12"
                rx="1.5"
                className="illustration__attn-cell illustration__attn-cell--raw"
                style={{ animationDelay: `${600 + (r * 5 + c) * 25}ms` }}
              />
            )),
          )}
        </g>

        <line x1="78" y1="44" x2="98" y2="44" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="88" y="38" textAnchor="middle" className="illustration__label illustration__label--small">softmax</text>

        {/* softmax 后概率矩阵 */}
        <g transform="translate(102, 4)">
          <text x="36" y="-2" className="illustration__label illustration__label--small" textAnchor="middle">attention 概率</text>
          {mainHead.map((row, r) =>
            row.map((v, c) => (
              <rect
                key={`sm-${r}-${c}`}
                x={c * 14}
                y={r * 14 + 4}
                width="12"
                height="12"
                rx="1.5"
                className="illustration__attn-cell"
                style={{
                  opacity: 0.18 + v * 0.82,
                  animationDelay: `${900 + (r * 5 + c) * 25}ms`,
                }}
              />
            )),
          )}
        </g>

        <line x1="180" y1="44" x2="200" y2="44" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="190" y="38" textAnchor="middle" className="illustration__label illustration__label--small">@V</text>

        {/* head 输出向量 */}
        <g transform="translate(204, 4)">
          <text x="20" y="-2" className="illustration__label illustration__label--small" textAnchor="middle">head 输出</text>
          {Array.from({ length: 5 }).map((_, r) => (
            <rect
              key={`ho-${r}`}
              x="0"
              y={r * 14 + 4}
              width="40"
              height="12"
              rx="2"
              className="illustration__featuremap"
              style={{ animationDelay: `${1200 + r * 40}ms` }}
            />
          ))}
        </g>
      </g>

      {/* ×8 头（堆叠表示） */}
      <g transform="translate(660, 300)">
        <text x="0" y="-4" className="illustration__label">8 头并行</text>
        {Array.from({ length: 7 }).map((_, i) => (
          <g key={i} transform={`translate(${i * 5}, ${i * 5})`} style={{ opacity: 1 - i * 0.1 }}>
            <rect width="52" height="72" rx="4" className="illustration__head-stack" />
          </g>
        ))}
        <text x="80" y="40" className="illustration__label illustration__label--small">
          每头独立子空间
        </text>
        <text x="80" y="56" className="illustration__label illustration__label--small">
          建模不同关系
        </text>
      </g>

      {/* Concat + W_O */}
      <g transform="translate(820, 296)">
        <line x1="-20" y1="40" x2="0" y2="40" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <rect width="78" height="32" y="24" rx="6" className="illustration__block illustration__block--alt" />
        <text x="39" y="44" textAnchor="middle" className="illustration__block-label">Concat</text>
        <text x="39" y="64" textAnchor="middle" className="illustration__label illustration__label--small">8·d/8 = d</text>

        <line x1="78" y1="40" x2="98" y2="40" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(98, 24)">
          <rect width="78" height="32" rx="6" className="illustration__proj illustration__proj--o" />
          <text x="39" y="20" textAnchor="middle" className="illustration__block-label">W_O</text>
        </g>
      </g>

      {/* MHA 输出箭头到 Add&Norm */}
      <line x1="552" y1="494" x2="552" y2="510" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      {/* -- 4b. Add & Norm 1 + 残差弧线 -------------------------------- */}
      <g>
        {/* 残差：从 X 入口绕过 MHA */}
        <path
          d="M 100 332 C 24 332, 24 528, 460 528"
          className="illustration__residual"
          fill="none"
          markerEnd="url(#arrow-head)"
        />
        <text x="20" y="430" className="illustration__label illustration__label--small" transform="rotate(-90 20 430)">
          residual (恒等捷径)
        </text>

        <g transform="translate(478, 510)">
          <circle r="14" cx="14" cy="14" className="illustration__addnorm" />
          <text x="14" y="19" textAnchor="middle" className="illustration__block-label">⊕</text>
          <line x1="28" y1="14" x2="48" y2="14" className="illustration__arrow" markerEnd="url(#arrow-head)" />
          <rect x="50" y="0" width="88" height="28" rx="6" className="illustration__block illustration__block--alt" />
          <text x="94" y="18" textAnchor="middle" className="illustration__block-label">LayerNorm</text>
        </g>
      </g>

      {/* 进入 FFN 的下行箭头 */}
      <line x1="572" y1="538" x2="572" y2="560" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      {/* -- 4c. Feed-Forward Network ----------------------------------- */}
      <rect x="58" y="560" width="984" height="80" rx="10" className="illustration__group illustration__group--inner" />
      <text x="72" y="578" className="illustration__label illustration__label--strong">
        c. Feed-Forward Network (position-wise，每个位置独立两层 MLP)
      </text>
      <g transform="translate(180, 596)">
        <rect width="160" height="32" rx="6" className="illustration__proj illustration__proj--ffn" />
        <text x="80" y="20" textAnchor="middle" className="illustration__block-label">Linear · d → 4d</text>

        <line x1="160" y1="16" x2="200" y2="16" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(200, 0)">
          <rect width="92" height="32" rx="6" className="illustration__proj illustration__proj--act" />
          <text x="46" y="20" textAnchor="middle" className="illustration__block-label">GELU</text>
        </g>

        <line x1="292" y1="16" x2="332" y2="16" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(332, 0)">
          <rect width="160" height="32" rx="6" className="illustration__proj illustration__proj--ffn" />
          <text x="80" y="20" textAnchor="middle" className="illustration__block-label">Linear · 4d → d</text>
        </g>

        <text x="540" y="20" className="illustration__label illustration__label--small">
          扩张 → 非线性 → 收缩
        </text>
      </g>

      {/* FFN 输出箭头到 Add&Norm 2 */}
      <line x1="552" y1="644" x2="552" y2="658" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      {/* -- 4d. Add & Norm 2 ------------------------------------------- */}
      <g>
        <path
          d="M 600 528 C 980 528, 980 670, 620 670"
          className="illustration__residual"
          fill="none"
          markerEnd="url(#arrow-head)"
        />
        <text x="990" y="600" className="illustration__label illustration__label--small" transform="rotate(-90 990 600)">
          residual
        </text>

        <g transform="translate(478, 660)">
          <circle r="14" cx="14" cy="14" className="illustration__addnorm" />
          <text x="14" y="19" textAnchor="middle" className="illustration__block-label">⊕</text>
          <line x1="28" y1="14" x2="48" y2="14" className="illustration__arrow" markerEnd="url(#arrow-head)" />
          <rect x="50" y="0" width="88" height="28" rx="6" className="illustration__block illustration__block--alt" />
          <text x="94" y="18" textAnchor="middle" className="illustration__block-label">LayerNorm</text>
        </g>
      </g>

      {/* ============================================================ */}
      {/* ⑤ 输出：5 个上下文化向量                                       */}
      {/* ============================================================ */}
      <line x1="572" y1="690" x2="572" y2="716" className="illustration__arrow" markerEnd="url(#arrow-head)" />
      <text x="30" y="734" className="illustration__label illustration__label--strong">
        ⑤ 输出 · 每个位置已聚合全局上下文（与输入同形状，可直接喂下一个 block）
      </text>
      {tokenXs.map((x, i) =>
        Array.from({ length: 8 }).map((_, k) => (
          <rect
            key={`out-${i}-${k}`}
            x={x - 24 + k * 6}
            y="724"
            width="4"
            height="28"
            rx="1"
            className="illustration__featuremap illustration__featuremap--ctx"
            style={{ animationDelay: `${1500 + i * 60 + k * 20}ms` }}
          />
        )),
      )}
    </svg>
  );
}

/* 位置编码 sin/cos 波形 —— 两条相位错开的连续波，slow 漂移 */
function PositionalWave({ x, y, width }: { x: number; y: number; width: number }) {
  const samples = 80;
  const amp = 10;
  const points = (phase: number) =>
    Array.from({ length: samples }, (_, i) => {
      const t = i / (samples - 1);
      const px = x + t * width;
      const py = y + 12 + Math.sin(t * Math.PI * 4 + phase) * amp;
      return `${i === 0 ? "M" : "L"} ${px} ${py}`;
    }).join(" ");

  return (
    <g>
      <path d={points(0)} className="illustration__wave illustration__wave--sin" fill="none" />
      <path d={points(Math.PI / 2)} className="illustration__wave illustration__wave--cos" fill="none" />
    </g>
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
