/**
 * CNN 架构演进 · 概念卡片视图
 *
 * 风格对齐 TimelineIllustration：viewBox 1100、illustration__svg--tall、
 * 用现有 .illustration__block / __featuremap / __arrow / __pixel / __attn-cell 等 token。
 * 每张图聚焦一个 CNN 概念，配 1–2 句话注解。不堆 markdown 长文。
 */

function ArrowDefs() {
  return (
    <defs>
      <marker
        id="cnn-arrow"
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
  );
}

type ConceptCardProps = {
  index: string;
  title: string;
  caption: string;
  footer?: string;
  children: React.ReactNode;
};

function ConceptCard({ index, title, caption, footer, children }: ConceptCardProps) {
  return (
    <section className="concept-card">
      <header className="concept-card__head">
        <span className="concept-card__index">{index}</span>
        <div>
          <h3>{title}</h3>
          <p className="concept-card__caption">{caption}</p>
        </div>
      </header>
      <div className="concept-card__figure">{children}</div>
      {footer && <p className="concept-card__footer">{footer}</p>}
    </section>
  );
}

export function CnnTrack() {
  return (
    <div className="concept-stack">
      <Concept1Conv />
      <Concept2OutputShape />
      <Concept3ReceptiveField />
      <Concept4Pooling />
      <Concept5OneByOne />
      <Concept6BatchNorm />
      <Concept7Dilated />
      <Concept8DepthwiseSeparable />
      <Concept9Transposed />
      <Concept10Evolution />
    </div>
  );
}

/* =====================================================================
 * ① 卷积基础：input × kernel → output feature map
 * ===================================================================== */
function Concept1Conv() {
  return (
    <ConceptCard
      index="01"
      title="卷积是什么"
      caption="一个 3×3 卷积核在输入上滑动，每个位置算一次加权和，输出特征图"
      footer="参数共享：同一个核走遍整张图；这就是为什么 CNN 比 FC 省参数。"
    >
      <svg viewBox="0 0 1100 320" className="illustration__svg illustration__svg--tall" role="img">
        <ArrowDefs />

        {/* 输入网格 6×6 */}
        <g transform="translate(40, 40)">
          <text x="0" y="-8" className="illustration__label illustration__label--strong">输入 X (6×6)</text>
          {Array.from({ length: 6 }).map((_, r) =>
            Array.from({ length: 6 }).map((_, c) => (
              <rect
                key={`x-${r}-${c}`}
                x={c * 32}
                y={r * 32}
                width="30"
                height="30"
                rx="2"
                className="illustration__pixel"
                style={{ animationDelay: `${(r + c) * 50}ms` }}
              />
            ))
          )}
          {/* 高亮当前 3×3 窗口（位于 (1,1) 起点） */}
          <rect x={1 * 32} y={1 * 32} width="94" height="94" rx="4" className="illustration__window" />
          <text x="96" y="200" textAnchor="middle" className="illustration__label illustration__label--small">
            当前 3×3 窗口
          </text>
        </g>

        {/* × kernel */}
        <text x="278" y="120" className="illustration__label illustration__label--strong">⊗</text>

        {/* kernel 3×3 */}
        <g transform="translate(310, 80)">
          <text x="0" y="-8" className="illustration__label illustration__label--strong">卷积核 K (3×3)</text>
          {[
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
          ].map((row, r) =>
            row.map((v, c) => (
              <g key={`k-${r}-${c}`}>
                <rect
                  x={c * 32}
                  y={r * 32}
                  width="30"
                  height="30"
                  rx="2"
                  className="illustration__proj illustration__proj--q"
                />
                <text x={c * 32 + 15} y={r * 32 + 19} textAnchor="middle" className="illustration__block-label illustration__block-label--small">
                  {v}
                </text>
              </g>
            ))
          )}
          <text x="48" y="116" textAnchor="middle" className="illustration__label illustration__label--small">
            可学习参数 · 共 9 个权重
          </text>
        </g>

        {/* = 等号 */}
        <text x="438" y="130" className="illustration__label illustration__label--strong">=</text>

        {/* 输出特征图 4×4 */}
        <g transform="translate(480, 80)">
          <text x="0" y="-8" className="illustration__label illustration__label--strong">输出 Y (4×4)</text>
          {Array.from({ length: 4 }).map((_, r) =>
            Array.from({ length: 4 }).map((_, c) => (
              <rect
                key={`y-${r}-${c}`}
                x={c * 32}
                y={r * 32}
                width="30"
                height="30"
                rx="2"
                className="illustration__featuremap"
                style={{ animationDelay: `${400 + (r + c) * 60}ms` }}
              />
            ))
          )}
          <text x="64" y="148" textAnchor="middle" className="illustration__label illustration__label--small">
            每格 = 一次窗口加权和
          </text>
        </g>

        {/* 计算公式 */}
        <g transform="translate(700, 80)">
          <rect width="370" height="180" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="30" className="illustration__label illustration__label--strong">公式</text>
          <text x="20" y="60" className="illustration__label">
            Y(i,j) = Σₘ Σₙ X(i+m, j+n) · K(m,n)
          </text>
          <text x="20" y="92" className="illustration__label illustration__label--small">
            ① 锁定输入上一个 k×k 窗口
          </text>
          <text x="20" y="110" className="illustration__label illustration__label--small">
            ② 与卷积核逐元素相乘
          </text>
          <text x="20" y="128" className="illustration__label illustration__label--small">
            ③ 求和 → 输出一个值
          </text>
          <text x="20" y="158" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
            然后滑到下一位置，重复
          </text>
        </g>
      </svg>
    </ConceptCard>
  );
}

/* =====================================================================
 * ② 输出尺寸公式 + padding 模式
 * ===================================================================== */
function Concept2OutputShape() {
  return (
    <ConceptCard
      index="02"
      title="输出多大？看 kernel / stride / padding"
      caption="三个旋钮决定输出 H′ 和 W′，padding 模式控制边界保留"
      footer="实务里 99% 用 padding='same' + stride=1 提特征，再用 stride=2 卷积或 pool 显式下采样。"
    >
      <svg viewBox="0 0 1100 280" className="illustration__svg illustration__svg--tall" role="img">
        {/* 公式块 */}
        <g transform="translate(30, 30)">
          <rect width="540" height="220" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="32" className="illustration__label illustration__label--strong">输出尺寸递推</text>
          <text x="20" y="68" className="illustration__label">
            H′ = ⌊(H + 2p − d(k−1) − 1) / s⌋ + 1
          </text>
          <text x="20" y="100" className="illustration__label illustration__label--small">k = kernel size · s = stride · p = padding · d = dilation</text>

          <text x="20" y="138" className="illustration__label illustration__label--small">不带 dilation 时退化为：</text>
          <text x="20" y="160" className="illustration__label">H′ = (H + 2p − k) / s + 1</text>

          <text x="20" y="196" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
            想保 H 不变？只要 p = (k − 1) / 2 + 卷积核为奇数即可。
          </text>
        </g>

        {/* 三种 padding 模式可视化 */}
        <g transform="translate(610, 30)">
          <text x="0" y="0" className="illustration__label illustration__label--strong">三种 padding 模式</text>

          {[
            { y: 20, label: "valid (p=0)", desc: "只在合法位置卷，输出比输入小 k−1", padOpacity: 0 },
            { y: 90, label: "same", desc: "补到与输入同 H/W（s=1 时严格）", padOpacity: 0.4 },
            { y: 160, label: "full", desc: "卷到一边只搭一格也算，少用", padOpacity: 0.85 },
          ].map((mode, i) => (
            <g key={i} transform={`translate(0, ${mode.y + 8})`}>
              {/* padding 灰带 */}
              {mode.padOpacity > 0 && (
                <rect x="0" y="0" width="180" height="44" rx="2" className="illustration__attn-cell illustration__attn-cell--raw" style={{ opacity: mode.padOpacity }} />
              )}
              {/* 真实数据 */}
              <rect x={mode.padOpacity > 0 ? 14 : 0} y="8" width="152" height="28" rx="2" className="illustration__featuremap" />
              <text x="200" y="20" className="illustration__label illustration__label--strong">{mode.label}</text>
              <text x="200" y="38" className="illustration__label illustration__label--small">{mode.desc}</text>
            </g>
          ))}
        </g>
      </svg>
    </ConceptCard>
  );
}

/* =====================================================================
 * ③ 感受野：3 层 3×3 = 1 层 7×7
 * ===================================================================== */
function Concept3ReceptiveField() {
  return (
    <ConceptCard
      index="03"
      title="感受野（receptive field）"
      caption="多层小核累加 = 单层大核，但参数更少、非线性更强"
      footer="VGG 之后的范式：用 3 层 3×3 替代 7×7，省 ~28% 参数，多两次非线性。"
    >
      <svg viewBox="0 0 1100 320" className="illustration__svg illustration__svg--tall" role="img">
        <ArrowDefs />

        <text x="30" y="30" className="illustration__label illustration__label--strong">
          连续 3 个 3×3 → 等效 7×7 感受野
        </text>

        {/* 4 层堆叠：原图、conv1、conv2、conv3 */}
        {[
          { y: 60, label: "input", k: 13 },
          { y: 130, label: "after conv1 (3×3)", k: 11, rf: 3 },
          { y: 200, label: "after conv2 (3×3)", k: 9, rf: 5 },
          { y: 270, label: "after conv3 (3×3)", k: 7, rf: 7 },
        ].map((layer, i) => (
          <g key={i}>
            <text x="30" y={layer.y + 12} className="illustration__label illustration__label--small">{layer.label}</text>
            <g transform={`translate(220, ${layer.y})`}>
              {Array.from({ length: layer.k }).map((_, c) => (
                <rect
                  key={c}
                  x={c * 18}
                  y="0"
                  width="16"
                  height="16"
                  rx="2"
                  className={c === Math.floor(layer.k / 2) ? "illustration__featuremap illustration__featuremap--ctx" : "illustration__pixel"}
                />
              ))}
              {/* 中心点 RF 范围（仅 conv 层显示） */}
              {layer.rf && (
                <g>
                  <rect
                    x={Math.floor(layer.k / 2) * 18 - (layer.rf - 1) / 2 * 18}
                    y="-4"
                    width={layer.rf * 18 - 2}
                    height="24"
                    rx="4"
                    className="illustration__window"
                  />
                </g>
              )}
            </g>
            {layer.rf && (
              <text x={220 + Math.floor(layer.k / 2) * 18 + 40} y={layer.y + 12} className="illustration__label illustration__label--small">
                RF = {layer.rf}
              </text>
            )}
          </g>
        ))}

        {/* 公式 + 参数对比 */}
        <g transform="translate(620, 60)">
          <rect width="450" height="200" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="30" className="illustration__label illustration__label--strong">感受野递推公式</text>
          <text x="20" y="60" className="illustration__label">RFₗ = RFₗ₋₁ + (kₗ − 1) · jₗ₋₁</text>
          <text x="20" y="80" className="illustration__label illustration__label--small">jₗ = jₗ₋₁ · sₗ（每层步幅累乘）</text>

          <text x="20" y="116" className="illustration__label illustration__label--strong">参数量对比</text>
          <text x="20" y="138" className="illustration__label illustration__label--small">单层 7×7：49·C² 个权重</text>
          <text x="20" y="156" className="illustration__label illustration__label--small">三层 3×3：3 · 9·C² = 27·C² 个权重</text>
          <text x="20" y="176" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
            等效感受野，但参数省 44%、非线性 3 倍
          </text>
        </g>
      </svg>
    </ConceptCard>
  );
}

/* =====================================================================
 * ④ Pooling：Max / Avg / GAP
 * ===================================================================== */
function Concept4Pooling() {
  const grid = [
    [3, 7, 2, 1],
    [4, 9, 5, 0],
    [6, 1, 8, 3],
    [2, 5, 4, 7],
  ];
  return (
    <ConceptCard
      index="04"
      title="Pooling：下采样不带参数"
      caption="MaxPool 取最强、AvgPool 取均值、GAP 整图汇成 C 个标量"
      footer="现代 CNN 偏好 stride 卷积（可学）替代 MaxPool；分类头普遍用 GAP 替代 FC。"
    >
      <svg viewBox="0 0 1100 320" className="illustration__svg illustration__svg--tall" role="img">
        {/* 输入 4×4 */}
        <g transform="translate(30, 50)">
          <text x="0" y="-8" className="illustration__label illustration__label--strong">输入 (4×4)</text>
          {grid.map((row, r) =>
            row.map((v, c) => (
              <g key={`g-${r}-${c}`}>
                <rect
                  x={c * 40}
                  y={r * 40}
                  width="36"
                  height="36"
                  rx="3"
                  className="illustration__pixel"
                />
                <text x={c * 40 + 18} y={r * 40 + 24} textAnchor="middle" className="illustration__block-label illustration__block-label--small">
                  {v}
                </text>
              </g>
            ))
          )}
          {/* 上左 2×2 窗口高亮 */}
          <rect x="0" y="0" width="76" height="76" rx="4" className="illustration__window" />
        </g>

        {/* MaxPool 2×2 → 2×2 输出 */}
        <g transform="translate(280, 50)">
          <text x="0" y="-8" className="illustration__label illustration__label--strong">MaxPool 2×2</text>
          {[
            [9, 5],
            [6, 8],
          ].map((row, r) =>
            row.map((v, c) => (
              <g key={`m-${r}-${c}`}>
                <rect x={c * 50} y={r * 50} width="46" height="46" rx="3" className="illustration__featuremap illustration__featuremap--ctx" />
                <text x={c * 50 + 23} y={r * 50 + 30} textAnchor="middle" className="illustration__block-label">
                  {v}
                </text>
              </g>
            ))
          )}
          <text x="50" y="130" textAnchor="middle" className="illustration__label illustration__label--small">取最大 · 对扰动稳健</text>
        </g>

        {/* AvgPool 2×2 → 2×2 输出 */}
        <g transform="translate(480, 50)">
          <text x="0" y="-8" className="illustration__label illustration__label--strong">AvgPool 2×2</text>
          {[
            [5.75, 2.0],
            [3.5, 5.5],
          ].map((row, r) =>
            row.map((v, c) => (
              <g key={`a-${r}-${c}`}>
                <rect x={c * 50} y={r * 50} width="46" height="46" rx="3" className="illustration__featuremap" />
                <text x={c * 50 + 23} y={r * 50 + 30} textAnchor="middle" className="illustration__block-label">
                  {v}
                </text>
              </g>
            ))
          )}
          <text x="50" y="130" textAnchor="middle" className="illustration__label illustration__label--small">取均值 · 平滑能量</text>
        </g>

        {/* GAP → 1 个标量 */}
        <g transform="translate(700, 50)">
          <text x="0" y="-8" className="illustration__label illustration__label--strong">Global Avg Pool</text>
          <rect x="0" y="0" width="76" height="76" rx="3" className="illustration__featuremap illustration__featuremap--ctx" />
          <text x="38" y="46" textAnchor="middle" className="illustration__block-label">4.19</text>
          <text x="38" y="100" textAnchor="middle" className="illustration__label illustration__label--small">整张图 → 1 个数</text>
          <text x="38" y="118" textAnchor="middle" className="illustration__label illustration__label--small">每 channel 独立做</text>
        </g>

        {/* 选型表 */}
        <g transform="translate(820, 50)">
          <rect width="250" height="180" rx="10" className="illustration__block" />
          <text x="20" y="28" className="illustration__label illustration__label--strong">选型</text>
          <text x="20" y="50" className="illustration__label illustration__label--small">MaxPool — 老 CNN 主力</text>
          <text x="20" y="68" className="illustration__label illustration__label--small">stride conv — 现代分类网</text>
          <text x="20" y="86" className="illustration__label illustration__label--small">AvgPool — 中间层抑噪</text>
          <text x="20" y="104" className="illustration__label illustration__label--small">GAP — 分类头替 FC</text>
          <text x="20" y="140" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
            分割慎用 Max
          </text>
          <text x="20" y="158" className="illustration__label illustration__label--small">小目标会被吞掉</text>
        </g>
      </svg>
    </ConceptCard>
  );
}

/* =====================================================================
 * ⑤ 1×1 卷积
 * ===================================================================== */
function Concept5OneByOne() {
  return (
    <ConceptCard
      index="05"
      title="1×1 卷积 = 通道维度的全连接"
      caption="不动空间维度，只对通道做线性组合 / 升降维 / 嵌入非线性"
      footer="GoogLeNet 用它降维、ResNet bottleneck 用它做夹心、MobileNet 用它配 depthwise。"
    >
      <svg viewBox="0 0 1100 320" className="illustration__svg illustration__svg--tall" role="img">
        <ArrowDefs />

        <text x="30" y="30" className="illustration__label illustration__label--strong">
          每个空间位置独立做一次 Linear(C_in → C_out)
        </text>

        {/* 输入：4 通道，每通道一张小图 */}
        <g transform="translate(60, 80)">
          <text x="0" y="-8" className="illustration__label">输入 (C=4, H, W)</text>
          {Array.from({ length: 4 }).map((_, i) => (
            <g key={i} transform={`translate(${i * 6}, ${i * 6})`} style={{ opacity: 1 - i * 0.1 }}>
              <rect width="80" height="80" rx="4" className="illustration__featuremap" />
            </g>
          ))}
          <text x="55" y="120" textAnchor="middle" className="illustration__label illustration__label--small">堆 4 张 H×W</text>
        </g>

        {/* 1×1 conv 内部：在每个位置做 Linear */}
        <g transform="translate(260, 80)">
          <line x1="-40" y1="50" x2="-20" y2="50" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
          <rect width="160" height="100" rx="8" className="illustration__proj illustration__proj--o" />
          <text x="80" y="40" textAnchor="middle" className="illustration__block-label">1×1 Conv</text>
          <text x="80" y="60" textAnchor="middle" className="illustration__label illustration__label--small">每位置独立 Linear</text>
          <text x="80" y="78" textAnchor="middle" className="illustration__label illustration__label--small">C_in × C_out 个权重</text>
          <line x1="160" y1="50" x2="180" y2="50" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
        </g>

        {/* 输出：8 通道 */}
        <g transform="translate(460, 80)">
          <text x="0" y="-8" className="illustration__label">输出 (C=8, H, W)</text>
          {Array.from({ length: 8 }).map((_, i) => (
            <g key={i} transform={`translate(${i * 4}, ${i * 4})`} style={{ opacity: 1 - i * 0.06 }}>
              <rect width="80" height="80" rx="4" className="illustration__featuremap illustration__featuremap--ctx" />
            </g>
          ))}
          <text x="55" y="130" textAnchor="middle" className="illustration__label illustration__label--small">堆 8 张 H×W（已升维）</text>
        </g>

        {/* 三大用途 */}
        <g transform="translate(680, 60)">
          <rect width="390" height="220" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="32" className="illustration__label illustration__label--strong">3 个核心用途</text>

          <text x="20" y="64" className="illustration__label">① 跨通道融合</text>
          <text x="36" y="82" className="illustration__label illustration__label--small">让 C 个通道线性混合</text>

          <text x="20" y="112" className="illustration__label">② 升降维（瓶颈）</text>
          <text x="36" y="130" className="illustration__label illustration__label--small">256 → 64 做 3×3 → 256，省 4×</text>

          <text x="20" y="160" className="illustration__label">③ 嵌入非线性</text>
          <text x="36" y="178" className="illustration__label illustration__label--small">1×1 + BN + ReLU = 轻量 MLP</text>

          <text x="20" y="208" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
            又叫 point-wise conv
          </text>
        </g>
      </svg>
    </ConceptCard>
  );
}

/* =====================================================================
 * ⑥ Batch Normalization
 * ===================================================================== */
function Concept6BatchNorm() {
  return (
    <ConceptCard
      index="06"
      title="Batch Normalization"
      caption="按 channel 在 batch + 空间上算均值方差，归一化后再仿射"
      footer="train 用当前 batch 的统计；eval 用累积移动平均。忘切 model.eval() 是经典 bug。"
    >
      <svg viewBox="0 0 1100 360" className="illustration__svg illustration__svg--tall" role="img">
        <ArrowDefs />

        {/* 前向公式 */}
        <g transform="translate(30, 40)">
          <rect width="520" height="280" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="30" className="illustration__label illustration__label--strong">前向（每 channel 独立）</text>

          <text x="20" y="68" className="illustration__label">μ_B = (1/m) Σ x_i</text>
          <text x="20" y="92" className="illustration__label">σ²_B = (1/m) Σ (x_i − μ_B)²</text>
          <text x="20" y="124" className="illustration__label">x̂ = (x − μ_B) / √(σ²_B + ε)</text>
          <text x="20" y="156" className="illustration__label">y = γ · x̂ + β</text>

          <text x="20" y="200" className="illustration__label illustration__label--small">m = 该 channel 在 batch 中的元素数 = B · H · W</text>
          <text x="20" y="220" className="illustration__label illustration__label--small">γ, β = 每 channel 可学习仿射（恢复表达力）</text>

          <text x="20" y="252" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
            归一化让中间层激活分布稳定 → 可用更大学习率
          </text>
        </g>

        {/* train vs eval */}
        <g transform="translate(580, 40)">
          <rect width="490" height="280" rx="10" className="illustration__block" />
          <text x="20" y="30" className="illustration__label illustration__label--strong">train vs eval 两套统计</text>

          {/* train 模式 */}
          <g transform="translate(20, 56)">
            <rect width="220" height="180" rx="8" className="illustration__proj illustration__proj--q" />
            <text x="110" y="26" textAnchor="middle" className="illustration__block-label">model.train()</text>
            <text x="110" y="58" textAnchor="middle" className="illustration__label illustration__label--small">用当前 batch 的</text>
            <text x="110" y="78" textAnchor="middle" className="illustration__label illustration__label--small">μ_B, σ²_B</text>
            <text x="110" y="118" textAnchor="middle" className="illustration__label illustration__label--small">同时维护移动平均</text>
            <text x="110" y="138" textAnchor="middle" className="illustration__label illustration__label--small">μ̂, σ̂²</text>
            <text x="110" y="164" textAnchor="middle" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
              统计有噪声 ≈ 正则
            </text>
          </g>

          {/* eval 模式 */}
          <g transform="translate(250, 56)">
            <rect width="220" height="180" rx="8" className="illustration__proj illustration__proj--v" />
            <text x="110" y="26" textAnchor="middle" className="illustration__block-label">model.eval()</text>
            <text x="110" y="58" textAnchor="middle" className="illustration__label illustration__label--small">用累积的</text>
            <text x="110" y="78" textAnchor="middle" className="illustration__label illustration__label--small">μ̂, σ̂²</text>
            <text x="110" y="118" textAnchor="middle" className="illustration__label illustration__label--small">和 batch 内容无关</text>
            <text x="110" y="138" textAnchor="middle" className="illustration__label illustration__label--small">输出可复现</text>
            <text x="110" y="164" textAnchor="middle" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
              忘切是经典 bug
            </text>
          </g>

          <text x="20" y="262" className="illustration__label illustration__label--small">小 batch → BN 失效，换 GroupNorm；多卡训练用 SyncBN</text>
        </g>
      </svg>
    </ConceptCard>
  );
}

/* =====================================================================
 * ⑦ Dilated / Atrous Conv
 * ===================================================================== */
function Concept7Dilated() {
  const sampling = (d: number): { dx: number; dy: number }[] => {
    const offs = [];
    for (let r = -1; r <= 1; r++) for (let c = -1; c <= 1; c++) offs.push({ dx: c * d, dy: r * d });
    return offs;
  };

  return (
    <ConceptCard
      index="07"
      title="Dilated / Atrous Conv（空洞卷积）"
      caption="卷积核采样点之间插孔：参数不变、感受野指数扩大"
      footer="DeepLab 系列做语义分割的核心：保留高分辨率特征图，同时看到足够大的上下文。"
    >
      <svg viewBox="0 0 1100 320" className="illustration__svg illustration__svg--tall" role="img">
        <text x="30" y="30" className="illustration__label illustration__label--strong">
          相同的 9 个权重，不同 dilation 看不同范围
        </text>

        {/* 三种 dilation 并列 */}
        {[1, 2, 4].map((d, idx) => {
          const offsets = sampling(d);
          const center = 100;
          const cellSize = 18;
          return (
            <g key={d} transform={`translate(${60 + idx * 340}, 70)`}>
              <text x="100" y="-12" textAnchor="middle" className="illustration__label illustration__label--strong">
                d = {d}（RF = {2 * d + 1}）
              </text>
              {/* 13×13 背景网格 */}
              {Array.from({ length: 13 }).map((_, r) =>
                Array.from({ length: 13 }).map((_, c) => (
                  <rect
                    key={`g-${r}-${c}`}
                    x={c * (cellSize - 2) + 6}
                    y={r * (cellSize - 2) + 6}
                    width={cellSize - 4}
                    height={cellSize - 4}
                    rx="1"
                    className="illustration__attn-cell illustration__attn-cell--raw"
                    style={{ opacity: 0.12 }}
                  />
                ))
              )}
              {/* 采样点高亮 */}
              {offsets.map((o, i) => (
                <rect
                  key={`s-${i}`}
                  x={center - 80 + (6 + o.dx) * (cellSize - 2)}
                  y={center - 80 + (6 + o.dy) * (cellSize - 2)}
                  width={cellSize - 4}
                  height={cellSize - 4}
                  rx="1"
                  className="illustration__featuremap illustration__featuremap--ctx"
                />
              ))}
              <text x="100" y="240" textAnchor="middle" className="illustration__label illustration__label--small">
                {d === 1 ? "普通 conv" : d === 2 ? "等效 5×5" : "等效 9×9"}
              </text>
            </g>
          );
        })}
      </svg>
    </ConceptCard>
  );
}

/* =====================================================================
 * ⑧ Depthwise Separable
 * ===================================================================== */
function Concept8DepthwiseSeparable() {
  return (
    <ConceptCard
      index="08"
      title="Depthwise Separable Conv"
      caption="标准 conv = 空间卷积 × 通道融合；拆成两步算力降一个量级"
      footer="MobileNet / EfficientNet 的支柱。注意 GPU 上访存比低，wall-clock 未必更快。"
    >
      <svg viewBox="0 0 1100 360" className="illustration__svg illustration__svg--tall" role="img">
        <ArrowDefs />

        {/* 上：标准 conv */}
        <text x="30" y="30" className="illustration__label illustration__label--strong">
          标准 3×3 conv：一次性混合空间 + 通道
        </text>
        <g transform="translate(60, 50)">
          {/* 输入 C_in=4 */}
          {Array.from({ length: 4 }).map((_, i) => (
            <rect key={i} x={i * 5} y={i * 5} width="60" height="60" rx="3" className="illustration__featuremap" style={{ opacity: 1 - i * 0.12 }} />
          ))}
          <text x="40" y="92" textAnchor="middle" className="illustration__label illustration__label--small">C_in=4</text>
        </g>
        <line x1="180" y1="80" x2="220" y2="80" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
        <g transform="translate(230, 50)">
          <rect width="180" height="60" rx="8" className="illustration__proj illustration__proj--ffn" />
          <text x="90" y="28" textAnchor="middle" className="illustration__block-label">3×3 Conv (C_in × C_out)</text>
          <text x="90" y="46" textAnchor="middle" className="illustration__label illustration__label--small">k²·C_in·C_out 权重</text>
        </g>
        <line x1="420" y1="80" x2="460" y2="80" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
        <g transform="translate(470, 50)">
          {Array.from({ length: 8 }).map((_, i) => (
            <rect key={i} x={i * 3} y={i * 3} width="60" height="60" rx="3" className="illustration__featuremap illustration__featuremap--ctx" style={{ opacity: 1 - i * 0.06 }} />
          ))}
          <text x="40" y="92" textAnchor="middle" className="illustration__label illustration__label--small">C_out=8</text>
        </g>

        <g transform="translate(620, 50)">
          <rect width="450" height="80" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="30" className="illustration__label illustration__label--small">计算量：3² · 4 · 8 · H · W = 288 HW</text>
          <text x="20" y="56" className="illustration__label illustration__label--small">参数量：3² · 4 · 8 = 288</text>
        </g>

        {/* 下：depthwise + pointwise */}
        <text x="30" y="190" className="illustration__label illustration__label--strong" style={{ fill: "var(--rose-700)" }}>
          拆成两步：depthwise（每通道独立空间卷）+ pointwise（1×1 跨通道融合）
        </text>

        <g transform="translate(60, 210)">
          {Array.from({ length: 4 }).map((_, i) => (
            <rect key={i} x={i * 5} y={i * 5} width="60" height="60" rx="3" className="illustration__featuremap" style={{ opacity: 1 - i * 0.12 }} />
          ))}
        </g>
        <line x1="180" y1="240" x2="200" y2="240" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
        <g transform="translate(210, 210)">
          <rect width="130" height="60" rx="8" className="illustration__proj illustration__proj--q" />
          <text x="65" y="28" textAnchor="middle" className="illustration__block-label illustration__block-label--small">3×3 DW</text>
          <text x="65" y="46" textAnchor="middle" className="illustration__label illustration__label--small">groups = C_in</text>
        </g>
        <line x1="350" y1="240" x2="370" y2="240" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
        <g transform="translate(380, 210)">
          <rect width="130" height="60" rx="8" className="illustration__proj illustration__proj--v" />
          <text x="65" y="28" textAnchor="middle" className="illustration__block-label illustration__block-label--small">1×1 PW</text>
          <text x="65" y="46" textAnchor="middle" className="illustration__label illustration__label--small">C_in × C_out</text>
        </g>
        <line x1="520" y1="240" x2="540" y2="240" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
        <g transform="translate(550, 210)">
          {Array.from({ length: 8 }).map((_, i) => (
            <rect key={i} x={i * 3} y={i * 3} width="60" height="60" rx="3" className="illustration__featuremap illustration__featuremap--ctx" style={{ opacity: 1 - i * 0.06 }} />
          ))}
        </g>

        <g transform="translate(720, 210)">
          <rect width="350" height="80" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="30" className="illustration__label illustration__label--small">
            计算：3²·4·HW + 4·8·HW = 68 HW
          </text>
          <text x="20" y="50" className="illustration__label illustration__label--small">参数：3²·4 + 4·8 = 68</text>
          <text x="20" y="70" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
            ↓ 省 4.2×（C_out 越大省得越多）
          </text>
        </g>
      </svg>
    </ConceptCard>
  );
}

/* =====================================================================
 * ⑨ Transposed Conv（上采样）
 * ===================================================================== */
function Concept9Transposed() {
  return (
    <ConceptCard
      index="09"
      title="Transposed Conv（反卷积 / 上采样）"
      caption="不是真正的逆，而是把每个输入像素喷成 k×k 图案再累加"
      footer="k % s ≠ 0 时易出棋盘伪影；现代生成模型常用 bilinear upsample + 3×3 conv 替代。"
    >
      <svg viewBox="0 0 1100 320" className="illustration__svg illustration__svg--tall" role="img">
        <ArrowDefs />

        <text x="30" y="30" className="illustration__label illustration__label--strong">
          ConvTranspose2d (k=4, s=2)：H × W → 2H × 2W
        </text>

        {/* 输入小图 3×3 */}
        <g transform="translate(60, 80)">
          <text x="48" y="-8" textAnchor="middle" className="illustration__label">输入 (3×3)</text>
          {Array.from({ length: 3 }).map((_, r) =>
            Array.from({ length: 3 }).map((_, c) => (
              <rect
                key={`i-${r}-${c}`}
                x={c * 32}
                y={r * 32}
                width="30"
                height="30"
                rx="3"
                className="illustration__pixel"
              />
            ))
          )}
        </g>

        <line x1="200" y1="120" x2="240" y2="120" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
        <text x="220" y="112" textAnchor="middle" className="illustration__label illustration__label--small">每像素 ⊗ 4×4 核</text>

        {/* 中间：扩展 */}
        <g transform="translate(260, 80)">
          <text x="48" y="-8" textAnchor="middle" className="illustration__label">扩展中（喷洒）</text>
          {Array.from({ length: 6 }).map((_, r) =>
            Array.from({ length: 6 }).map((_, c) => {
              // 棋盘式叠加效果模拟
              const isCorner = (r % 2 === 0) && (c % 2 === 0);
              return (
                <rect
                  key={`m-${r}-${c}`}
                  x={c * 20}
                  y={r * 20}
                  width="18"
                  height="18"
                  rx="2"
                  className="illustration__featuremap"
                  style={{ opacity: isCorner ? 1 : 0.55 }}
                />
              );
            })
          )}
        </g>

        <line x1="400" y1="160" x2="440" y2="160" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />

        {/* 输出 6×6 */}
        <g transform="translate(460, 80)">
          <text x="60" y="-8" textAnchor="middle" className="illustration__label">输出 (6×6)</text>
          {Array.from({ length: 6 }).map((_, r) =>
            Array.from({ length: 6 }).map((_, c) => (
              <rect
                key={`o-${r}-${c}`}
                x={c * 20}
                y={r * 20}
                width="18"
                height="18"
                rx="2"
                className="illustration__featuremap illustration__featuremap--ctx"
              />
            ))
          )}
        </g>

        {/* 用例 + 棋盘陷阱 */}
        <g transform="translate(620, 70)">
          <rect width="450" height="200" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="30" className="illustration__label illustration__label--strong">典型用例</text>
          <text x="20" y="58" className="illustration__label illustration__label--small">• U-Net 分割：encoder ↓ + decoder ↑ + skip</text>
          <text x="20" y="78" className="illustration__label illustration__label--small">• DCGAN 生成：100 维噪声 → 64×64 图</text>
          <text x="20" y="98" className="illustration__label illustration__label--small">• 超分辨率：低分图 → 高分图</text>

          <text x="20" y="134" className="illustration__label illustration__label--strong" style={{ fill: "var(--rose-700)" }}>
            棋盘伪影
          </text>
          <text x="20" y="158" className="illustration__label illustration__label--small">k % s ≠ 0 时输出格子明暗规则化</text>
          <text x="20" y="176" className="illustration__label illustration__label--small">缓解：k=4 s=2 / bilinear+conv / Pixel Shuffle</text>
        </g>
      </svg>
    </ConceptCard>
  );
}

/* =====================================================================
 * ⑩ 架构演进 mini timeline
 * ===================================================================== */
/* ---------- 演进图里每个里程碑的迷你架构缩图 ---------- */

function MiniAlexNet() {
  return (
    <g>
      {/* 5 conv 块（递减）+ 3 fc 细条 */}
      {[
        { x: 0, w: 28 }, { x: 32, w: 22 }, { x: 58, w: 18 },
        { x: 80, w: 16 }, { x: 100, w: 14 },
      ].map((b, i) => (
        <rect key={i} x={b.x} y={26 - b.w / 2 + 14} width={b.w} height={b.w} rx="2"
          className="illustration__layer illustration__layer--conv" />
      ))}
      {[120, 130, 140].map((x) => (
        <rect key={x} x={x} y="22" width="6" height="32" rx="1"
          className="illustration__layer illustration__layer--fc" />
      ))}
    </g>
  );
}

function MiniVGG() {
  // 强调"深而整齐"：一长串等大方块
  return (
    <g>
      {Array.from({ length: 13 }).map((_, i) => (
        <rect key={i} x={i * 9} y="22" width="7" height="32" rx="1.5"
          className="illustration__layer illustration__layer--conv" />
      ))}
      {[120, 130, 140].map((x) => (
        <rect key={x} x={x} y="22" width="6" height="32" rx="1"
          className="illustration__layer illustration__layer--fc" />
      ))}
    </g>
  );
}

function MiniGoogLeNet() {
  // Inception：4 路并行 → concat
  return (
    <g>
      <rect x="0" y="32" width="14" height="14" rx="2" className="illustration__featuremap" />
      {[
        { y: 8, label: "1×1" },
        { y: 28, label: "3×3" },
        { y: 48, label: "5×5" },
        { y: 68, label: "pool" },
      ].map((br, i) => (
        <g key={i}>
          <path d={`M 14 39 C 30 39, 40 ${br.y + 6}, 56 ${br.y + 6}`} className="illustration__branch illustration__branch--q" fill="none" />
          <rect x="56" y={br.y} width="32" height="12" rx="2" className="illustration__proj illustration__proj--q" />
          <text x="72" y={br.y + 9} textAnchor="middle" fontSize="8" className="illustration__block-label illustration__block-label--small">{br.label}</text>
          <path d={`M 88 ${br.y + 6} C 100 ${br.y + 6}, 110 39, 124 39`} className="illustration__branch illustration__branch--k" fill="none" />
        </g>
      ))}
      <rect x="124" y="32" width="22" height="14" rx="2" className="illustration__featuremap illustration__featuremap--ctx" />
    </g>
  );
}

function MiniResNet() {
  // 残差：旁路 skip arc
  return (
    <g>
      <rect x="0" y="32" width="14" height="14" rx="2" className="illustration__featuremap" />
      {[30, 60, 90].map((x, i) => (
        <rect key={i} x={x} y="32" width="20" height="14" rx="2" className="illustration__proj illustration__proj--ffn" />
      ))}
      <circle cx="125" cy="39" r="6" className="illustration__addnorm" />
      <text x="125" y="42" textAnchor="middle" fontSize="10">⊕</text>
      <rect x="140" y="32" width="14" height="14" rx="2" className="illustration__featuremap illustration__featuremap--ctx" />
      {/* skip 弧 */}
      <path d="M 7 32 C 7 6, 125 6, 125 32" className="illustration__residual" fill="none" />
    </g>
  );
}

function MiniDenseNet() {
  // 稠密连接：每层接收前面所有层
  return (
    <g>
      {[0, 1, 2, 3, 4].map((i) => (
        <rect key={i} x={i * 32} y="32" width="22" height="14" rx="2"
          className={i === 0 ? "illustration__featuremap" : "illustration__proj illustration__proj--v"} />
      ))}
      {/* 所有跨层连接 */}
      {[
        { from: 0, to: 2 }, { from: 0, to: 3 }, { from: 0, to: 4 },
        { from: 1, to: 3 }, { from: 1, to: 4 }, { from: 2, to: 4 },
      ].map((c, i) => {
        const x1 = c.from * 32 + 11;
        const x2 = c.to * 32 + 11;
        const dy = -8 - (c.to - c.from) * 3;
        return (
          <path key={i} d={`M ${x1} 32 C ${x1} ${dy}, ${x2} ${dy}, ${x2} 32`}
            className="illustration__residual" fill="none" />
        );
      })}
    </g>
  );
}

function MiniSENet() {
  // 通道重标定：feature → GAP → FC → FC → sigmoid → × feature
  return (
    <g>
      <rect x="0" y="22" width="30" height="32" rx="2" className="illustration__featuremap" />
      <path d="M 30 38 C 42 38, 42 12, 54 12" className="illustration__branch illustration__branch--q" fill="none" />
      {[60, 86, 112].map((x, i) => (
        <rect key={i} x={x} y="6" width="20" height="14" rx="2"
          className={i === 0 ? "illustration__block illustration__block--alt" : "illustration__proj illustration__proj--v"} />
      ))}
      <text x="70" y="16" textAnchor="middle" fontSize="7">GAP</text>
      <text x="96" y="16" textAnchor="middle" fontSize="7">FC</text>
      <text x="122" y="16" textAnchor="middle" fontSize="7">σ</text>
      <path d="M 132 12 C 140 12, 140 38, 134 38" className="illustration__branch illustration__branch--q" fill="none" />
      <text x="138" y="42" fontSize="11">×</text>
      <rect x="146" y="22" width="30" height="32" rx="2" className="illustration__featuremap illustration__featuremap--ctx" />
    </g>
  );
}

function MiniResNeXt() {
  // Cardinality：分支并行
  return (
    <g>
      <rect x="0" y="32" width="14" height="14" rx="2" className="illustration__featuremap" />
      {[6, 24, 42, 60].map((y, i) => (
        <g key={i}>
          <path d={`M 14 39 C 28 39, 36 ${y + 5}, 50 ${y + 5}`} className="illustration__branch illustration__branch--q" fill="none" />
          <rect x="50" y={y} width="36" height="10" rx="2" className="illustration__proj illustration__proj--ffn" />
          <path d={`M 86 ${y + 5} C 100 ${y + 5}, 108 39, 122 39`} className="illustration__branch illustration__branch--v" fill="none" />
        </g>
      ))}
      <circle cx="128" cy="39" r="6" className="illustration__addnorm" />
      <text x="128" y="42" textAnchor="middle" fontSize="10">⊕</text>
    </g>
  );
}

function MiniMobileNet() {
  // Depthwise + Pointwise
  return (
    <g>
      <text x="0" y="14" fontSize="8" className="illustration__label illustration__label--small">DW 3×3</text>
      {[0, 1, 2, 3].map((i) => (
        <rect key={i} x="0" y={20 + i * 11} width="40" height="8" rx="1"
          className="illustration__proj illustration__proj--q" />
      ))}
      <line x1="48" y1="38" x2="64" y2="38" className="illustration__arrow" />
      <text x="56" y="14" fontSize="8" className="illustration__label illustration__label--small">PW 1×1</text>
      <rect x="70" y="20" width="40" height="44" rx="2" className="illustration__proj illustration__proj--v" />
      <line x1="118" y1="38" x2="134" y2="38" className="illustration__arrow" />
      <rect x="140" y="22" width="20" height="40" rx="2" className="illustration__featuremap illustration__featuremap--ctx" />
    </g>
  );
}

function MiniEfficientNet() {
  // 三轴等比缩放
  return (
    <g>
      <rect x="20" y="22" width="44" height="44" rx="3" className="illustration__featuremap" />
      {/* depth 轴：向上 */}
      <line x1="42" y1="22" x2="42" y2="6" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
      <text x="50" y="12" fontSize="9" className="illustration__label illustration__label--small">depth ↑</text>
      {/* width 轴：向右 */}
      <line x1="64" y1="44" x2="100" y2="44" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
      <text x="106" y="46" fontSize="9" className="illustration__label illustration__label--small">width →</text>
      {/* resolution 轴：右下 */}
      <line x1="64" y1="66" x2="90" y2="86" className="illustration__arrow" markerEnd="url(#cnn-arrow)" />
      <text x="96" y="86" fontSize="9" className="illustration__label illustration__label--small">res ↗</text>
      <text x="120" y="76" fontSize="9" fontWeight="700" className="illustration__label illustration__label--small">
        αᵈ·βʷ·γʳ ≤ 2
      </text>
    </g>
  );
}

function MiniConvNeXt() {
  // ResNet 骨架但用 ViT 风格组件
  return (
    <g>
      <rect x="0" y="32" width="14" height="14" rx="2" className="illustration__featuremap" />
      {/* 7×7 DW */}
      <rect x="22" y="28" width="28" height="22" rx="3" className="illustration__proj illustration__proj--q" />
      <text x="36" y="42" textAnchor="middle" fontSize="7">7×7 DW</text>
      {/* LN */}
      <rect x="56" y="28" width="20" height="22" rx="3" className="illustration__block illustration__block--alt" />
      <text x="66" y="42" textAnchor="middle" fontSize="8">LN</text>
      {/* 1×1 ×4 */}
      <rect x="82" y="28" width="24" height="22" rx="3" className="illustration__proj illustration__proj--ffn" />
      <text x="94" y="42" textAnchor="middle" fontSize="7">1×1↑4d</text>
      {/* GELU */}
      <rect x="112" y="28" width="20" height="22" rx="3" className="illustration__proj illustration__proj--act" />
      <text x="122" y="42" textAnchor="middle" fontSize="8">GELU</text>
      <circle cx="142" cy="39" r="6" className="illustration__addnorm" />
      <text x="142" y="42" textAnchor="middle" fontSize="10">⊕</text>
      {/* 残差弧 */}
      <path d="M 7 32 C 7 6, 142 6, 142 32" className="illustration__residual" fill="none" />
    </g>
  );
}

function Concept10Evolution() {
  type Era = {
    year: string;
    name: string;
    contrib: string;
    render: () => React.ReactElement;
  };
  const eras: Era[] = [
    { year: "2012", name: "AlexNet", contrib: "把 CNN 跑通", render: () => <MiniAlexNet /> },
    { year: "2014", name: "VGG", contrib: "深而整齐", render: () => <MiniVGG /> },
    { year: "2014", name: "GoogLeNet", contrib: "1×1 + 多尺度并行", render: () => <MiniGoogLeNet /> },
    { year: "2015", name: "ResNet", contrib: "残差跨深层", render: () => <MiniResNet /> },
    { year: "2016", name: "DenseNet", contrib: "稠密连接复用", render: () => <MiniDenseNet /> },
    { year: "2017", name: "SE-Net", contrib: "通道重标定", render: () => <MiniSENet /> },
    { year: "2017", name: "ResNeXt", contrib: "Cardinality 并行", render: () => <MiniResNeXt /> },
    { year: "2017", name: "MobileNet", contrib: "DW + PW", render: () => <MiniMobileNet /> },
    { year: "2019", name: "EfficientNet", contrib: "三轴等比缩放", render: () => <MiniEfficientNet /> },
    { year: "2022", name: "ConvNeXt", contrib: "ViT 反哺 CNN", render: () => <MiniConvNeXt /> },
  ];

  return (
    <ConceptCard
      index="10"
      title="架构演进 10 步（2012 → 2022）"
      caption="每一代都在修上一代的具体瓶颈 —— 把每个里程碑的「特征 shape」直接画出来"
      footer="2020 ViT 一度抢戏；2022 ConvNeXt 反过来证明：架构选择 > 卷积 vs 注意力。"
    >
      <svg viewBox="0 0 1100 560" className="illustration__svg illustration__svg--tall" role="img">
        <ArrowDefs />

        {/* 5 列 × 2 行布局 */}
        {eras.map((era, i) => {
          const col = i % 5;
          const row = Math.floor(i / 5);
          const cellW = 210;
          const cellH = 260;
          const x = 20 + col * cellW;
          const y = row * cellH + 12;
          return (
            <g key={era.name} transform={`translate(${x}, ${y})`}>
              {/* 卡片底框 */}
              <rect width={cellW - 12} height={cellH - 12} rx="10"
                className="illustration__block" />
              {/* 年份 */}
              <text x="14" y="22" className="illustration__label illustration__label--small">
                {era.year}
              </text>
              {/* 迷你架构 */}
              <g transform="translate(14, 38)">{era.render()}</g>
              {/* 名字 */}
              <text x="14" y="178" className="illustration__label illustration__label--strong">
                {era.name}
              </text>
              {/* 一句话 */}
              <text x="14" y="200" className="illustration__label illustration__label--small">
                {era.contrib}
              </text>
            </g>
          );
        })}
      </svg>
    </ConceptCard>
  );
}
