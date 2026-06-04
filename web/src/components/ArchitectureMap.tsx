/**
 * 三入口地图 —— 让首屏一眼看到知识库的三层结构。
 * · Layer 0 foundations  无时间维度的工具箱
 * · Layer 1 timeline     2012 起的编年主线（当前层）
 * · Layer 2 tracks       跨年代的主题深挖
 */

type ArchitectureMapProps = {
  /** 当前用户视线在哪一层（默认 L1 时间线） */
  activeLayer?: "foundation" | "timeline" | "track";
  onScrollToTimeline: () => void;
  /** 打开 web 内置长文页 */
  onOpenTrack: (trackId: string) => void;
};

type LayerDef = {
  id: "foundation" | "timeline" | "track";
  code: string;
  title: string;
  desc: string;
  cta: string;
  href?: string;
  /** 优先于 href：内部 web 跳转 */
  trackId?: string;
};

const LAYERS: LayerDef[] = [
  {
    id: "foundation",
    code: "L0",
    title: "基础工具箱",
    desc: "线代 · 概率 · BP · 激活 · 归一化 · 嵌入 …",
    cta: "翻看 16 + 篇基础",
    href: "../foundations/",
  },
  {
    id: "timeline",
    code: "L1",
    title: "编年主线",
    desc: "2012 → 2025 · 每节点 = 旧瓶颈 / 突破 / 解决 / 新问题",
    cta: "在下方主轴浏览",
  },
  {
    id: "track",
    code: "L2",
    title: "主题深挖",
    desc: "vision · language · scale-multi · alignment · systems",
    cta: "进 CNN 架构演进（已上线）",
    trackId: "vision/cnn-architectures",
  },
];

export function ArchitectureMap({
  activeLayer = "timeline",
  onScrollToTimeline,
  onOpenTrack,
}: ArchitectureMapProps) {
  return (
    <nav className="architecture-map" aria-label="知识库三层地图">
      {LAYERS.map((layer) => {
        const isActive = layer.id === activeLayer;
        const Body = (
          <>
            <div className="architecture-map__top">
              <span className="architecture-map__code">{layer.code}</span>
              <h3>{layer.title}</h3>
            </div>
            <p className="architecture-map__desc">{layer.desc}</p>
            <span className="architecture-map__cta">{layer.cta} →</span>
          </>
        );

        if (layer.id === "timeline") {
          return (
            <button
              key={layer.id}
              type="button"
              className="architecture-map__card"
              data-active={isActive}
              data-layer={layer.id}
              onClick={onScrollToTimeline}
            >
              {Body}
            </button>
          );
        }

        // L2 内部跳转
        if (layer.trackId) {
          return (
            <button
              key={layer.id}
              type="button"
              className="architecture-map__card"
              data-active={isActive}
              data-layer={layer.id}
              onClick={() => onOpenTrack(layer.trackId!)}
            >
              {Body}
            </button>
          );
        }

        return (
          <a
            key={layer.id}
            href={layer.href}
            className="architecture-map__card"
            data-active={isActive}
            data-layer={layer.id}
          >
            {Body}
          </a>
        );
      })}
    </nav>
  );
}
