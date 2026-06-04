import { useRef, type CSSProperties } from "react";
import type { TimelineNode } from "../data/timeline";
import { phaseFamilyOf } from "../data/phaseFamily";

type TimelineAxisProps = {
  nodes: TimelineNode[];
  activeYear: string;
  onSelect: (year: string) => void;
  onOpenPrehistory: () => void;
  onJumpToTopic: (topicId: string) => void;
};

/**
 * 主题入口节点：站在主轴最左侧、与年份节点同一行，
 * 表示"在这之后一段年份范围内属于某个主题"。
 * 点击 → 滚到下方对应的主题概念图集合。
 */
type TopicEntry = {
  id: string;
  shortLabel: string; // 节点上的小字（比如 "CNN"）
  fullLabel: string; // 节点下的副标题
  family: "vision" | "language" | "scale" | "multimodal" | "alignment";
  spanLabel: string; // 时间跨度文本（2012–2022）
};

const TOPIC_ENTRIES: TopicEntry[] = [
  {
    id: "cnn-track",
    shortLabel: "CNN",
    fullLabel: "卷积神经网络",
    family: "vision",
    spanLabel: "2012–2022",
  },
];

export function TimelineAxis({
  activeYear,
  nodes,
  onSelect,
  onOpenPrehistory,
  onJumpToTopic,
}: TimelineAxisProps) {
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const activeIndex = Math.max(
    0,
    nodes.findIndex((node) => node.year === activeYear),
  );
  const progress =
    nodes.length > 1 ? (activeIndex / (nodes.length - 1)) * 100 : 0;
  const progressStyle = {
    "--timeline-progress": `${progress}%`,
  } as CSSProperties;

  function scrollBy(direction: -1 | 1) {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    scroller.scrollBy({ left: direction * 320, behavior: "smooth" });
  }

  return (
    <section className="timeline-panel" aria-labelledby="timeline-heading">
      <div className="timeline-panel__header">
        <h2 id="timeline-heading">横向主时间线 · 2012 起</h2>
        <p>左右滚动浏览完整链路，点击年份查看；左端 CNN 主题节点可进 10 张概念图。</p>
        <button
          type="button"
          className="timeline-panel__prehistory"
          onClick={onOpenPrehistory}
          aria-label="查看深度学习前史"
        >
          ⏪ 深度学习前史
        </button>
      </div>
      <div className="timeline-axis" style={progressStyle}>
        <div className="timeline-axis__track" aria-hidden="true">
          <span />
        </div>

        <button
          aria-label="向左滚动时间线"
          className="timeline-axis__nav timeline-axis__nav--left"
          onClick={() => scrollBy(-1)}
          type="button"
        >
          ‹
        </button>
        <button
          aria-label="向右滚动时间线"
          className="timeline-axis__nav timeline-axis__nav--right"
          onClick={() => scrollBy(1)}
          type="button"
        >
          ›
        </button>

        <div className="timeline-axis__nodes" ref={scrollerRef}>
          {/* 主题入口节点 —— 站在主轴最左端，与年份节点同行 */}
          {TOPIC_ENTRIES.map((topic) => (
            <button
              key={topic.id}
              type="button"
              className="timeline-node timeline-node--topic"
              data-family={topic.family}
              onClick={() => onJumpToTopic(topic.id)}
              aria-label={`${topic.shortLabel} 主题主线，跳到下方概念图`}
            >
              <span className="timeline-node__year">{topic.shortLabel}</span>
              <span className="timeline-node__dot timeline-node__dot--topic" aria-hidden="true">
                ⟶
              </span>
              <span className="timeline-node__title">{topic.fullLabel}</span>
              <span className="timeline-node__phase">{topic.spanLabel}</span>
            </button>
          ))}

          {nodes.map((node) => {
            const isActive = node.year === activeYear;
            const family = phaseFamilyOf(node.phase);

            return (
              <button
                aria-label={`${node.year} ${node.shortTitle}`}
                aria-current={isActive ? "step" : undefined}
                className="timeline-node"
                data-active={isActive}
                data-family={family}
                key={node.year}
                onClick={() => onSelect(node.year)}
                type="button"
              >
                <span className="timeline-node__year">{node.year}</span>
                <span className="timeline-node__dot" aria-hidden="true" />
                <span className="timeline-node__title">{node.shortTitle}</span>
                <span className="timeline-node__phase">{node.phase}</span>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
