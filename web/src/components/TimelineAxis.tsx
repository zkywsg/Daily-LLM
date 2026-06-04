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
 * 主题主线条目：一条横跨多年的主题轨。
 * fromIndex / toIndex 指 nodes 数组中的索引（含两端）。
 */
type TopicTrack = {
  id: string;
  label: string;
  family: "vision" | "language" | "scale" | "multimodal" | "alignment" | "foundation";
  fromYear: string;
  toYear: string;
};

const TOPIC_TRACKS: TopicTrack[] = [
  {
    id: "cnn-track",
    label: "CNN · 卷积神经网络",
    family: "vision",
    fromYear: "2012",
    toYear: "2022",
  },
];

const NODE_MIN_W = 96; // 与 .timeline-node min-width 对齐
const NODE_GAP = 4; // 与 .timeline-axis__nodes gap 对齐

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
        <p>左右滚动浏览完整链路，点击年份查看下方内容；底部 CNN 主题条点击展开。</p>
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

        <div className="timeline-axis__scroller" ref={scrollerRef}>
          {/* 年份节点行 */}
          <div className="timeline-axis__nodes">
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

          {/* 主题主线行 —— 与年份对齐的横向 pill */}
          <div className="timeline-axis__topics" aria-label="主题主线">
            {(() => {
              // 把年份轴划成 N 段，每个 topic 是一条从 fromIdx 到 toIdx 的 pill
              const N = nodes.length;
              const cells: { idx: number; topic?: TopicTrack; span: number }[] = [];
              let i = 0;
              while (i < N) {
                const topic = TOPIC_TRACKS.find(
                  (t) => nodes.findIndex((n) => n.year === t.fromYear) === i,
                );
                if (topic) {
                  const from = i;
                  const to = nodes.findIndex((n) => n.year === topic.toYear);
                  const span = Math.max(1, to - from + 1);
                  cells.push({ idx: i, topic, span });
                  i += span;
                } else {
                  cells.push({ idx: i, span: 1 });
                  i += 1;
                }
              }

              return cells.map((cell) => {
                const cellWidth =
                  NODE_MIN_W * cell.span + NODE_GAP * (cell.span - 1);
                const flexBasis = `${cellWidth}px`;
                if (cell.topic) {
                  return (
                    <button
                      key={cell.topic.id}
                      type="button"
                      className="timeline-axis__topic"
                      data-family={cell.topic.family}
                      onClick={() => onJumpToTopic(cell.topic!.id)}
                      style={{
                        flex: `${cell.span} 0 ${flexBasis}`,
                        minWidth: flexBasis,
                      }}
                    >
                      <span className="timeline-axis__topic-arrow">⟶</span>
                      <span className="timeline-axis__topic-label">
                        {cell.topic.label} · {cell.topic.fromYear}–{cell.topic.toYear}
                      </span>
                      <span className="timeline-axis__topic-cta">展开 10 张概念图 ↓</span>
                    </button>
                  );
                }
                return (
                  <div
                    key={`spacer-${cell.idx}`}
                    className="timeline-axis__topic-spacer"
                    aria-hidden="true"
                    style={{
                      flex: `1 0 ${flexBasis}`,
                      minWidth: flexBasis,
                    }}
                  />
                );
              });
            })()}
          </div>
        </div>
      </div>
    </section>
  );
}
