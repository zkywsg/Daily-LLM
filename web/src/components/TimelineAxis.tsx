import { useRef, type CSSProperties } from "react";
import type { TimelineNode } from "../data/timeline";
import { phaseFamilyOf } from "../data/phaseFamily";

type TimelineAxisProps = {
  nodes: TimelineNode[];
  activeYear: string;
  onSelect: (year: string) => void;
  onOpenPrehistory: () => void;
  onJumpToTopic?: (topicId: string) => void;
};

export function TimelineAxis({
  activeYear,
  nodes,
  onSelect,
  onOpenPrehistory,
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
        <h2 id="timeline-heading">横向主时间线 · 1989 LeNet → 2025</h2>
        <p>左右滚动浏览完整链路，点击年份查看下方内容。</p>
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
