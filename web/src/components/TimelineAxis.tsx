import type { CSSProperties } from "react";
import type { TimelineNode } from "../data/timeline";

type TimelineAxisProps = {
  nodes: TimelineNode[];
  activeYear: string;
  onSelect: (year: string) => void;
};

export function TimelineAxis({
  activeYear,
  nodes,
  onSelect,
}: TimelineAxisProps) {
  const activeIndex = Math.max(
    0,
    nodes.findIndex((node) => node.year === activeYear),
  );
  const progress =
    nodes.length > 1 ? (activeIndex / (nodes.length - 1)) * 100 : 0;
  const progressStyle = {
    "--timeline-progress": `${progress}%`,
  } as CSSProperties;

  return (
    <section className="timeline-panel" aria-labelledby="timeline-heading">
      <div className="timeline-panel__header">
        <h2 id="timeline-heading">横向主时间线</h2>
        <p>左右滚动浏览完整链路，点击年份查看下方内容。</p>
      </div>
      <div className="timeline-axis" style={progressStyle}>
        <div className="timeline-axis__track" aria-hidden="true">
          <span />
        </div>
        <div className="timeline-axis__nodes">
          {nodes.map((node) => {
            const isActive = node.year === activeYear;

            return (
              <button
                aria-current={isActive ? "step" : undefined}
                className="timeline-node"
                data-active={isActive}
                key={node.year}
                onClick={() => onSelect(node.year)}
                type="button"
              >
                <span className="timeline-node__year">{node.year}</span>
                <span className="timeline-node__dot" aria-hidden="true" />
                <span className="timeline-node__title">{node.shortTitle}</span>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
