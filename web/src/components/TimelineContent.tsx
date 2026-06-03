import type { TimelineNode } from "../data/timeline";
import { phaseFamilyOf } from "../data/phaseFamily";
import { TimelineIllustration } from "./TimelineIllustration";

type TimelineContentProps = {
  node: TimelineNode;
};

export function TimelineContent({ node }: TimelineContentProps) {
  const family = phaseFamilyOf(node.phase);

  return (
    <article
      className="content-card content-card--main"
      data-family={family}
      key={node.year}
    >
      <div className="content-card__eyebrow">
        {node.year} · {node.phase}
      </div>
      <h2>{node.title}</h2>
      <p className="content-card__lead">{node.whatHappened}</p>

      <TimelineIllustration year={node.year} />

      <div className="reading-path" aria-label="当前节点阅读线索">
        <span data-tone="problem">旧瓶颈</span>
        <span data-tone="break">关键突破</span>
        <span data-tone="problem">新边界</span>
      </div>

      <div className="explain-grid">
        <section className="explain-block explain-block--input">
          <span className="explain-block__index" aria-hidden="true">
            01
          </span>
          <h3>之前卡在哪</h3>
          <p>{node.previousLimit}</p>
        </section>
        <section className="explain-block explain-block--compute">
          <span className="explain-block__index" aria-hidden="true">
            02
          </span>
          <h3>发生了什么</h3>
          <p>{node.whatHappened}</p>
        </section>
        <section className="explain-block explain-block--output">
          <span className="explain-block__index" aria-hidden="true">
            03
          </span>
          <h3>解决了什么</h3>
          <p>{node.solved}</p>
        </section>
        <section className="explain-block explain-block--limit">
          <span className="explain-block__index" aria-hidden="true">
            04
          </span>
          <h3>留下什么新问题</h3>
          <p>{node.newProblems}</p>
        </section>
      </div>
    </article>
  );
}
