import type { TimelineNode } from "../data/timeline";

type TimelineContentProps = {
  node: TimelineNode;
};

export function TimelineContent({ node }: TimelineContentProps) {
  return (
    <article className="content-card content-card--main">
      <div className="content-card__eyebrow">
        {node.year} · {node.phase}
      </div>
      <h2>{node.title}</h2>
      <p className="content-card__lead">{node.whatHappened}</p>

      <div className="explain-grid">
        <section className="explain-block explain-block--input">
          <h3>之前卡在哪</h3>
          <p>{node.previousLimit}</p>
        </section>
        <section className="explain-block explain-block--compute">
          <h3>发生了什么</h3>
          <p>{node.whatHappened}</p>
        </section>
        <section className="explain-block explain-block--output">
          <h3>解决了什么</h3>
          <p>{node.solved}</p>
        </section>
        <section className="explain-block explain-block--limit">
          <h3>留下什么新问题</h3>
          <p>{node.newProblems}</p>
        </section>
      </div>
    </article>
  );
}
