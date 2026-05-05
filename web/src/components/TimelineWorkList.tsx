import type { TimelineWork } from "../data/timeline";

type TimelineWorkListProps = {
  works: TimelineWork[];
};

export function TimelineWorkList({ works }: TimelineWorkListProps) {
  return (
    <section className="content-card">
      <div className="content-card__eyebrow">同年关键工作</div>
      <div className="work-list">
        {works.map((work) => {
          const content = (
            <>
              <strong>{work.name}</strong>
              <span>{work.contribution}</span>
            </>
          );

          return work.modulePath ? (
            <a className="work-item" href={work.modulePath} key={work.name}>
              {content}
            </a>
          ) : (
            <div className="work-item" key={work.name}>
              {content}
            </div>
          );
        })}
      </div>
    </section>
  );
}
