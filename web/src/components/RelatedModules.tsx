import type { RelatedModule } from "../data/timeline";

type RelatedModulesProps = {
  modules: RelatedModule[];
};

export function RelatedModules({ modules }: RelatedModulesProps) {
  return (
    <section className="content-card">
      <div className="content-card__eyebrow">关联模块</div>
      <div className="module-links">
        {modules.map((module) => (
          <a href={module.path} key={module.path}>
            {module.label}
          </a>
        ))}
      </div>
    </section>
  );
}
