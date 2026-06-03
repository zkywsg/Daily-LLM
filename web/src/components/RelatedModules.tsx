import type { RelatedModule } from "../data/timeline";

type RelatedModulesProps = {
  prerequisites: RelatedModule[];
  tracks: RelatedModule[];
};

/**
 * 侧栏关联模块 —— 拆成两栏：
 * · 前置基础（Layer 0）—— 入门补课，无时间维度
 * · 主题深挖（Layer 2）—— 跨年代纵向 track
 */
export function RelatedModules({
  prerequisites,
  tracks,
}: RelatedModulesProps) {
  return (
    <>
      {prerequisites.length > 0 && (
        <section className="content-card link-group" data-tone="foundation">
          <div className="content-card__eyebrow">前置基础 · L0</div>
          <p className="link-group__hint">
            看不懂当前内容时，先去这里补课。
          </p>
          <div className="module-links">
            {prerequisites.map((m) => (
              <a href={m.path} key={m.path}>
                {m.label}
              </a>
            ))}
          </div>
        </section>
      )}

      {tracks.length > 0 && (
        <section className="content-card link-group" data-tone="track">
          <div className="content-card__eyebrow">主题深挖 · L2</div>
          <p className="link-group__hint">
            想沿这条主线纵向往后追，进入下面的 track。
          </p>
          <div className="module-links">
            {tracks.map((m) => (
              <a href={m.path} key={m.path}>
                {m.label}
              </a>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
