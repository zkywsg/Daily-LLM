import type { RelatedModule } from "../data/timeline";
import { isKnownTrack } from "./TrackView";

type RelatedModulesProps = {
  prerequisites: RelatedModule[];
  tracks: RelatedModule[];
  onOpenTrack: (trackId: string) => void;
};

/**
 * 侧栏关联模块 —— 拆成两栏：
 * · 前置基础（Layer 0）—— 入门补课，无时间维度
 * · 主题深挖（Layer 2）—— 跨年代纵向 track
 *
 * 路径形如 `../tracks/vision/cnn-architectures/` 的链接：
 * · 如果 web 里已经接入这条 track（isKnownTrack），就走内部 #track= 跳转
 * · 否则保留 href 跳到外部 GitHub 仓库文件
 */
export function RelatedModules({
  prerequisites,
  tracks,
  onOpenTrack,
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
            {tracks.map((m) => {
              const internalTrackId = pathToTrackId(m.path);
              if (internalTrackId && isKnownTrack(internalTrackId)) {
                return (
                  <button
                    type="button"
                    key={m.path}
                    onClick={() => onOpenTrack(internalTrackId)}
                    className="module-links__internal"
                  >
                    {m.label}
                    <span className="module-links__badge">web</span>
                  </button>
                );
              }
              return (
                <a href={m.path} key={m.path}>
                  {m.label}
                </a>
              );
            })}
          </div>
        </section>
      )}
    </>
  );
}

/**
 * 把 `../tracks/vision/cnn-architectures/` 解析成 track id
 * `vision/cnn-architectures`；非 tracks 路径返回 null
 */
function pathToTrackId(path: string): string | null {
  const m = path.match(/(?:^|\/)tracks\/([^/]+(?:\/[^/]+)?)\/?$/);
  return m ? m[1] : null;
}
