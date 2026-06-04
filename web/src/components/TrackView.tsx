import { CnnTrack } from "./tracks/CnnTrack";

/**
 * 长文章页 dispatcher
 * 每条 track 用一套与 TimelineIllustration 同风格的「概念 SVG 卡片」呈现，
 * 不再做 markdown 长文倾倒。
 */

type TrackMeta = {
  title: string;
  subtitle: string;
  render: () => React.ReactElement;
};

const TRACKS: Record<string, TrackMeta> = {
  "vision/cnn-architectures": {
    title: "CNN 架构演进",
    subtitle: "tracks · vision · 2012 → 2022",
    render: () => <CnnTrack />,
  },
};

export function isKnownTrack(trackId: string): boolean {
  return trackId in TRACKS;
}

type TrackViewProps = {
  trackId: string;
  onBack: () => void;
};

export function TrackView({ trackId, onBack }: TrackViewProps) {
  const meta = TRACKS[trackId];

  if (!meta) {
    return (
      <section className="track-view track-view--missing">
        <h2>未找到这条 track</h2>
        <p>暂时只接入了 vision/cnn-architectures。返回时间线继续浏览。</p>
        <button type="button" onClick={onBack}>← 回时间线</button>
      </section>
    );
  }

  return (
    <section className="track-view" aria-label={meta.title}>
      <header className="track-view__hero">
        <button type="button" className="track-view__back" onClick={onBack}>
          ← 回时间线
        </button>
        <p className="track-view__kicker">{meta.subtitle}</p>
        <h1>{meta.title}</h1>
        <p className="track-view__source">
          每张图聚焦一个核心概念，与时间线节点上的架构图同一套视觉语言
        </p>
      </header>

      <div className="track-view__concepts">{meta.render()}</div>
    </section>
  );
}
