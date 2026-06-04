import { useEffect, useMemo, useRef, useState } from "react";
import { ArchitectureMap } from "./components/ArchitectureMap";
import { PrehistoryDrawer } from "./components/PrehistoryDrawer";
import { RelatedModules } from "./components/RelatedModules";
import { TimelineAxis } from "./components/TimelineAxis";
import { TimelineContent } from "./components/TimelineContent";
import { TimelineWorkList } from "./components/TimelineWorkList";
import { TrackView, isKnownTrack } from "./components/TrackView";
import { CnnTrack } from "./components/tracks/CnnTrack";
import { getNodeByYear, timelineNodes } from "./data/timeline";
import {
  PHASE_FAMILY_LABEL,
  PHASE_FAMILY_ORDER,
  phaseFamilyOf,
} from "./data/phaseFamily";

const DEFAULT_YEAR = "2012";

type HashState = {
  year?: string;
  prehistory: boolean;
  track?: string;
};

/**
 * 解析 hash：
 * · 旧格式 `#2017` —— 兼容老链接
 * · 新格式 `#year=2017` / `#prehistory`
 * · `#track=vision/cnn-architectures` —— 主题深挖长文页
 * · `#anchor=xxx` —— TrackView 内章节锚点，主路由忽略它
 */
function parseHash(hash: string): HashState {
  // 锚点放到任意位置都能识别；从主路由里剥掉
  let raw = hash.replace(/^#/, "");
  raw = raw.split("#anchor=")[0];
  if (!raw) return { prehistory: false };
  if (raw === "prehistory") return { prehistory: true };
  if (raw.startsWith("year=")) {
    return { year: raw.slice("year=".length), prehistory: false };
  }
  if (raw.startsWith("track=")) {
    return { track: raw.slice("track=".length), prehistory: false };
  }
  // legacy: pure year
  if (/^\d{4}$/.test(raw)) return { year: raw, prehistory: false };
  return { prehistory: false };
}

function initialState(): {
  year: string;
  prehistoryOpen: boolean;
  track?: string;
} {
  const parsed = parseHash(window.location.hash);
  const year =
    parsed.year && getNodeByYear(parsed.year) ? parsed.year : DEFAULT_YEAR;
  return {
    year,
    prehistoryOpen: parsed.prehistory,
    track: parsed.track && isKnownTrack(parsed.track) ? parsed.track : undefined,
  };
}

export default function App() {
  const initial = useMemo(initialState, []);
  const [activeYear, setActiveYear] = useState(initial.year);
  const [prehistoryOpen, setPrehistoryOpen] = useState(initial.prehistoryOpen);
  const [activeTrack, setActiveTrack] = useState<string | undefined>(
    initial.track,
  );
  const timelineRef = useRef<HTMLDivElement | null>(null);

  const activeNode = useMemo(
    () => getNodeByYear(activeYear) ?? getNodeByYear(DEFAULT_YEAR)!,
    [activeYear],
  );
  const activeFamily = phaseFamilyOf(activeNode.phase);

  function selectYear(year: string) {
    setActiveYear(year);
    setActiveTrack(undefined);
    window.location.hash = `year=${year}`;
  }

  function openPrehistory() {
    setPrehistoryOpen(true);
    window.location.hash = "prehistory";
  }
  function closePrehistory() {
    setPrehistoryOpen(false);
    // 恢复到当前年份 hash，避免回到顶
    window.location.hash = `year=${activeYear}`;
  }

  function openTrack(trackId: string) {
    setActiveTrack(trackId);
    window.location.hash = `track=${trackId}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function closeTrack() {
    setActiveTrack(undefined);
    window.location.hash = `year=${activeYear}`;
    window.scrollTo({ top: 0 });
  }

  function scrollToTimeline() {
    if (activeTrack) {
      closeTrack();
      return;
    }
    timelineRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  // 跟随浏览器前进/后退
  useEffect(() => {
    function onHash() {
      const parsed = parseHash(window.location.hash);
      if (parsed.year && getNodeByYear(parsed.year)) {
        setActiveYear(parsed.year);
      }
      setPrehistoryOpen(parsed.prehistory);
      setActiveTrack(
        parsed.track && isKnownTrack(parsed.track) ? parsed.track : undefined,
      );
    }
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // 进入 TrackView 模式：只渲染 TrackView，隐藏时间线相关 UI
  if (activeTrack) {
    return (
      <main className="app-shell app-shell--track" data-active-family={activeFamily}>
        <TrackView trackId={activeTrack} onBack={closeTrack} />
      </main>
    );
  }

  return (
    <main className="app-shell" data-active-family={activeFamily}>
      <header className="hero">
        <div className="hero__title">
          <p className="hero__kicker">Daily-LLM</p>
          <h1>深度学习与大模型演进时间线</h1>
          <p className="hero__summary">
            每一个技术的出现，背后都有一个不得不解决的问题。
          </p>
        </div>
        <div className="hero__status" aria-label="当前节点">
          <span>2012 → 2025</span>
          <strong>
            当前：{activeNode.year} · {activeNode.shortTitle}
          </strong>
        </div>
      </header>

      <ArchitectureMap
        onScrollToTimeline={scrollToTimeline}
        onOpenTrack={openTrack}
      />

      <nav className="phase-rail" aria-label="叙事主线">
        {PHASE_FAMILY_ORDER.map((family) => {
          const isActive = family === activeFamily;
          return (
            <div
              key={family}
              className="phase-rail__segment"
              data-family={family}
              data-active={isActive}
            >
              <span className="phase-rail__bar" aria-hidden="true" />
              <span className="phase-rail__label">
                {PHASE_FAMILY_LABEL[family]}
              </span>
            </div>
          );
        })}
      </nav>

      <div ref={timelineRef}>
        <TimelineAxis
          activeYear={activeYear}
          nodes={timelineNodes}
          onSelect={selectYear}
          onOpenPrehistory={openPrehistory}
          onJumpToTopic={(topicId) => {
            document
              .getElementById(topicId)
              ?.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
        />
      </div>

      <section className="content-layout" aria-label="当前时间线内容">
        <TimelineContent node={activeNode} />
        <aside className="side-column">
          <TimelineWorkList works={activeNode.keyWorks} />
          <RelatedModules
            prerequisites={activeNode.prerequisites}
            tracks={activeNode.tracks}
            onOpenTrack={openTrack}
          />
        </aside>
      </section>

      {/* ===========================================================
        * CNN 主题主线 · 编年视角下的一条独立子轨
        * 与年份节点平级（在主时间线页内、不嵌入任何年份内容卡）
        * =========================================================== */}
      <section className="topic-track" aria-label="CNN 主题主线" id="cnn-track">
        <header className="topic-track__head">
          <span className="topic-track__tag">主题主线 · vision</span>
          <h2>CNN · 卷积神经网络</h2>
          <p>
            从 2012 AlexNet 把 CNN 跑通，到 2022 ConvNeXt 把 ViT 的训练技术倒灌回来 ——
            这十块积木撑起了整条视觉线。
          </p>
        </header>
        <CnnTrack />
      </section>

      <PrehistoryDrawer open={prehistoryOpen} onClose={closePrehistory} />
    </main>
  );
}
