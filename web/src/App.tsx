import { useMemo, useState } from "react";
import { RelatedModules } from "./components/RelatedModules";
import { TimelineAxis } from "./components/TimelineAxis";
import { TimelineContent } from "./components/TimelineContent";
import { TimelineWorkList } from "./components/TimelineWorkList";
import { getNodeByYear, timelineNodes } from "./data/timeline";

const DEFAULT_YEAR = "2012";

function readInitialYear(): string {
  const hashYear = window.location.hash.replace("#", "");

  return getNodeByYear(hashYear)?.year ?? DEFAULT_YEAR;
}

export default function App() {
  const [activeYear, setActiveYear] = useState(readInitialYear);
  const activeNode = useMemo(
    () => getNodeByYear(activeYear) ?? getNodeByYear(DEFAULT_YEAR)!,
    [activeYear],
  );

  function selectYear(year: string) {
    setActiveYear(year);
    window.location.hash = year;
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="hero__kicker">Daily-LLM</p>
          <h1>深度学习与大模型演进时间线</h1>
          <p className="hero__summary">
            每一个技术的出现，背后都有一个不得不解决的问题。
          </p>
        </div>
        <div className="hero__status" aria-label="当前节点">
          <span>1948 → 2025</span>
          <strong>
            当前：{activeNode.year} · {activeNode.shortTitle}
          </strong>
        </div>
      </header>

      <TimelineAxis
        activeYear={activeYear}
        nodes={timelineNodes}
        onSelect={selectYear}
      />

      <section className="content-layout" aria-label="当前时间线内容">
        <TimelineContent node={activeNode} />
        <aside className="side-column">
          <TimelineWorkList works={activeNode.keyWorks} />
          <RelatedModules modules={activeNode.relatedModules} />
        </aside>
      </section>
    </main>
  );
}
