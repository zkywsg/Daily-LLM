import { useEffect } from "react";
import { prehistoryNodes } from "../data/timeline";

type PrehistoryDrawerProps = {
  open: boolean;
  onClose: () => void;
};

/**
 * 深度学习「前史」抽屉
 * —— 列出 1948 信息论 / 1958 感知机 / 1986 反向传播 / 1997 LSTM
 * 明确语义：这些不是「被前一代逼出来的突破」，是后来一切深度学习的基石。
 */
export function PrehistoryDrawer({ open, onClose }: PrehistoryDrawerProps) {
  // ESC 关闭
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="prehistory-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="深度学习前史"
      onClick={onClose}
    >
      <div
        className="prehistory-drawer"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="prehistory-drawer__header">
          <div>
            <p className="prehistory-drawer__kicker">深度学习前史 · 非主线</p>
            <h2>2012 之前，重要的不是年份，是基石</h2>
            <p className="prehistory-drawer__subtitle">
              这些里程碑没有进入主时间线，因为它们不是「被前一代逼出来的突破」，
              而是后来一切深度学习的奠基石。建议跟着每张卡片底部的链接进入基础知识区慢慢看。
            </p>
          </div>
          <button
            type="button"
            className="prehistory-drawer__close"
            onClick={onClose}
            aria-label="关闭前史"
          >
            ✕
          </button>
        </header>

        <ol className="prehistory-list">
          {prehistoryNodes.map((node, i) => (
            <li key={node.year} className="prehistory-card">
              <div className="prehistory-card__year">
                <span className="prehistory-card__ordinal">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="prehistory-card__yearnum">{node.year}</span>
              </div>
              <div className="prehistory-card__body">
                <h3>{node.title}</h3>
                <p className="prehistory-card__why">
                  <strong>为什么重要：</strong>
                  {node.why}
                </p>
                <p className="prehistory-card__contrib">
                  <strong>它做了什么：</strong>
                  {node.contribution}
                </p>
                <a className="prehistory-card__link" href={node.foundationPath}>
                  继续学：{node.foundationLabel} ↗
                </a>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
