import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHSAGE_SOURCE_PATH } from "../lib/prose";
import { AggregatorCompareWidget } from "../widgets/AggregatorCompareWidget";
import { ShuffleButtonWidget } from "../widgets/ShuffleButtonWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

const BASE_NEIGHBORS = [1, 3, 6];

function shuffled(arr: number[]): number[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function AggregatorStage({ mechanism2Prose }: Props) {
  const [order, setOrder] = useState(BASE_NEIGHBORS);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:可学习聚合函数
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        mean / max-pool 对邻居的输入顺序不敏感(图本身没有顺序),而 LSTM 聚合器本质上是顺序敏感的 —— 这是它在图任务里的一个已知局限,通常需要先随机打乱邻居顺序来缓解。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GRAPHSAGE_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <ShuffleButtonWidget onShuffle={() => setOrder(shuffled(order))} />
          <AggregatorCompareWidget neighborIds={order} />
          <p className={styles.caption}>
            当前邻居顺序:{order.join(" → ")}。点"打乱邻居顺序"看 order-sensitive(粉点)会跟着移动,但 mean(蓝)/max(绿)始终不变。
          </p>
        </div>
      </div>
    </div>
  );
}
