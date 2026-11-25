// RAG demo:玩具文档库 + 3 个查询 + 手工 curated 检索分数。
// 不真跑 embedding 模型,但模拟一个能让 viewer 看明白\"语义检索 + 拼接\"的样子。

export interface Doc {
  id: string;
  title: string;
  text: string;
  /** 2D 投影坐标(用于散点图) */
  pos: { x: number; y: number };
  group: "biology" | "history" | "physics" | "code" | "cooking";
}

export const CORPUS: Doc[] = [
  // biology cluster
  { id: "B1", title: "光合作用基础", text: "植物通过叶绿体把阳光转换成糖。光合作用的总反应:6CO₂+6H₂O→C₆H₁₂O₆+6O₂。", pos: { x: 140, y: 90 }, group: "biology" },
  { id: "B2", title: "DNA 双螺旋", text: "DNA 由两条反向平行链组成,A-T、G-C 通过氢键配对。Watson & Crick 1953 解出结构。", pos: { x: 180, y: 130 }, group: "biology" },
  { id: "B3", title: "细胞分裂", text: "有丝分裂分为前期/中期/后期/末期,染色体复制后均分到两个子细胞。", pos: { x: 110, y: 150 }, group: "biology" },

  // history cluster
  { id: "H1", title: "工业革命", text: "18 世纪英国蒸汽机商业化,工厂取代手工作坊,棉纺织业率先变革。", pos: { x: 480, y: 80 }, group: "history" },
  { id: "H2", title: "二战转折点", text: "斯大林格勒战役 1942-43 是东线转折,中途岛海战 1942 是太平洋转折。", pos: { x: 520, y: 130 }, group: "history" },

  // physics cluster
  { id: "P1", title: "相对论与时间", text: "爱因斯坦狭义相对论指出时间和空间不是绝对的,光速在所有惯性系中相同。", pos: { x: 360, y: 220 }, group: "physics" },
  { id: "P2", title: "量子叠加", text: "薛定谔的猫:量子粒子在测量前处于叠加态,观测才坍缩为某一本征态。", pos: { x: 420, y: 260 }, group: "physics" },

  // code cluster
  { id: "C1", title: "Python list comprehension", text: "[x*2 for x in range(10)] 一行生成新列表,比 for 循环更紧凑。", pos: { x: 80, y: 280 }, group: "code" },
  { id: "C2", title: "git rebase", text: "git rebase 把当前分支的提交重放到目标分支上,保持线性历史。", pos: { x: 130, y: 320 }, group: "code" },

  // cooking cluster
  { id: "K1", title: "完美煮蛋", text: "8 分钟得到溏心蛋,12 分钟得到全熟。先冷水入锅,水沸后开始计时。", pos: { x: 290, y: 360 }, group: "cooking" },
];

export const GROUP_COLOR: Record<Doc["group"], string> = {
  biology: "#10b981",
  history: "#f59e0b",
  physics: "#3b82f6",
  code: "#ec4899",
  cooking: "#a78bfa",
};

export interface Query {
  text: string;
  /** query 在 2D 空间的位置 */
  pos: { x: number; y: number };
  /** 哪些 doc id 是相关的(curated) */
  relevant: string[];
  /** 把召回的 chunk 拼进 prompt 后,模型应该回答的内容 */
  answer: string;
}

export const QUERIES: Query[] = [
  {
    text: "光合作用的化学方程式是什么?",
    pos: { x: 145, y: 100 },
    relevant: ["B1", "B3", "B2"],
    answer:
      "光合作用的总反应是 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂,即二氧化碳和水在叶绿体中通过光能合成葡萄糖和氧气。",
  },
  {
    text: "斯大林格勒战役为什么重要?",
    pos: { x: 510, y: 110 },
    relevant: ["H2", "H1"],
    answer:
      "斯大林格勒战役(1942-43)是二战东线的转折点,德军在此役战略性失败后转入防御。",
  },
  {
    text: "什么是 git rebase?",
    pos: { x: 130, y: 320 },
    relevant: ["C2", "C1"],
    answer:
      "git rebase 把当前分支的提交\"重放\"到目标分支上,得到线性的提交历史 —— 跟 merge 留下分叉 commit 不同。",
  },
];

/**
 * 模拟 cosine similarity:用 2D 位置之间的负距离当 logit。
 * query 在某个聚类附近就该召回该聚类的 docs。
 */
export function similarity(query: Query, doc: Doc): number {
  const d = Math.hypot(query.pos.x - doc.pos.x, query.pos.y - doc.pos.y);
  // 距离越小 sim 越大;归一到 [0, 1]
  return Math.max(0, 1 - d / 400);
}

/** top-k 检索 */
export function topK(query: Query, k: number): Array<{ doc: Doc; sim: number }> {
  return CORPUS.map((d) => ({ doc: d, sim: similarity(query, d) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, k);
}
