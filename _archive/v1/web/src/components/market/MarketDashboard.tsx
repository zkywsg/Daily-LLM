import { useMemo, useState } from "react";
import { ExternalLink, Lightbulb, Minus, Sparkles } from "lucide-react";
import styles from "./MarketDashboard.module.css";

type Segment = "direct" | "adjacent";
type MoneyModel = "open" | "book" | "ecosystem" | "subscription";

type Product = {
  id: string;
  name: string;
  segment: Segment;
  form: string;
  positioning: string;
  highlight: string;
  weakness: string;
  monetization: string;
  moneyModel: MoneyModel;
  lesson: string;
  source: string;
};

const products: Product[] = [
  {
    id: "distill",
    name: "Distill",
    segment: "direct",
    form: "互动式研究文章",
    positioning: "网页原生的机器学习解释刊物",
    highlight: "交互图解不是装饰，而是推导和建立直觉的核心媒介。",
    weakness: "主题覆盖不连续；2016–2021 运营后进入无限期休刊。",
    monetization: "免费开放，无直接收费；依靠机构和编辑投入。",
    moneyModel: "open",
    lesson: "学习它的解释品质，但必须建立可复用的交互组件，避免每篇都成为高成本定制项目。",
    source: "https://distill.pub/about/",
  },
  {
    id: "d2l",
    name: "Dive into Deep Learning",
    segment: "direct",
    form: "在线教材 + Notebook",
    positioning: "可运行代码驱动的系统深度学习教材",
    highlight: "把代码、数学、正文、讨论和多框架实现放进同一套内容工程。",
    weakness: "体量大、教材感强；浏览式探索和模型演化叙事相对较弱。",
    monetization: "在线内容免费，纸质书销售形成内容衍生收入。",
    moneyModel: "book",
    lesson: "Markdown 继续做内容正本，Web 端重点负责导航、比较和交互，两者不需要互相替代。",
    source: "https://d2l.ai/",
  },
  {
    id: "hf-learn",
    name: "Hugging Face Learn",
    segment: "direct",
    form: "课程中心 + 开源生态",
    positioning: "以开放课程为入口的 AI 开发者生态",
    highlight: "课程与模型、数据集、Demo、社区天然连接，主题更新速度快。",
    weakness: "内容明显围绕自家工具生态；历史脉络和跨架构统一叙事较弱。",
    monetization: "课程免费获客，导向 PRO、Team、Enterprise 与算力服务。",
    moneyModel: "ecosystem",
    lesson: "免费知识可以承担长期获客；每个知识节点后面都能连接实验、模型和实战资源。",
    source: "https://huggingface.co/learn",
  },
  {
    id: "jay-alammar",
    name: "Jay Alammar",
    segment: "direct",
    form: "视觉长文 + 个人品牌",
    positioning: "一次讲透一个复杂 AI 概念",
    highlight: "用连续插图把 Transformer 等抽象结构转化为容易传播的视觉记忆。",
    weakness: "单作者产能有限；内容体系、练习与学习进度管理较弱。",
    monetization: "免费文章建立影响力，延伸到图书、课程与个人品牌。",
    moneyModel: "book",
    lesson: "金标节点需要一张能被记住和分享的代表性视觉，而不只是内容完整。",
    source: "https://jalammar.github.io/",
  },
  {
    id: "udl",
    name: "Understanding Deep Learning",
    segment: "direct",
    form: "教材 + 教师资源",
    positioning: "围绕权威教材建立的开放教学资源库",
    highlight: "免费 PDF、练习 Notebook、矢量图稿和课程幻灯片资源齐全。",
    weakness: "交互体验分散，更像一本书的配套站点，而非独立 Web 产品。",
    monetization: "开放数字资源扩大影响力，主要通过纸质书销售变现。",
    moneyModel: "book",
    lesson: "未来可以把现有文档衍生成教师包、图表素材与课件，扩大教育使用场景。",
    source: "https://udlbook.github.io/udlbook/",
  },
  {
    id: "fastai",
    name: "fast.ai",
    segment: "direct",
    form: "免费视频课 + Notebook",
    positioning: "实践优先、降低门槛的深度学习课程",
    highlight: "从真实任务切入，学习者很快就能训练并部署可用模型。",
    weakness: "视频课时较长；按架构检索、横向比较与碎片化阅读不够强。",
    monetization: "课程和在线书免费，结合纸质书与开源生态扩大影响力。",
    moneyModel: "book",
    lesson: "每个家族末尾加入一个能产出作品的项目，比继续堆正文更能形成完成感。",
    source: "https://course.fast.ai/",
  },
  {
    id: "deeplearning-ai",
    name: "DeepLearning.AI",
    segment: "adjacent",
    form: "视频课程 + 实验 + 证书",
    positioning: "面向职业提升的规模化 AI 课程平台",
    highlight: "教师品牌、学习路径、实验、证书和持续更新能力构成完整闭环。",
    weakness: "视频和课程主导，探索式阅读与架构全景比较较弱。",
    monetization: "视频免费；实验、测验、项目和证书通过 Pro 订阅收费。",
    moneyModel: "subscription",
    lesson: "成熟路径是基础内容免费、深度练习与成果证明付费；首版仍不必引入登录。",
    source: "https://www.deeplearning.ai/the-batch/introducing-deeplearning-ai-pro",
  },
  {
    id: "brilliant",
    name: "Brilliant",
    segment: "adjacent",
    form: "微课 + 交互练习",
    positioning: "用交互练习和习惯机制驱动的 STEM 平台",
    highlight: "边做边学、即时反馈、学习路径和连续学习机制非常成熟。",
    weakness: "AI 深度和前沿架构覆盖有限；免费层存在每日使用限制。",
    monetization: "Freemium，提供个人、年度与家庭 Premium 订阅。",
    moneyModel: "subscription",
    lesson: "交互要让用户做判断并得到反馈；可以先给少数核心节点设计三分钟微挑战。",
    source: "https://brilliant.org/faq/",
  },
  {
    id: "educative",
    name: "Educative",
    segment: "adjacent",
    form: "文本课程 + 在线编码",
    positioning: "面向开发者职业成长的交互式文本课程",
    highlight: "无需配置环境，阅读、运行代码和项目练习在同一流程完成。",
    weakness: "内容规模很大，但视觉解释并非核心，差异化更依赖题库和职业服务。",
    monetization: "个人分级订阅，并通过项目、面试训练与团队方案提升客单价。",
    moneyModel: "subscription",
    lesson: "付费价值应放在可运行实验、项目反馈和团队学习包，而不是普通文章数量。",
    source: "https://www.educative.io/unlimited",
  },
];

const moneyLabels: Record<MoneyModel, string> = {
  open: "开放支持",
  book: "图书衍生",
  ecosystem: "生态导流",
  subscription: "订阅",
};

export function MarketDashboard() {
  const [segment, setSegment] = useState<"all" | Segment>("all");
  const [money, setMoney] = useState<"all" | MoneyModel>("all");
  const [selectedId, setSelectedId] = useState(products[0].id);

  const filtered = useMemo(
    () =>
      products.filter(
        (product) =>
          (segment === "all" || product.segment === segment) &&
          (money === "all" || product.moneyModel === money),
      ),
    [money, segment],
  );

  const selected =
    products.find((product) => product.id === selectedId) ?? products[0];

  return (
    <div className={styles.page}>
      <header className={styles.intro}>
        <div>
          <p className={styles.eyebrow}>MARKET MAP · 2026</p>
          <h1>AI 学习产品竞品看板</h1>
          <p className={styles.lede}>
            直接同类看内容与体验，邻近替代看获客与变现。
          </p>
        </div>
        <div className={styles.metrics} aria-label="样本概览">
          <div><strong>9</strong><span>产品样本</span></div>
          <div><strong>6</strong><span>直接同类</span></div>
          <div><strong>4</strong><span>变现模型</span></div>
        </div>
      </header>

      <section className={styles.toolbar} aria-label="筛选竞品">
        <div className={styles.segmentControl}>
          {([
            ["all", "全部"],
            ["direct", "直接同类"],
            ["adjacent", "邻近替代"],
          ] as const).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={segment === value ? styles.activeFilter : styles.filter}
              onClick={() => setSegment(value)}
              aria-pressed={segment === value}
            >
              {label}
            </button>
          ))}
        </div>
        <label className={styles.moneyFilter}>
          <span>变现方式</span>
          <select
            value={money}
            onChange={(event) => setMoney(event.target.value as "all" | MoneyModel)}
          >
            <option value="all">全部方式</option>
            <option value="open">开放支持</option>
            <option value="book">图书衍生</option>
            <option value="ecosystem">生态导流</option>
            <option value="subscription">订阅</option>
          </select>
        </label>
        <span className={styles.resultCount}>显示 {filtered.length} / {products.length}</span>
      </section>

      <main className={styles.workspace}>
        <section className={styles.listPanel} aria-label="竞品列表">
          <div className={styles.tableHeader} aria-hidden="true">
            <span>产品</span><span>核心形态</span><span>变现</span>
          </div>
          <div className={styles.productList}>
            {filtered.map((product) => (
              <button
                className={`${styles.productRow} ${selected.id === product.id ? styles.selectedRow : ""}`}
                type="button"
                key={product.id}
                onClick={() => setSelectedId(product.id)}
                aria-pressed={selected.id === product.id}
              >
                <span className={styles.productName}>
                  <span className={`${styles.typeDot} ${product.segment === "adjacent" ? styles.adjacentDot : ""}`} />
                  <span><strong>{product.name}</strong><small>{product.positioning}</small></span>
                </span>
                <span>{product.form}</span>
                <span className={styles.moneyTag}>{moneyLabels[product.moneyModel]}</span>
              </button>
            ))}
            {filtered.length === 0 && <p className={styles.empty}>当前筛选下没有产品。</p>}
          </div>
          <div className={styles.legend}>
            <span><i className={styles.typeDot} />直接同类</span>
            <span><i className={`${styles.typeDot} ${styles.adjacentDot}`} />邻近替代</span>
          </div>
        </section>

        <aside className={styles.detailPanel} aria-live="polite">
          <div className={styles.detailTitle}>
            <div><span>{selected.form}</span><h2>{selected.name}</h2></div>
            <a href={selected.source} target="_blank" rel="noreferrer" aria-label={`查看 ${selected.name} 官方页面`}>
              官方来源 <ExternalLink size={15} aria-hidden="true" />
            </a>
          </div>
          <div className={styles.detailSection}>
            <h3><Sparkles size={17} aria-hidden="true" />亮点</h3>
            <p>{selected.highlight}</p>
          </div>
          <div className={styles.detailSection}>
            <h3><Minus size={17} aria-hidden="true" />短板</h3>
            <p>{selected.weakness}</p>
          </div>
          <div className={styles.detailSection}>
            <h3>变现方式</h3>
            <p>{selected.monetization}</p>
          </div>
          <div className={styles.lesson}>
            <h3><Lightbulb size={17} aria-hidden="true" />给 Daily-LLM 的启发</h3>
            <p>{selected.lesson}</p>
          </div>
        </aside>
      </main>

      <section className={styles.opportunity} aria-labelledby="opportunity-title">
        <div className={styles.opportunityIntro}>
          <p className={styles.eyebrow}>WHITE SPACE</p>
          <h2 id="opportunity-title">Daily-LLM 应该占据的位置</h2>
        </div>
        <div><strong>01</strong><h3>互动解释 × 系统地图</h3><p>把模型演化全景与单节点互动解释连成同一条学习路径。</p></div>
        <div><strong>02</strong><h3>中文优先 × 国际正本</h3><p>中文叙事、英文术语和论文出处并存，不做简单翻译站。</p></div>
        <div><strong>03</strong><h3>免费入口 × 付费进阶</h3><p>基础浏览继续开放，未来让实验、项目反馈和团队学习包收费。</p></div>
      </section>

      <section className={styles.roadmap} aria-labelledby="roadmap-title">
        <header className={styles.roadmapHeader}>
          <div>
            <p className={styles.eyebrow}>OPTIMIZATION ROADMAP</p>
            <h2 id="roadmap-title">我建议优先优化的方向</h2>
          </div>
          <p>先增强公开产品的理解与传播，再补学习闭环，最后验证收费。</p>
        </header>

        <div className={styles.roadmapGrid}>
          <section className={styles.roadmapLane}>
            <div className={styles.laneTitle}>
              <span className={styles.priorityNow}>P0</span>
              <div><h3>现在做</h3><p>上线前必须完成</p></div>
            </div>
            <article className={styles.actionItem}>
              <span>01</span>
              <div>
                <h4>把首页变成明确入口</h4>
                <p>让首次访问者在 10 秒内知道项目是什么，并提供“从零开始、按家族浏览、看精选互动”三个入口。</p>
                <small>验收：陌生用户不需要解释就能找到第一篇内容。</small>
              </div>
            </article>
            <article className={styles.actionItem}>
              <span>02</span>
              <div>
                <h4>集中打磨 6–8 个金标节点</h4>
                <p>不要平均改完所有页面。优先覆盖 CNN、Transformer、BERT、GPT、Diffusion、RAG、MoE、推理模型。</p>
                <small>验收：每个节点都有一个可操作、可截图传播的核心解释。</small>
              </div>
            </article>
            <article className={styles.actionItem}>
              <span>03</span>
              <div>
                <h4>补齐公开传播基础</h4>
                <p>为每个节点生成独立标题、摘要、稳定网址、分享预览与站点地图，让内容能被搜索和单独分享。</p>
                <small>验收：分享任一节点时，标题和简介都准确对应该模型。</small>
              </div>
            </article>
          </section>

          <section className={styles.roadmapLane}>
            <div className={styles.laneTitle}>
              <span className={styles.priorityNext}>P1</span>
              <div><h3>接着做</h3><p>形成学习闭环</p></div>
            </div>
            <article className={styles.actionItem}>
              <span>04</span>
              <div>
                <h4>建立三条结构化学习路径</h4>
                <p>先做“深度学习入门、LLM 核心、应用与 Agent”三条路线；无需登录，用浏览器保存阅读进度。</p>
                <small>验收：用户完成一篇后，永远知道下一篇该看什么。</small>
              </div>
            </article>
            <article className={styles.actionItem}>
              <span>05</span>
              <div>
                <h4>让文档与 Web 真正共用一份内容</h4>
                <p>Markdown 继续作为正本，网页只增加导航和交互；页面标注来源与更新时间，避免两套内容逐渐分叉。</p>
                <small>验收：正文修改一次，文档和网页同步生效。</small>
              </div>
            </article>
            <article className={styles.actionItem}>
              <span>06</span>
              <div>
                <h4>建立最小数据反馈</h4>
                <p>关注节点访问、阅读完成、下一篇点击和互动使用情况，并给每页增加一句“哪里没讲清楚”。</p>
                <small>验收：下一轮内容排序由真实使用数据决定。</small>
              </div>
            </article>
          </section>

          <section className={styles.roadmapLane}>
            <div className={styles.laneTitle}>
              <span className={styles.priorityLater}>P2</span>
              <div><h3>验证后做</h3><p>探索可持续变现</p></div>
            </div>
            <article className={styles.actionItem}>
              <span>07</span>
              <div>
                <h4>先卖成果，不卖基础阅读</h4>
                <p>第一批收费尝试可选实战项目包、带答案的练习、教师课件包或团队学习包，正文继续公开。</p>
                <small>验收：先用预约或小规模预售验证需求，再建设支付系统。</small>
              </div>
            </article>
            <article className={styles.actionItem}>
              <span>08</span>
              <div>
                <h4>建立稳定的内容生产节奏</h4>
                <p>固定“旧节点升级 + 新节点补充”的月度节奏，公开更新日志，让持续更新本身成为产品信誉。</p>
                <small>验收：每月都有可被用户感知的一次内容版本更新。</small>
              </div>
            </article>
          </section>
        </div>
      </section>

      <p className={styles.disclaimer}>判断基于产品公开页面与公开体验；价格和方案以各产品实时页面为准。</p>
    </div>
  );
}
