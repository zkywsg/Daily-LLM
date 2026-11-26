"""教学语料：用一套固定规则随机造句。规则已知，所以能算出“损失最低能到多少”。"""
import numpy as np

# 主语 -> (爱吃的, 住的地方, 爱做的事)
ANIMALS = {
    "小猫": ("鱼", "沙发上", "睡觉"),
    "小狗": ("骨头", "院子里", "跑步"),
    "小鸟": ("虫子", "大树上", "唱歌"),
    "兔子": ("胡萝卜", "草地上", "蹦跳"),
    "熊猫": ("竹子", "高山里", "爬树"),
}
TEMPLATES = [
    "{s}喜欢吃{food}。",
    "{s}住在{place}。",
    "{s}最爱{act}。",
    "{s}吃了{food}，很开心。",
]


def sentence(s, t):
    food, place, act = ANIMALS[s]
    return TEMPLATES[t].format(s=s, food=food, place=place, act=act)


def make_text(n_sentences, seed):
    rng = np.random.default_rng(seed)
    names = list(ANIMALS)
    return "".join(sentence(names[rng.integers(len(names))], rng.integers(len(TEMPLATES)))
                   for _ in range(n_sentences))


def entropy_floor():
    """每个字符的最小平均交叉熵（nats）。
    每句话只有两处随机：选哪种动物（5 选 1）和选哪个句式（4 选 1），其余字符都被规则定死。
    所以“每句的熵 / 每句平均字数”就是任何模型能达到的下限（假设它看得到整句开头）。"""
    h = np.log(len(ANIMALS)) + np.log(len(TEMPLATES))
    length = np.mean([len(sentence(s, t)) for s in ANIMALS for t in range(len(TEMPLATES))])
    return float(h / length)


def vocab():
    chars = sorted({c for s in ANIMALS for t in range(len(TEMPLATES)) for c in sentence(s, t)})
    return chars
