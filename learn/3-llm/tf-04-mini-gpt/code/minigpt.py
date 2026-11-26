"""迷你 GPT：只依赖 numpy，前向和反向全部手写。
结构：词向量 + 位置向量 → N × [x + Attn(LN(x)); x + FFN(LN(x))] → LN → 与词向量转置相乘得 logits。
"""
import warnings

import numpy as np

# 部分 macOS 的 NumPy（Accelerate 后端）会在结果完全正常时对 matmul 报虚假的溢出警告。
# 这里屏蔽它；test_grad.py 里另有断言保证所有梯度都是有限数。
warnings.filterwarnings("ignore", message=".*encountered in matmul")

C = np.sqrt(2 / np.pi)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def gelu(x):  # GPT-2 使用的 tanh 近似
    return 0.5 * x * (1 + np.tanh(C * (x + 0.044715 * x ** 3)))


def dgelu(x):
    t = np.tanh(C * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C * (1 + 3 * 0.044715 * x * x)


def ln_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    s = np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)
    xh = (x - mu) / s
    return g * xh + b, (xh, s, g)


def ln_bwd(dy, cache):
    xh, s, g = cache
    dg = (dy * xh).sum((0, 1))
    db = dy.sum((0, 1))
    d = dy * g
    dx = (d - d.mean(-1, keepdims=True) - xh * (d * xh).mean(-1, keepdims=True)) / s
    return dx, dg, db


class MiniGPT:
    def __init__(self, vocab, d=32, n_layers=2, n_heads=4, d_ff=128, ctx=24, seed=0,
                 causal=True, use_pos=True, residual=True, dtype=np.float32):
        assert d % n_heads == 0
        self.cfg = dict(vocab=vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, ctx=ctx)
        self.causal, self.use_pos, self.residual = causal, use_pos, residual
        rng = np.random.default_rng(seed)
        r = lambda *s, std=0.02: (rng.standard_normal(s) * std).astype(dtype)
        p = {"E": r(vocab, d), "P": r(ctx, d)}
        res_std = 0.02 / np.sqrt(2 * n_layers)  # 写回残差流的矩阵初始化得小一些（GPT-2 的做法）
        for l in range(n_layers):
            for n in ("Wq", "Wk", "Wv"):
                p[f"{l}.{n}"] = r(d, d)
            p[f"{l}.Wo"] = r(d, d, std=res_std)
            p[f"{l}.W1"] = r(d, d_ff)
            p[f"{l}.b1"] = np.zeros(d_ff, dtype)
            p[f"{l}.W2"] = r(d_ff, d, std=res_std)
            p[f"{l}.b2"] = np.zeros(d, dtype)
            for n in ("ln1", "ln2"):
                p[f"{l}.{n}.g"] = np.ones(d, dtype)
                p[f"{l}.{n}.b"] = np.zeros(d, dtype)
        p["lnf.g"], p["lnf.b"] = np.ones(d, dtype), np.zeros(d, dtype)
        self.p = p

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def forward(self, idx, keep_attn=False):
        """idx: [B, n] 整数。返回 logits [B, n, vocab] 和反向要用的缓存。"""
        p, c = self.p, self.cfg
        B, n = idx.shape
        h, dk = c["n_heads"], c["d"] // c["n_heads"]
        x = p["E"][idx] + (p["P"][:n] if self.use_pos else 0)
        split = lambda M: M.reshape(B, n, h, dk).transpose(0, 2, 1, 3)
        merge = lambda M: M.transpose(0, 2, 1, 3).reshape(B, n, h * dk)
        mask = np.triu(np.ones((n, n), bool), 1)
        caches, attns = [], []
        for l in range(c["n_layers"]):
            # ① 注意力子层
            z, c1 = ln_fwd(x, p[f"{l}.ln1.g"], p[f"{l}.ln1.b"])
            Q, K, V = (split(z @ p[f"{l}.{n_}"]) for n_ in ("Wq", "Wk", "Wv"))
            S = Q @ K.swapaxes(-1, -2) / np.sqrt(dk)
            if self.causal:
                S = np.where(mask, -1e9, S)
            A = softmax(S)
            O = merge(A @ V)
            a_out = O @ p[f"{l}.Wo"]
            x = (x + a_out) if self.residual else a_out
            # ② 前馈子层
            z2, c2 = ln_fwd(x, p[f"{l}.ln2.g"], p[f"{l}.ln2.b"])
            u = z2 @ p[f"{l}.W1"] + p[f"{l}.b1"]
            g = gelu(u)
            f_out = g @ p[f"{l}.W2"] + p[f"{l}.b2"]
            x = (x + f_out) if self.residual else f_out
            caches.append((c1, z, Q, K, V, A, O, c2, z2, u, g))
            if keep_attn:
                attns.append(A)
        y, cf = ln_fwd(x, p["lnf.g"], p["lnf.b"])
        logits = y @ p["E"].T  # 权重共享：输出层复用词向量表
        return logits, (idx, caches, cf, y, attns)

    def loss_and_grads(self, idx, tgt):
        p, c = self.p, self.cfg
        logits, (idx, caches, cf, y, _) = self.forward(idx)
        B, n = idx.shape
        h, dk = c["n_heads"], c["d"] // c["n_heads"]
        P = softmax(logits)
        bi, ni = np.arange(B)[:, None], np.arange(n)[None, :]
        loss = float(-np.log(P[bi, ni, tgt] + 1e-12).mean())
        dlog = P.copy()
        dlog[bi, ni, tgt] -= 1
        dlog /= B * n
        G = {k: np.zeros_like(v) for k, v in p.items()}
        T = lambda M: M.reshape(-1, M.shape[-1])
        G["E"] += T(dlog).T @ T(y)
        dx, G["lnf.g"], G["lnf.b"] = ln_bwd(dlog @ p["E"], cf)
        split = lambda M: M.reshape(B, n, h, dk).transpose(0, 2, 1, 3)
        merge = lambda M: M.transpose(0, 2, 1, 3).reshape(B, n, h * dk)
        for l in reversed(range(c["n_layers"])):
            c1, z, Q, K, V, A, O, c2, z2, u, g = caches[l]
            # ② 前馈子层（反向）
            dout = dx
            G[f"{l}.W2"], G[f"{l}.b2"] = T(g).T @ T(dout), dout.sum((0, 1))
            du = (dout @ p[f"{l}.W2"].T) * dgelu(u)
            G[f"{l}.W1"], G[f"{l}.b1"] = T(z2).T @ T(du), du.sum((0, 1))
            dxs, G[f"{l}.ln2.g"], G[f"{l}.ln2.b"] = ln_bwd(du @ p[f"{l}.W1"].T, c2)
            dx = dx + dxs if self.residual else dxs
            # ① 注意力子层（反向）
            dout = dx
            G[f"{l}.Wo"] = T(O).T @ T(dout)
            dO = split(dout @ p[f"{l}.Wo"].T)
            dA = dO @ V.swapaxes(-1, -2)
            dV = A.swapaxes(-1, -2) @ dO
            dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(dk)
            dQ = dS @ K
            dK = dS.swapaxes(-1, -2) @ Q
            dz = 0
            for name, dM in (("Wq", dQ), ("Wk", dK), ("Wv", dV)):
                dM = merge(dM)
                G[f"{l}.{name}"] = T(z).T @ T(dM)
                dz = dz + dM @ p[f"{l}.{name}"].T
            dxs, G[f"{l}.ln1.g"], G[f"{l}.ln1.b"] = ln_bwd(dz, c1)
            dx = dx + dxs if self.residual else dxs
        np.add.at(G["E"], idx, dx)
        if self.use_pos:
            G["P"][:n] = dx.sum(0)
        return loss, G

    def generate(self, ids, n_new, temperature=1.0, rng=None):
        ids = list(ids)
        for _ in range(n_new):
            window = np.array([ids[-self.cfg["ctx"]:]])
            logits = self.forward(window)[0][0, -1]
            if temperature == 0:
                ids.append(int(logits.argmax()))
            else:
                pr = softmax(logits / temperature).astype(np.float64)
                ids.append(int(rng.choice(len(pr), p=pr / pr.sum())))
        return ids


class AdamW:
    def __init__(self, params, lr=3e-3, wd=0.01, b1=0.9, b2=0.99, eps=1e-8):
        self.lr, self.wd, self.b1, self.b2, self.eps, self.t = lr, wd, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, p, G, lr_scale=1.0):
        self.t += 1
        norm = np.sqrt(sum((g ** 2).sum() for g in G.values()))
        clip = min(1.0, 1.0 / (norm + 1e-9))  # 梯度裁剪
        lr = self.lr * lr_scale
        for k in p:
            g = G[k] * clip
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            decay = self.wd if p[k].ndim == 2 else 0.0  # 偏置和归一化参数不做权重衰减
            p[k] -= (lr * (mh / (np.sqrt(vh) + self.eps) + decay * p[k])).astype(p[k].dtype)
