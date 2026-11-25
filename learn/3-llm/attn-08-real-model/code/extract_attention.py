"""用 Hugging Face transformers 取出 GPT-2 每层每头的注意力权重，导出成 JSON，
再把 JSON 粘贴到 demo/index.html 里画热力图。

状态：脚本已通过 Python 语法检查，但本仓库没有安装 torch / transformers，**还没有真实运行过**。
首次运行请先跑一遍并核对输出形状。

安装：  pip install torch transformers
用法：  python extract_attention.py "The cat did not cross the road because it was too tired" > attn.json
"""
import json
import sys

import torch
from transformers import AutoModel, AutoTokenizer

text = sys.argv[1] if len(sys.argv) > 1 else "The cat did not cross the road because it was too tired"
tok = AutoTokenizer.from_pretrained("gpt2")
# attn_implementation="eager"：保证能返回完整的注意力权重（有些快速实现不返回）
model = AutoModel.from_pretrained("gpt2", attn_implementation="eager", output_attentions=True)
model.eval()

enc = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model(**enc)

# out.attentions：长度 = 层数（GPT-2 small 为 12）的元组，每个元素形状 [batch, heads, n, n]
layers = [a[0].tolist() for a in out.attentions]          # [layer][head][n][n]
tokens = [t.replace("Ġ", " ") for t in tok.convert_ids_to_tokens(enc["input_ids"][0])]
n = len(tokens)
assert all(len(l) == model.config.n_head and len(l[0]) == n for l in layers), "形状与预期不符"
json.dump({"model": "gpt2", "tokens": tokens, "layers": layers}, sys.stdout)
