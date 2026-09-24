"""
XTJ-LM 0.1 生成脚本
加载 checkpoint → 重建模型与 tokenizer → 给定开头生成中文
"""
import os
import torch
from model import XTJLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 加载 checkpoint ----------
checkpoint = torch.load(
    os.path.join(BASE_DIR, "xtj_lm_0.1.pt"),
    map_location=device,
    weights_only=False,
)
config = checkpoint["config"]
chars = checkpoint["chars"]

# ---------- 重建 tokenizer ----------
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

# ---------- 重建模型并载入权重 ----------
model = XTJLM(**config).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ---------- 给定开头生成 ----------
prompts = ["今天", "我喜欢", "你好", "人工智能"]
for prompt in prompts:
    start = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=30)
    print(f"开头「{prompt}」→ {decode(out[0].tolist())}")
    print("-" * 40)
