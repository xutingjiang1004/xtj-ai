"""
XTJ-LM 0.1 训练脚本
读取 data/train.txt → 字符级 tokenizer → 训练 Transformer → 保存 checkpoint
"""
import os
import torch
from model import XTJLM

# ---------- 超参数 ----------
batch_size = 16
block_size = 32
n_embd = 64
n_head = 4
n_layer = 2
max_iters = 3000
eval_interval = 200
eval_iters = 20
learning_rate = 3e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ---------- 读取数据 + 字符级 tokenizer ----------
with open(os.path.join(BASE_DIR, "data", "train.txt"), encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    """随机取 batch_size 个长度为 block_size 的片段，目标右移一位"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = get_batch()
        losses[k] = model(xb, yb)[1].item()
    model.train()
    return losses.mean().item()

# ---------- 模型 + 优化器 ----------
model = XTJLM(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head,
              n_layer=n_layer, block_size=block_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"设备: {device} | 词表: {vocab_size} | 参数量: {n_params/1e3:.0f}K")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------- 训练循环 ----------
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        loss = estimate_loss()
        print(f"step {it:4d} | train loss {loss:.4f}")
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------- 保存 checkpoint ----------
checkpoint = {
    "model_state": model.state_dict(),
    "config": {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
    },
    "chars": chars,
}
save_path = os.path.join(BASE_DIR, "xtj_lm_0.1.pt")
torch.save(checkpoint, save_path)
print("训练完成，权重已保存:", save_path)
