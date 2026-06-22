"""Independent check: rebuild gradient's nano-GPT in PyTorch, run autograd, and
diff every gradient against the hand-derived backward exported from the JS/WGSL
code (verify/reference.json). The architecture mirrors src/nn/nanogpt.ts exactly:
pre-LN block, single-head causal attention, tanh-GELU MLP, weight-tied nothing,
biased layer-norm variance, mean cross-entropy."""

import json
import math
import pathlib
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")

ref = json.loads((pathlib.Path(__file__).parent / "reference.json").read_text())
cfg = ref["cfg"]
dE, dFF, V, T = cfg["dEmbed"], cfg["dFF"], cfg["vocab"], cfg["blockSize"]
ids = torch.tensor(ref["ids"], dtype=torch.long)
targets = torch.tensor(ref["targets"], dtype=torch.long)


def param(name, shape):
    t = torch.tensor(ref["weights"][name], dtype=torch.float64).reshape(shape)
    return t.requires_grad_(True)


tokEmb = param("tokEmb", (V, dE))
posEmb = param("posEmb", (cfg["blockSize"], dE))
ln1g, ln1b = param("ln1.g", (dE,)), param("ln1.b", (dE,))
Wq, Wk, Wv, Wo = (param(n, (dE, dE)) for n in ("Wq", "Wk", "Wv", "Wo"))
ln2g, ln2b = param("ln2.g", (dE,)), param("ln2.b", (dE,))
Wff1, bff1 = param("Wff1", (dE, dFF)), param("bff1", (dFF,))
Wff2, bff2 = param("Wff2", (dFF, dE)), param("bff2", (dE,))
lnfg, lnfb = param("lnf.g", (dE,)), param("lnf.b", (dE,))
head = param("head", (dE, V))

EPS = 1e-5


def ln(x, g, b):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS) * g + b


def gelu(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x**3)))


x = tokEmb[ids] + posEmb[:T]
h = ln(x, ln1g, ln1b)
scores = (h @ Wq) @ (h @ Wk).t() / math.sqrt(dE)
scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
attn = torch.softmax(scores, dim=-1)
xa = x + (attn @ (h @ Wv)) @ Wo
h2 = ln(xa, ln2g, ln2b)
xb = xa + (gelu(h2 @ Wff1 + bff1) @ Wff2 + bff2)
logits = ln(xb, lnfg, lnfb) @ head
loss = torch.nn.functional.cross_entropy(logits, targets)
loss.backward()

params = {
    "tokEmb": tokEmb, "posEmb": posEmb, "ln1.g": ln1g, "ln1.b": ln1b,
    "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "ln2.g": ln2g, "ln2.b": ln2b,
    "Wff1": Wff1, "bff1": bff1, "Wff2": Wff2, "bff2": bff2,
    "lnf.g": lnfg, "lnf.b": lnfb, "head": head,
}

dloss = abs(ref["loss"] - loss.item())
print(f"loss   js={ref['loss']:.10f}   torch={loss.item():.10f}   |diff|={dloss:.2e}\n")
print(f"{'param':8} {'shape':11} {'max|abs|':>10} {'rel(1e-3)':>10}")
worst_abs = 0.0
worst_rel = 0.0
for name, t in params.items():
    gt = t.grad.reshape(-1)
    gj = torch.tensor(ref["grads"][name], dtype=torch.float64)
    ae = (gt - gj).abs()
    rel = ae / gj.abs().clamp_min(1e-3)
    worst_abs = max(worst_abs, ae.max().item())
    worst_rel = max(worst_rel, rel.max().item())
    print(f"{name:8} {str(tuple(t.shape)):11} {ae.max().item():>10.2e} {rel.max().item():>10.2e}")

print(f"\nmax abs err:              {worst_abs:.2e}")
print(f"max rel err (floor 1e-3): {worst_rel:.2e}")
ok = worst_abs < 1e-5
print(f"\n{'PASS' if ok else 'FAIL'} — hand-derived gradients match PyTorch autograd to f32 precision")
