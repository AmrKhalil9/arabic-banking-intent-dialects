"""Confidence cascade: MARBERTv2 (seed-averaged softmax) answers when its confidence >= tau, otherwise the
retrieval-augmented LLM answers. tau is chosen on the VALIDATION subsample only: the largest tau for which, among the
validation items routed to the LLM, the LLM is at least as accurate as MARBERT. Reported on the test subsamples.
Also reports the full tau curve (for the figure) and the LLM share of traffic."""
import glob, json, os
import numpy as np
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
runs = [d for d in sorted(glob.glob(f"{R}/runs/marbertv2_s*")) if os.path.exists(f"{d}/meta.json")]
def load(rows, split):
    idx = np.array([r["i"] for r in rows]); y = np.array([r["y"] for r in rows]); llm = np.array([r["few"] for r in rows])
    P = np.mean([softmax(np.load(f"{d}/logits.npz")[split])[idx] for d in runs], 0)
    return y, llm, P.max(1), P.argmax(1)
GRID = np.round(np.arange(0.30, 0.991, 0.01), 2)
yv, lv, cv, pv = load(json.load(open(f"{R}/runs/llm_qwen/llm_val.json"))["rows"]["val"], "val")
ok = [t for t in GRID if (cv < t).any() and (lv[cv < t] == yv[cv < t]).mean() >= (pv[cv < t] == yv[cv < t]).mean()]
tau = float(max(ok)) if ok else float(GRID[0])
out = {"tau": tau, "val_bert": float((pv == yv).mean()), "val_llm": float((lv == yv).mean()), "tests": {}, "curve": {}}
for s, rows in json.load(open(f"{R}/runs/llm_qwen/llm.json"))["rows"].items():
    y, llm, c, p = load(rows, s); use = c < tau; fin = np.where(use, llm, p)
    out["tests"][s] = {"n": len(y), "bert": round(float((p == y).mean()) * 100, 1), "llm": round(float((llm == y).mean()) * 100, 1),
                       "cascade": round(float((fin == y).mean()) * 100, 1), "to_llm": round(float(use.mean()) * 100, 1)}
    rng = np.random.default_rng(0); B = rng.integers(0, len(y), (10000, len(y)))      # paired bootstrap over items
    for name, other in (("bert", p), ("llm", llm)):
        d = ((fin == y).astype(float) - (other == y))[B].mean(1) * 100
        out["tests"][s][f"vs_{name}_lo"], out["tests"][s][f"vs_{name}_hi"] = round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)
    out["curve"][s] = [[float(t), round(float((np.where(c < t, llm, p) == y).mean()) * 100, 2), round(float((c < t).mean()) * 100, 2)] for t in GRID]
json.dump(out, open(f"{R}/results/cascade.json", "w"), indent=1)
print("tau (chosen on validation) =", tau); [print(s, v) for s, v in out["tests"].items()]
