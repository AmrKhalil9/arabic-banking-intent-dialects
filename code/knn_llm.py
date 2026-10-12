"""Two non-fine-tuned baselines.
knn : bge-m3 dense embeddings (CLS, L2-normalised); class score = sum of similarities of the k=10 nearest training
      sentences of that class -> logits for val and every test split (same npz format as the trained models).
llm : local LLM through an OpenAI-compatible endpoint, (a) zero-shot with the 77 intent names, (b) 10 retrieved
      training examples (nearest by bge-m3) in the prompt. Run on a fixed, seeded, label-stratified subsample.
Usage: python knn_llm.py knn <out_dir>  |  python knn_llm.py llm <out_dir> <model> <n_per_split>
"""
import json, os, re, sys, time, urllib.request
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, os.path.dirname(__file__))
from common import xy, LABELS, AR2EN, EVAL_SPLITS

MODE, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3"); emb_model = AutoModel.from_pretrained("BAAI/bge-m3").half().to(dev).eval()


@torch.no_grad()
def embed(texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
        out.append(torch.nn.functional.normalize(emb_model(**b).last_hidden_state[:, 0].float(), dim=-1).cpu())
    return torch.cat(out)


Xtr, ytr = xy("train"); Etr = embed(Xtr); ytr_t = torch.tensor(ytr)
data = {s: xy(s) for s in EVAL_SPLITS}


def knn_logits(E, k=10):
    sim = E @ Etr.T; top = sim.topk(k, dim=1)
    scores = torch.zeros(len(E), len(LABELS))
    scores.scatter_add_(1, ytr_t[top.indices], top.values)
    return scores.numpy() * 10.0      # scaled so softmax is not flat; temperature is refitted on val anyway


if MODE == "knn":
    t0 = time.time()
    np.savez_compressed(os.path.join(OUT, "logits.npz"), **{s: knn_logits(embed(data[s][0])) for s in EVAL_SPLITS},
                        **{f"y_{s}": np.array(data[s][1]) for s in EVAL_SPLITS})
    json.dump({"model": "bge-m3-knn10", "train_s": time.time() - t0}, open(os.path.join(OUT, "meta.json"), "w"))
    print("knn done", time.time() - t0)
    sys.exit()

LLM, NPER = sys.argv[3], int(sys.argv[4])
URL = os.environ.get("LLM_URL", "http://127.0.0.1:11434/api/chat")   # Ollama native API: lets us disable thinking
names = "\n".join(f"{i}: {AR2EN[l]} ({l})" for i, l in enumerate(LABELS))
SYS = ("You classify Arabic banking customer messages (any dialect) into exactly one of 77 intents. "
       "Reply with only the intent number.\n\nIntents:\n" + names)


def ask(msgs):
    body = json.dumps({"model": LLM, "messages": msgs, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": 8}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=300))
    txt = r["message"]["content"]; m = re.search(r"\d+", txt)
    p = int(m.group()) if m and int(m.group()) < len(LABELS) else -1
    return p, (time.time() - t) * 1000


rng = np.random.default_rng(0); res = {}
SPL = os.environ.get("LLM_SPLITS", ",".join(x for x in EVAL_SPLITS if x != "val")).split(",")
for s in SPL:
    X, y = data[s]; idx = rng.permutation(len(X))[:NPER]
    E = embed([X[i] for i in idx]); nn = (E @ Etr.T).topk(10, dim=1).indices
    rows = []
    for j, i in enumerate(idx):
        p0, l0 = ask([{"role": "system", "content": SYS}, {"role": "user", "content": X[i]}])
        shots = [{"role": m, "content": c} for t in nn[j].tolist() for m, c in
                 (("user", Xtr[t]), ("assistant", str(ytr[t])))]
        p1, l1 = ask([{"role": "system", "content": SYS}] + shots + [{"role": "user", "content": X[i]}])
        rows.append({"i": int(i), "y": int(y[i]), "zero": p0, "few": p1, "lat_zero": l0, "lat_few": l1})
    res[s] = rows
    acc0 = np.mean([r["zero"] == r["y"] for r in rows]); acc1 = np.mean([r["few"] == r["y"] for r in rows])
    print(f"{s}: zero-shot {acc0:.3f}  10-shot-retrieved {acc1:.3f}  (n={len(rows)})", flush=True)
json.dump({"model": LLM, "n_per_split": NPER, "rows": res}, open(os.path.join(OUT, os.environ.get("LLM_OUT", "llm.json")), "w"))
