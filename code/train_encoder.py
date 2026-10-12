"""Fine-tune a BERT-style encoder for 77-way intent classification on the official MSA+PAL training split.
Best epoch chosen by validation macro-F1. Saves logits for val and every test split, plus GPU/CPU latency.
Usage: python train_encoder.py <hf_model_id> <seed> <out_dir>
"""
import json, os, sys, time
import numpy as np, torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
sys.path.insert(0, os.path.dirname(__file__))
from common import xy, LABELS, EVAL_SPLITS

MID, SEED, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
EPOCHS, BS, LR, MAXLEN = 5, 32, 3e-5, 64
torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForSequenceClassification.from_pretrained(MID, num_labels=len(LABELS)).to(dev)
Xtr, ytr = xy("train")
data = {s: xy(s) for s in EVAL_SPLITS}


def enc(texts):
    return tok(texts, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")


@torch.no_grad()
def logits_of(texts, bs=128):
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        b = {k: v.to(dev) for k, v in enc(texts[i:i + bs]).items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**b).logits.float().cpu())
    return torch.cat(out).numpy()


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ((len(Xtr) + BS - 1) // BS)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler("cuda")
rng = np.random.default_rng(SEED); best, best_ep, t0 = -1, -1, time.time()
for ep in range(EPOCHS):
    model.train(); order = rng.permutation(len(Xtr))
    for i in range(0, len(order), BS):
        idx = order[i:i + BS]
        b = {k: v.to(dev) for k, v in enc([Xtr[j] for j in idx]).items()}
        y = torch.tensor([ytr[j] for j in idx], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(**b).logits.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
    lv = logits_of(data["val"][0]); f = f1_score(data["val"][1], lv.argmax(1), average="macro")
    print(f"ep {ep+1} val macroF1 {f:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if f > best:
        best, best_ep = f, ep + 1
        np.savez_compressed(os.path.join(OUT, "logits.npz"), **{s: logits_of(data[s][0]) for s in EVAL_SPLITS},
                            **{f"y_{s}": np.array(data[s][1]) for s in EVAL_SPLITS})
# latency, batch size 1, on 200 validation sentences, GPU then CPU (8 threads)
sample = data["val"][0][:200]
def lat(device):
    model.to(device).eval(); ts = []
    with torch.no_grad():
        for t in sample:
            b = {k: v.to(device) for k, v in enc([t]).items()}; s = time.perf_counter(); model(**b)
            if device.type == "cuda": torch.cuda.synchronize()
            ts.append((time.perf_counter() - s) * 1000)
    return float(np.median(ts[10:]))
torch.set_num_threads(8)
meta = {"model": MID, "seed": SEED, "best_epoch": best_ep, "val_macro_f1": best, "train_s": time.time() - t0,
        "params": sum(p.numel() for p in model.parameters()), "lat_gpu_ms": lat(dev), "lat_cpu8_ms": lat(torch.device("cpu"))}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1); print(meta)
