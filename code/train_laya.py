"""Fine-tune Laya (multilingual 322M) for 77-way ArBanking77 intents with Laya's RLCD policy-gradient + soft-CE recipe,
on the official MSA+PAL training split. Choice temperature fitted on the official validation split.
Saves raw logits for val + every test split (same format as train_encoder.py) and latency.
Usage: python train_laya.py <seed> <out_dir> [epochs]
"""
import json, os, random, re, sys, time
import numpy as np, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_sequence, render_options, build_model, proper_reward, QTYPES
sys.path.insert(0, os.path.dirname(__file__))
from common import xy, LABELS, AR2EN, EVAL_SPLITS
from coarse import CATS, cat_of

SEED, OUT = int(sys.argv[1]), sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LOSS = os.environ.get("LOSS", "rlcd")          # rlcd = Laya's recipe; ce = plain cross-entropy (BERT-style protocol)
MODE = os.environ.get("MODE", "fine")          # fine = 77 intents; coarse = 12 contact-centre categories
SELECT = os.environ.get("SELECT", "0") == "1"  # pick the best epoch on validation (same protocol as train_encoder.py)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUT, exist_ok=True)
root = snapshot_download("convaiinnovations/laya"); _fix_tokenizer_config(root)
mdir = os.path.join(root, "multilingual"); _fix_tokenizer_config(mdir)
tok = AutoTokenizer.from_pretrained(os.path.join(mdir, "tokenizer"))
cfg = json.load(open(os.path.join(mdir, "rl_agent_config.json")))
cfg.update(gradient_checkpointing=True, max_len=1024, head_max_len=1024)   # 77 options need ~975 tokens
key = lambda en: re.sub(r"[^a-z0-9]+", "_", en.lower()).strip("_")
if MODE == "fine":
    crit = {key(AR2EN[l]): AR2EN[l] for l in LABELS}          # option i <-> LABELS[i]; English names keep all 77 in 1024 tokens
    qdef = {"t": "choice", "ins": "Which banking intent does this customer message express?", "crit": crit}
    remap = lambda y: y
else:
    crit = dict(CATS)                                           # option j <-> list(CATS)[j]
    qdef = {"t": "choice", "ins": "What is the customer asking about in this banking support message?", "crit": crit}
    _c = [list(CATS).index(cat_of(AR2EN[l])) for l in LABELS]; remap = lambda y: _c[y]
K = len(render_options({"t": "choice", "crit": crit})); assert K == len(crit), K
_xy = xy
def xy(split):
    X, Y = _xy(split); return X, [remap(v) for v in Y]


def items_of(texts, ys):
    out, drop = [], 0
    for t, y in zip(texts, ys):
        seq, mk = build_sequence(tok, t, qdef, cfg["max_len"], cfg["head_max_len"])
        if len(mk) != K: drop += 1; continue
        tg = [0.0] * K; tg[y] = 1.0
        out.append({"ids": seq, "markers": mk, "qtype": QTYPES["choice"], "target": tg, "label": y})
    return out, drop


def collate(its):
    n, L = len(its), max(len(i["ids"]) for i in its)
    ids = torch.full((n, L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, K), dtype=torch.long); mm = torch.ones((n, K), dtype=torch.bool); tg = torch.zeros((n, K))
    for i, it in enumerate(its):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"]); att[i, :len(it["ids"])] = 1
        mpos[i] = torch.tensor(it["markers"]); tg[i] = torch.tensor(it["target"])
    return ids, att, mpos, mm, tg, torch.tensor([i["qtype"] for i in its])


dev = torch.device("cuda")
model = build_model(cfg, encoder_dir=os.path.join(mdir, "encoder"))
model.load_state_dict(load_file(os.path.join(mdir, "model.safetensors")), strict=True)
model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.head_checkpointing = True; model.to(dev).train()
train, drop = items_of(*xy("train")); print(f"{len(train)} train items ({drop} dropped), K={K}", flush=True)
assert drop <= 0.01 * (len(train) + drop), f"too many items dropped: {drop}"

MICRO, ACCUM, GROUP = 4, 8, 4
LR_ENC, LR_HEAD, S0, S1 = 2.5e-5, 1e-4, 0.4, 0.1
enc = [p for n, p in model.named_parameters() if "encoder." in n]; head = [p for n, p in model.named_parameters() if "encoder." not in n]
opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENC}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, (len(train) // (MICRO * ACCUM)) * EPOCHS), eta_min=1e-6)
@torch.no_grad()
def logits_of(texts, bs=32):
    model.eval(); out = []
    its = [build_sequence(tok, t, qdef, cfg["max_len"], cfg["head_max_len"]) for t in texts]
    for i in range(0, len(its), bs):
        chunk = [{"ids": s, "markers": m, "qtype": QTYPES["choice"], "target": [0.0] * K} for s, m in its[i:i + bs]]
        ids, att, mpos, mm, _, qt = [x.to(dev) for x in collate(chunk)]
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(ids, att, mpos, mm, qt)[0].float().cpu())
    return torch.cat(out).numpy()


Xval, Yval = xy("val"); best_acc, best_state, best_ep = -1.0, None, 0
scaler = torch.amp.GradScaler("cuda"); t0 = time.time()
for ep in range(EPOCHS):
    random.shuffle(train); sigma = S0 + (S1 - S0) * (ep / max(1, EPOCHS - 1)); opt.zero_grad(set_to_none=True)
    for step, b in enumerate(range(0, len(train), MICRO), 1):
        ids, att, mpos, mm, tg, qt = [x.to(dev) for x in collate(train[b:b + MICRO])]
        with torch.autocast("cuda", dtype=torch.float16):
            logits, act = model(ids, att, mpos, mm, qt)
        logits = logits.float()
        eps = torch.randn((GROUP,) + logits.shape, device=dev) * sigma
        eps = eps - eps.mean(-1, keepdim=True)
        z = logits.detach().unsqueeze(0) + eps
        with torch.no_grad():
            r = proper_reward(torch.softmax(z, -1), tg.unsqueeze(0), qt, mm, w_sph=0.75, w_rps=1.0)
            adv = r - r.mean(0, keepdim=True); adv = adv / (adv.std() + 1e-6)
        logp = -((z - logits.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        ce = -(tg * torch.log_softmax(logits, -1)).sum(-1).mean()
        loss = ((ce if LOSS == "ce" else -(adv * logp).mean() + ce)) / ACCUM + 0.0 * act.sum()
        scaler.scale(loss).backward()
        if step % ACCUM == 0 or b + MICRO >= len(train):
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0: print(f" ep {ep+1} step {step} loss {loss.item()*ACCUM:.3f} {time.time()-t0:.0f}s", flush=True)
    print(f"=== epoch {ep+1} done {time.time()-t0:.0f}s", flush=True)
    if SELECT:
        acc = float((logits_of(Xval).argmax(1) == np.array(Yval)).mean()); model.train()
        print(f"    val acc {acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, best_ep = acc, ep + 1
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
if SELECT and best_state is not None:
    model.load_state_dict(best_state); print(f"best epoch {best_ep} (val acc {best_acc:.4f})", flush=True)


data = {s: xy(s) for s in EVAL_SPLITS}
np.savez_compressed(os.path.join(OUT, "logits.npz"), **{s: logits_of(data[s][0]) for s in EVAL_SPLITS},
                    **{f"y_{s}": np.array(data[s][1]) for s in EVAL_SPLITS})
sample = data["val"][0][:200]; ts = []
for t in sample:
    s = time.perf_counter(); logits_of([t], bs=1); torch.cuda.synchronize(); ts.append((time.perf_counter() - s) * 1000)
meta = {"model": "laya-multilingual-322M", "seed": SEED, "epochs": EPOCHS, "loss": LOSS, "mode": MODE,
        "best_epoch": best_ep if SELECT else EPOCHS, "train_s": time.time() - t0,
        "params": sum(p.numel() for p in model.parameters()), "lat_gpu_ms": float(np.median(ts[10:]))}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1); print(meta)
