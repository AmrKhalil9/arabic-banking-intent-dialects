"""All numbers for the paper, from runs/*/logits.npz (+ llm.json). Writes results/metrics.csv, results/numbers.json.
Models are grouped by family (seeds pooled as mean +- std). Calibration: one temperature per run, fitted on the
validation split only (NLL, LBFGS), applied unchanged to every test split.
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np, torch
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from common import LABELS, AR2EN
from coarse import cat_of, CATS

R = str(__import__("pathlib").Path(__file__).resolve().parents[1]); OUT = f"{R}/results"; os.makedirs(OUT, exist_ok=True)
TESTS = ["test_msa", "test_pal", "test_saudi", "test_moroccan", "test_tunisian"]
COARSE = np.array([list(CATS).index(cat_of(AR2EN[l])) for l in LABELS])     # 77 -> 12


def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_temp(z, y):
    zt, yt = torch.tensor(z, dtype=torch.float64), torch.tensor(y)
    lt = torch.zeros(1, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=200)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(zt / lt.exp(), yt); l.backward(); return l
    opt.step(cl); return float(lt.exp().detach())


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def coverage_at(p, y, target):
    """Largest share of traffic answered automatically (highest confidence first) with accuracy >= target."""
    order = np.argsort(-p.max(1)); ok = (p.argmax(1) == y)[order]
    acc = np.cumsum(ok) / np.arange(1, len(ok) + 1); good = np.where(acc >= target)[0]
    return float((good.max() + 1) / len(ok)) if len(good) else 0.0


def aurc(p, y):
    order = np.argsort(-p.max(1)); err = (p.argmax(1) != y)[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def family(run): return re.sub(r"_s\d+$", "", run)


rows, probs = [], defaultdict(dict)
for f in sorted(glob.glob(f"{R}/runs/*/logits.npz")):
    if not os.path.exists(os.path.join(os.path.dirname(f), "meta.json")):
        continue                                   # still training: best-epoch logits are not final yet
    run = os.path.basename(os.path.dirname(f)); d = np.load(f)
    T = fit_temp(d["val"], d["y_val"])
    for s in TESTS:
        z, y = d[s], d[f"y_{s}"]; p_raw, p = softmax(z), softmax(z / T); pred = p.argmax(1)
        if z.shape[1] == len(CATS):            # a model trained directly on the 12 routing categories
            rows.append({"run": run, "family": family(run), "split": s, "n": len(y), "temp": T,
                         "acc": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y))),
                         "ece_raw": ece(p_raw, y), "ece": ece(p, y), "aurc": aurc(p, y), "cov90": coverage_at(p, y, .90),
                         "cov95": coverage_at(p, y, .95), "coarse_acc": float((pred == y).mean()), "err_within_coarse": float("nan")})
            continue
        pc = np.zeros((len(p), len(CATS))); np.add.at(pc.T, COARSE, p.T)
        rows.append({"run": run, "family": family(run), "split": s, "n": len(y), "temp": T,
                     "acc": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y))),
                     "ece_raw": ece(p_raw, y), "ece": ece(p, y), "aurc": aurc(p, y),
                     "cov90": coverage_at(p, y, .90), "cov95": coverage_at(p, y, .95),
                     "coarse_acc": float((pc.argmax(1) == COARSE[y]).mean()),
                     "err_within_coarse": float(np.mean(COARSE[pred[pred != y]] == COARSE[y[pred != y]])) if (pred != y).any() else 0.0})
        probs[family(run)].setdefault(s, []).append(p)
import csv
with open(f"{OUT}/metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# family summary: mean / std over seeds
fam = defaultdict(lambda: defaultdict(list))
for r in rows:
    for k in ("acc", "macro_f1", "ece_raw", "ece", "aurc", "cov90", "cov95", "coarse_acc", "err_within_coarse"):
        fam[(r["family"], r["split"])][k].append(r[k])
N = {}
for (fa, s), d in fam.items():
    for k, v in d.items():
        N[f"{fa}.{s}.{k}"] = round(100 * float(np.mean(v)), 1); N[f"{fa}.{s}.{k}.sd"] = round(100 * float(np.std(v)), 1)
    N[f"{fa}.{s}.seeds"] = len(d["acc"])

# paired bootstrap: seed-averaged predictions of each family vs the best encoder baseline, per split
rng = np.random.default_rng(0)
ys = {s: np.load(glob.glob(f"{R}/runs/*/logits.npz")[0])[f"y_{s}"] for s in TESTS}
enc_fams = [f for f in probs if not f.startswith("laya") and f != "knn"]
for s in TESTS:
    best = max(enc_fams, key=lambda f: N[f"{f}.{s}.macro_f1"]); N[f"best_encoder.{s}"] = best
    if "laya" not in probs: continue
    a = (np.mean(probs["laya"][s], 0).argmax(1) == ys[s]); b = (np.mean(probs[best][s], 0).argmax(1) == ys[s])
    idx = rng.integers(0, len(a), (5000, len(a))); diff = (a[idx].mean(1) - b[idx].mean(1)) * 100
    N[f"laya_vs_best.{s}.d"] = round(float((a.mean() - b.mean()) * 100), 1)
    N[f"laya_vs_best.{s}.lo"], N[f"laya_vs_best.{s}.hi"] = [round(float(x), 1) for x in np.percentile(diff, [2.5, 97.5])]

# LLM baseline (subsample) + the trained models on the same items
lf = f"{R}/runs/llm_qwen/llm.json"
if os.path.exists(lf):
    L = json.load(open(lf))
    for s, rs in L["rows"].items():
        idx = np.array([r["i"] for r in rs]); y = np.array([r["y"] for r in rs])
        N[f"llm.{s}.n"] = len(rs)
        for mode in ("zero", "few"):
            N[f"llm_{mode}.{s}.acc"] = round(100 * float(np.mean([r[mode] == r["y"] for r in rs])), 1)
            N[f"llm_{mode}.{s}.lat_ms"] = round(float(np.median([r[f"lat_{mode}"] for r in rs])), 0)
        for fa in probs:
            N[f"{fa}.{s}.acc_on_llm_subset"] = round(100 * float((np.mean(probs[fa][s], 0)[idx].argmax(1) == y).mean()), 1)

for m in glob.glob(f"{R}/runs/*/meta.json"):
    meta = json.load(open(m)); fa = family(os.path.basename(os.path.dirname(m)))
    for k in ("lat_gpu_ms", "lat_cpu8_ms", "params", "train_s"):
        if k in meta: N.setdefault(f"{fa}.{k}", []).append(meta[k])
for k in [k for k, v in N.items() if isinstance(v, list)]:
    N[k] = round(float(np.mean(N[k])), 1)
if os.path.exists(f"{R}/runs/leak_check.json"):
    for s, d in json.load(open(f"{R}/runs/leak_check.json")).items():
        for k, v in d.items(): N[f"leak.{s}.{k}"] = v
json.dump(N, open(f"{OUT}/numbers.json", "w"), indent=1, sort_keys=True)
fams = sorted({r["family"] for r in rows})
print(f"{'family':38s}" + "".join(f"{s[5:]:>10s}" for s in TESTS))
for fa in fams:
    print(f"{fa:38s}" + "".join(f"{N.get(f'{fa}.{s}.macro_f1', float('nan')):10.1f}" for s in TESTS))
