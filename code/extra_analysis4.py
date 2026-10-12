"""Fourth set: pairwise significance vs MARBERTv2, per-intent F1 comparison, Tunisian category confusion matrix.
Appends \\R{} numbers to <out_dir>/extra_numbers.tex. Usage: python extra_analysis4.py <out_dir>"""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, AR2EN
from coarse import cat_of, CATS

R = str(__import__("pathlib").Path(__file__).resolve().parents[1]); OUT = sys.argv[1]
T = ["test_msa", "test_pal", "test_saudi", "test_moroccan", "test_tunisian"]
TN = {"test_msa": "MSA", "test_pal": "Pal.", "test_saudi": "Saudi", "test_moroccan": "Moroccan", "test_tunisian": "Tunisian"}
M = {}
def put(k, v, fmt="{:.1f}"): M[k] = fmt.format(v); return M[k]
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def runs(fam): return [d for d in sorted(glob.glob(f"{R}/runs/{fam}_s*")) if os.path.exists(f"{d}/meta.json")] if fam != "knn" else [f"{R}/runs/knn"]
def pred(fam, split):
    ds = runs(fam); P = np.mean([softmax(np.load(f"{d}/logits.npz")[split]) for d in ds], 0)
    return P.argmax(1), np.load(f"{ds[0]}/logits.npz")[f"y_{split}"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
sysn = [("bert-base-arabertv02", "AraBERTv02"), ("bert-base-arabic-camelbert-mix", "CAMeLBERT-mix"), ("xlm-roberta-base", "XLM-R base"),
        ("knn", "bge-m3 $k$NN"), ("laya", "Laya (RLCD)"), ("layace", "Laya (CE)")]

# ---- paired bootstrap of every system against MARBERTv2 (seed-averaged predictions, full test sets)
rng = np.random.default_rng(0); rows = []; nsig = 0
ref = {s: pred("marbertv2", s) for s in T}
for f, n in sysn:
    cells = []
    for s in T:
        pb, y = ref[s]; pa, _ = pred(f, s); d = (pa == y).astype(float) - (pb == y)
        B = d[rng.integers(0, len(d), (10000, len(d)))].mean(1) * 100; lo, hi = np.percentile(B, [2.5, 97.5])
        sig = lo > 0 or hi < 0; nsig += sig
        cells.append(f"{100 * d.mean():+.1f}{'$^{\\dagger}$' if sig else ''}")
    rows.append(f"{n} & " + " & ".join(cells) + " \\\\")
put("sig.n", nsig, "{:d}"); put("sig.total", len(sysn) * len(T), "{:d}")
open(f"{OUT}/tab_sig.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Accuracy difference (percentage points) of each system against MARBERTv2 on the full "
    "test sets, both seed-averaged. $^{\\dagger}$: the 95\\% paired bootstrap interval excludes zero.}\\label{tab:sig}\n"
    "\\begin{tabular}{lrrrrr}\\toprule\nSystem & MSA & Pal. & Saudi & Moroccan & Tunisian \\\\\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---- per-intent F1 on the unseen dialects pooled: MARBERTv2 vs Laya (CE) vs kNN
ys, pm, pl, pk = [], [], [], []
for s in ["test_saudi", "test_moroccan", "test_tunisian"]:
    a, y = pred("marbertv2", s); b, _ = pred("layace", s); c, _ = pred("knn", s); ys.append(y); pm.append(a); pl.append(b); pk.append(c)
y, pm, pl, pk = map(np.concatenate, (ys, pm, pl, pk)); lab = np.unique(y)
fm, fl, fk = (f1_score(y, p, labels=lab, average=None) * 100 for p in (pm, pl, pk))
o = np.argsort(fm)
fig, ax = plt.subplots(figsize=(7.0, 2.4))
ax.plot(range(len(lab)), fm[o], lw=1.2, label="MARBERTv2"); ax.plot(range(len(lab)), fl[o], lw=0.9, label="Laya (CE)")
ax.plot(range(len(lab)), fk[o], lw=0.9, label="bge-m3 kNN")
ax.set_xlabel("Intents, sorted by MARBERTv2 F1"); ax.set_ylabel("F1 (%)"); ax.legend(fontsize=6, frameon=False); ax.grid(alpha=.3, lw=.4)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_intentf1.pdf")
put("intf1.mar_beats_laya", (fm > fl).mean() * 100, "{:.0f}"); put("intf1.mar_beats_knn", (fm > fk).mean() * 100, "{:.0f}")
put("intf1.below50", int((fm < 50).sum()), "{:d}")

# ---- Tunisian category confusion matrix (MARBERTv2, summed probabilities)
cats = list(CATS); C = np.array([cats.index(cat_of(AR2EN[l])) for l in LABELS])
ds = runs("marbertv2"); P = np.mean([softmax(np.load(f"{d}/logits.npz")["test_tunisian"]) for d in ds], 0)
yt = C[np.load(f"{ds[0]}/logits.npz")["y_test_tunisian"]]; Pc = np.stack([P[:, C == k].sum(1) for k in range(len(cats))], 1); pc = Pc.argmax(1)
used = [k for k in range(len(cats)) if (yt == k).sum() > 0]
Mx = np.array([[((yt == a) & (pc == b)).sum() for b in range(len(cats))] for a in used])
Mn = Mx / Mx.sum(1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(4.6, 3.0))
im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_yticks(range(len(used))); ax.set_yticklabels([cats[k].replace("_", " ") for k in used], fontsize=7)
ax.set_xticks(range(len(cats))); ax.set_xticklabels([c.replace("_", " ") for c in cats], rotation=45, ha="right", fontsize=7)
for i in range(len(used)):
    for j in range(len(cats)):
        if Mn[i, j] >= 5: ax.text(j, i, f"{Mn[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if Mn[i, j] > 60 else "black")
ax.set_ylabel("Gold category"); ax.set_xlabel("Predicted category"); fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="% of gold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_tuncm.pdf")

with open(f"{OUT}/extra_numbers.tex") as f: old = f.read().replace("\\makeatother\n", "")
open(f"{OUT}/extra_numbers.tex", "w").write(old + "\n".join(f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(M.items())) + "\n\\makeatother\n")
nj = json.load(open(f"{OUT}/numbers.json")); nj.update(M); json.dump(nj, open(f"{OUT}/numbers.json", "w"), indent=1, sort_keys=True)
print(M)
