"""Additional analyses for the journal-length version of the paper, from the same runs/ as analyze.py.
Writes figures, tables and \\R{} number definitions into <out_dir> (extra_numbers.tex is \\input after numbers.tex).
Usage: python extra_analysis.py <out_dir>"""
import glob, json, os, re, sys
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, AR2EN, xy
from coarse import cat_of, CATS
from data import load, SPLITS

R = str(__import__("pathlib").Path(__file__).resolve().parents[1]); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
T = ["test_msa", "test_pal", "test_saudi", "test_moroccan", "test_tunisian"]
TN = {"test_msa": "MSA", "test_pal": "Pal.", "test_saudi": "Saudi", "test_moroccan": "Moroccan", "test_tunisian": "Tunisian", "val": "Validation"}
EN = [AR2EN[l] for l in LABELS]
M = {}
def put(k, v, fmt="{:.1f}"): M[k] = fmt.format(v); return M[k]
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def runs(fam): return [d for d in sorted(glob.glob(f"{R}/runs/{fam}_s*")) if os.path.exists(f"{d}/meta.json")]
def avg_probs(fam, split):
    ds = runs(fam); return np.mean([softmax(np.load(f"{d}/logits.npz")[split]) for d in ds], 0), np.load(f"{ds[0]}/logits.npz")[f"y_{split}"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})

# ---- dataset statistics
rows = []
for s in ["train", "val"] + T:
    X, Y = xy(s); words = [len(x.split()) for x in X]
    rows.append(f"{TN.get(s, 'Train')} & {len(X):,} & {len(set(Y))} & {np.mean(words):.1f} & {np.median(words):.0f} \\\\".replace(",", "{,}"))
    put(f"data.{s}.n", len(X), "{:,}"); put(f"data.{s}.words", np.mean(words))
open(f"{OUT}/tab_data.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Official ArBanking77 splits used in this study: number of messages, intents "
    "covered, and mean / median message length in words.}\\label{tab:data}\n\\begin{tabular}{lrrrr}\\toprule\n"
    "Split & Messages & Intents & Mean & Median \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---- per-seed accuracies (appendix)
fams = [("marbertv2", "MARBERTv2"), ("bert-base-arabertv02", "AraBERTv02"), ("bert-base-arabic-camelbert-mix", "CAMeLBERT-mix"),
        ("xlm-roberta-base", "XLM-R base"), ("laya", "Laya (RLCD)"), ("layace", "Laya (CE)"), ("layacoarse", "Laya (12-way)")]
rows = []
for f, n in fams:
    for i, d in enumerate(runs(f), 1):
        z = np.load(f"{d}/logits.npz"); meta = json.load(open(f"{d}/meta.json"))
        acc = [100 * (z[s].argmax(1) == z[f"y_{s}"]).mean() for s in T]
        rows.append(f"{n if i == 1 else ''} & {i} & " + " & ".join(f"{a:.1f}" for a in acc) + f" & {meta.get('best_epoch', '--')} \\\\")
open(f"{OUT}/tab_seeds.tex", "w").write(
    "\\begin{table*}[t]\\centering\\small\\caption{Accuracy (\\%) of every individual training run (seed) on each test set, and the "
    "epoch selected on validation. Laya (12-way) is scored on the 12 routing categories.}\\label{tab:seeds}\n"
    "\\begin{tabular}{llrrrrrr}\\toprule\nSystem & Seed & MSA & Pal. & Saudi & Moroccan & Tunisian & Epoch \\\\\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

# ---- confusions and hardest intents (MARBERTv2, seed-averaged, unseen dialects pooled)
pairs, per_int = Counter(), {}
for s in ["test_saudi", "test_moroccan", "test_tunisian"]:
    P, y = avg_probs("marbertv2", s); pred = P.argmax(1)
    for a, b in zip(y, pred):
        if a != b: pairs[(a, b)] += 1
    for c in np.unique(y):
        m = y == c; per_int.setdefault(c, [0, 0]); per_int[c][0] += int((pred[m] == c).sum()); per_int[c][1] += int(m.sum())
tot_err = sum(pairs.values())
rows = [f"{EN[a]} & {EN[b]} & {k} & {'yes' if cat_of(EN[a]) == cat_of(EN[b]) else 'no'} \\\\" for (a, b), k in pairs.most_common(12)]
put("conf.top12.share", 100 * sum(k for _, k in pairs.most_common(12)) / tot_err)
put("conf.top12.samecat", sum(cat_of(EN[a]) == cat_of(EN[b]) for (a, b), _ in pairs.most_common(12)), "{:d}")
open(f"{OUT}/tab_confusions.tex", "w").write(
    "\\begin{table*}[t]\\centering\\small\\caption{The twelve most frequent confusions of MARBERTv2 (seed-averaged) on the three unseen "
    "dialects pooled, with whether both intents fall in the same routing category.}\\label{tab:confusions}\n"
    "\\begin{tabular}{llrc}\\toprule\nGold intent & Predicted intent & Errors & Same category \\\\\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
worst = sorted(((v[0] / v[1], c, v[1]) for c, v in per_int.items() if v[1] >= 40))[:10]
rows = [f"{EN[c]} & {cat_of(EN[c]).replace('_', ' ')} & {n} & {100 * r:.1f} \\\\" for r, c, n in worst]
open(f"{OUT}/tab_hardest.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{The ten intents with the lowest recall for MARBERTv2 on the unseen dialects pooled "
    "(intents with at least 40 test messages).}\\label{tab:hardest}\n\\begin{tabular}{llrr}\\toprule\n"
    "Intent & Category & $n$ & Recall (\\%) \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---- reliability diagrams (MARBERTv2 seed 1, raw vs temperature-scaled) and confidence histograms
import torch
def fit_temp(z, y):
    zt, yt = torch.tensor(z, dtype=torch.float64), torch.tensor(y)
    lt = torch.zeros(1, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=200)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(zt / lt.exp(), yt); l.backward(); return l
    opt.step(cl); return float(lt.exp().detach())
z = np.load(f"{runs('marbertv2')[0]}/logits.npz"); Tm = fit_temp(z["val"], z["y_val"]); put("marbertv2.temp.s1", Tm, "{:.2f}")
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
bins = np.linspace(0, 1, 11)
for ax, s in zip(axes, ["test_msa", "test_saudi", "test_tunisian"]):
    for zz, lab, mk in ((z[s], "raw", "o"), (z[s] / Tm, f"T={Tm:.2f}", "s")):
        p = softmax(zz); c, ok = p.max(1), p.argmax(1) == z[f"y_{s}"]; xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (c > lo) & (c <= hi)
            if m.sum() >= 15: xs.append(c[m].mean()); ys.append(ok[m].mean())
        ax.plot(xs, ys, marker=mk, ms=3, lw=1, label=lab)
    ax.plot([0, 1], [0, 1], color="gray", lw=0.6, ls=":"); ax.set_title(TN[s], fontsize=8); ax.set_xlabel("Confidence"); ax.grid(alpha=.3, lw=.4)
axes[0].set_ylabel("Accuracy"); axes[0].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_reliability.pdf")
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for s in T:
    P, y = avg_probs("marbertv2", s); ax.hist(P.max(1), bins=np.linspace(0, 1, 26), histtype="step", density=True, lw=1, label=TN[s])
    put(f"marbertv2.{s}.lowconf", 100 * (P.max(1) < 0.6).mean())
ax.set_xlabel("MARBERTv2 confidence (seed-averaged)"); ax.set_ylabel("Density"); ax.legend(fontsize=6, frameon=False, loc="upper left")
ax.grid(alpha=.3, lw=.4); fig.tight_layout(); fig.savefig(f"{OUT}/fig_confhist.pdf")

# ---- risk-coverage curves
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for s in T:
    P, y = avg_probs("marbertv2", s); o = np.argsort(-P.max(1)); ok = (P.argmax(1) == y)[o]
    cov = np.arange(1, len(ok) + 1) / len(ok); risk = 1 - np.cumsum(ok) / np.arange(1, len(ok) + 1)
    ax.plot(100 * cov, 100 * risk, lw=1, label=TN[s])
ax.axhline(10, color="gray", lw=0.6, ls=":"); ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Error rate on answered (%)")
ax.legend(fontsize=6, frameon=False); ax.grid(alpha=.3, lw=.4); fig.tight_layout(); fig.savefig(f"{OUT}/fig_riskcov.pdf")

# ---- LLM: zero-shot vs retrieval, invalid answers, complementarity with the encoder
L = json.load(open(f"{R}/runs/llm_qwen/llm.json"))["rows"]
rows, crow = [], []
for s in T:
    rr = L[s]; y = np.array([r["y"] for r in rr]); z0 = np.array([r["zero"] for r in rr]); f = np.array([r["few"] for r in rr])
    lat0, lat1 = np.median([r["lat_zero"] for r in rr]), np.median([r["lat_few"] for r in rr])
    rows.append(f"{TN[s]} & {100*(z0==y).mean():.1f} & {100*(z0<0).mean():.1f} & {lat0:.0f} & {100*(f==y).mean():.1f} & {100*(f<0).mean():.1f} & {lat1:.0f} \\\\")
    idx = np.array([r["i"] for r in rr]); P, yy = avg_probs("marbertv2", s); b = P.argmax(1)[idx] == y; l = f == y
    both, ob, ol, none = b & l, b & ~l, ~b & l, ~b & ~l
    crow.append(f"{TN[s]} & {100*both.mean():.1f} & {100*ob.mean():.1f} & {100*ol.mean():.1f} & {100*none.mean():.1f} & {100*(b|l).mean():.1f} \\\\")
    put(f"comp.{s}.oracle", 100 * (b | l).mean()); put(f"comp.{s}.onlyllm", 100 * ol.mean()); put(f"comp.{s}.onlybert", 100 * ob.mean())
open(f"{OUT}/tab_llm.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Qwen3.5-9B on the 400-message subsamples: accuracy (\\%), share of answers that are not "
    "a valid intent number (\\%), and median latency (ms), without and with 10 retrieved examples.}\\label{tab:llm}\n"
    "\\begin{tabular}{lrrrrrr}\\toprule\n& \\multicolumn{3}{c}{Zero-shot} & \\multicolumn{3}{c}{10 retrieved}\\\\\n"
    "Test & Acc. & Invalid & ms & Acc. & Invalid & ms \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
open(f"{OUT}/tab_complement.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Complementarity of MARBERTv2 (seed-averaged) and the retrieval-augmented LLM on the same "
    "400 messages per test set (\\% of messages). The last column is the accuracy of an oracle that always picks the correct "
    "system when one of them is right.}\\label{tab:complement}\n\\begin{tabular}{lrrrrr}\\toprule\n"
    "Test & Both & Encoder only & LLM only & Neither & Oracle \\\\\\midrule\n" + "\n".join(crow) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---- threshold sensitivity and cost model
C = json.load(open(f"{R}/results/cascade.json")); N = json.load(open(f"{R}/results/numbers.json"))
enc_ms, llm_ms = N["marbertv2.lat_gpu_ms"], np.median([r["lat_few"] for s in T for r in L[s]])
rows = []
for tau in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    cells = []
    for s in ["test_saudi", "test_moroccan", "test_tunisian"]:
        t, acc, share = min(C["curve"][s], key=lambda r: abs(r[0] - tau)); cells += [f"{acc:.1f}", f"{share:.1f}"]
    rows.append(f"{tau:.1f}{'$^*$' if abs(tau - C['tau']) < 1e-9 else ''} & " + " & ".join(cells) + " \\\\")
open(f"{OUT}/tab_tau.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Sensitivity of the cascade to the threshold $\\tau$: accuracy (\\%) and share routed to "
    "the LLM (\\%) on the 400-message subsamples. $^*$ marks the value chosen on validation.}\\label{tab:tau}\n"
    "\\begin{tabular}{lrrrrrr}\\toprule\n& \\multicolumn{2}{c}{Saudi} & \\multicolumn{2}{c}{Moroccan} & \\multicolumn{2}{c}{Tunisian}\\\\\n"
    "$\\tau$ & Acc. & LLM & Acc. & LLM & Acc. & LLM \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
rows = []
for s in T:
    sh = C["tests"][s]["to_llm"] / 100; exp_ms = enc_ms + sh * llm_ms
    rows.append(f"{TN[s]} & {enc_ms:.1f} & {llm_ms:.0f} & {exp_ms:.0f} & {100 * exp_ms / llm_ms:.0f} \\\\")
    put(f"cost.{s}.ms", exp_ms, "{:.0f}"); put(f"cost.{s}.pct", 100 * exp_ms / llm_ms, "{:.0f}")
put("cost.llm_ms", llm_ms, "{:.0f}")
open(f"{OUT}/tab_cost.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Expected GPU latency per message (ms): encoder only, retrieval-augmented LLM only, and the "
    "cascade (encoder for every message plus the LLM for the routed share), and the cascade's latency as a percentage of the "
    "LLM-only system.}\\label{tab:cost}\n\\begin{tabular}{lrrrr}\\toprule\nTest & Encoder & LLM & Cascade & \\% of LLM \\\\\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---- 77 -> 12 mapping (appendix)
by = {}
for e in EN: by.setdefault(cat_of(e), []).append(e)
rows = [f"{c.replace('_', ' ')} & {len(v)} & {', '.join(sorted(v))} \\\\" for c, v in by.items()]
open(f"{OUT}/tab_mapping.tex", "w").write(
    "\\begin{table*}[t]\\centering\\footnotesize\\caption{Mapping of the 77 ArBanking77 intents (English names) to the 12 contact-centre "
    "routing categories used in Section~\\ref{sec:routing}.}\\label{tab:mapping}\n"
    "\\begin{tabular}{lrp{0.72\\textwidth}}\\toprule\nCategory & $n$ & Intents \\\\\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

# ---- error examples (Arabic, rendered with XeLaTeX as a standalone figure)
ex = []
for s, k in (("test_saudi", 3), ("test_moroccan", 2), ("test_tunisian", 2)):
    X = [t for t, _ in load(SPLITS[s])] if False else xy(s)[0]
    P, y = avg_probs("marbertv2", s); pred, conf = P.argmax(1), P.max(1)
    cand = [i for i in np.argsort(-conf) if pred[i] != y[i] and len(X[i].split()) <= 12][:k]
    ex += [(TN[s], X[i], EN[y[i]], EN[pred[i]], conf[i]) for i in cand]
esc = lambda t: t.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
body = "\n".join(f"{d} & \\textarabic{{{esc(t)}}} & {esc(g)} & {esc(p)} & {c:.2f}\\\\" for d, t, g, p, c in ex)
open(f"{OUT}/examples_src.tex", "w").write(
    "\\documentclass[border=2pt]{standalone}\n\\usepackage{fontspec}\n\\usepackage{polyglossia}\n\\setmainlanguage{english}\n"
    "\\setotherlanguage{arabic}\n\\newfontfamily\\arabicfont[Script=Arabic,Scale=1.05]{Amiri}\n\\usepackage{booktabs}\n"
    "\\begin{document}\\small\n\\begin{tabular}{@{}l r l l r@{}}\\toprule\nDialect & Message & Gold intent & Predicted & Conf.\\\\\\midrule\n"
    + body + "\n\\bottomrule\\end{tabular}\n\\end{document}\n")

lines = ["\\makeatletter"] + [f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(M.items())] + ["\\makeatother"]
open(f"{OUT}/extra_numbers.tex", "w").write("\n".join(lines) + "\n")
nj = json.load(open(f"{OUT}/numbers.json")) if os.path.exists(f"{OUT}/numbers.json") else {}
nj.update(M); json.dump(nj, open(f"{OUT}/numbers.json", "w"), indent=1, sort_keys=True)
print(len(M), "extra numbers;", sorted(f for f in os.listdir(OUT) if f.startswith(("tab_", "fig_", "examples"))))
