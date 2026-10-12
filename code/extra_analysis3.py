"""Third set of journal-length analyses: efficiency, training curves, cascade accuracy decomposition, system agreement,
cascade rescue/harm examples. Appends \\R{} numbers to <out_dir>/extra_numbers.tex. Usage: python extra_analysis3.py <out_dir>"""
import glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, AR2EN, xy

R = str(__import__("pathlib").Path(__file__).resolve().parents[1]); OUT = sys.argv[1]
T = ["test_msa", "test_pal", "test_saudi", "test_moroccan", "test_tunisian"]
TN = {"test_msa": "MSA", "test_pal": "Pal.", "test_saudi": "Saudi", "test_moroccan": "Moroccan", "test_tunisian": "Tunisian"}
EN = [AR2EN[l] for l in LABELS]; M = {}
def put(k, v, fmt="{:.1f}"): M[k] = fmt.format(v); return M[k]
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def runs(fam): return [d for d in sorted(glob.glob(f"{R}/runs/{fam}_s*")) if os.path.exists(f"{d}/meta.json")]
def avg_probs(fam, split):
    ds = runs(fam) if fam != "knn" else [f"{R}/runs/knn"]
    return np.mean([softmax(np.load(f"{d}/logits.npz")[split]) for d in ds], 0), np.load(f"{ds[0]}/logits.npz")[f"y_{split}"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
L = json.load(open(f"{R}/runs/llm_qwen/llm.json"))["rows"]; C = json.load(open(f"{R}/results/cascade.json")); tau = C["tau"]

# ---- efficiency
fams = [("marbertv2", "MARBERTv2"), ("bert-base-arabertv02", "AraBERTv02"), ("bert-base-arabic-camelbert-mix", "CAMeLBERT-mix"),
        ("xlm-roberta-base", "XLM-R base"), ("laya", "Laya (RLCD)"), ("layace", "Laya (CE)"), ("layacoarse", "Laya (12-way)")]
rows = []
for f, n in fams:
    ms = [json.load(open(f"{d}/meta.json")) for d in runs(f)]
    gpu = np.mean([m["lat_gpu_ms"] for m in ms]); cpu = [m.get("lat_cpu8_ms") for m in ms]
    cpu_s = f"{np.mean(cpu):.1f}" if all(c is not None for c in cpu) else "--"
    rows.append(f"{n} & {ms[0]['params'] / 1e6:.0f} & {np.mean([m['train_s'] for m in ms]) / 60:.0f} & {gpu:.1f} & {cpu_s} \\\\")
    put(f"eff.{f}.trainmin", np.mean([m["train_s"] for m in ms]) / 60, "{:.0f}")
llm_ms = np.median([r["lat_few"] for s in T for r in L[s]]); llm0_ms = np.median([r["lat_zero"] for s in T for r in L[s]])
rows.append(f"Qwen3.5-9B, zero-shot & $\\approx$9{{,}}000 & -- & {llm0_ms:.0f} & -- \\\\")
rows.append(f"Qwen3.5-9B, 10 retrieved & $\\approx$9{{,}}000 & -- & {llm_ms:.0f} & -- \\\\")
open(f"{OUT}/tab_efficiency.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Model size (millions of parameters), mean training time per run (minutes, one GPU), "
    "and median latency for one message (ms) on the GPU and, for the encoders, on 8 CPU threads.}\\label{tab:efficiency}\n"
    "\\begin{tabular}{lrrrr}\\toprule\nSystem & Params & Train & GPU & CPU \\\\\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---- training curves from the logs
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
for f, n in fams[:4]:
    cur = []
    for d in runs(f):
        v = [float(x) for x in re.findall(r"^ep \d+ val macroF1 ([0-9.]+)", open(f"{R}/logs/{os.path.basename(d)}.log").read(), re.M)]
        if v: cur.append(v)
    k = min(map(len, cur)); a = 100 * np.array([c[:k] for c in cur])
    axes[0].errorbar(range(1, k + 1), a.mean(0), yerr=a.std(0), marker="o", ms=3, lw=1, capsize=2, label=n)
for f, n in fams[4:]:
    cur = []
    for d in runs(f):
        log = open(f"{R}/logs/{os.path.basename(d)}.log").read()
        v = [float(x) for x in re.findall(r"^\s+val acc ([0-9.]+)", log, re.M)]   # per-epoch lines, not the final summary
        if v: cur.append(v)
    if not cur: continue
    k = min(map(len, cur)); a = 100 * np.array([c[:k] for c in cur])
    axes[1].errorbar(range(1, k + 1), a.mean(0), yerr=a.std(0), marker="s", ms=3, lw=1, capsize=2, label=n)
axes[0].set_ylabel("Validation macro-F1 (%)"); axes[1].set_ylabel("Validation accuracy (%)")
for ax in axes: ax.set_xlabel("Epoch"); ax.grid(alpha=.3, lw=.4); ax.legend(fontsize=6, frameon=False, loc="lower right")
axes[0].set_title("Encoders", fontsize=8); axes[1].set_title("Decision model (runs with validation selection)", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_epochs.pdf")

# ---- cascade decomposition
rows = []
for s in T:
    idx = np.array([r["i"] for r in L[s]]); y = np.array([r["y"] for r in L[s]]); llm = np.array([r["few"] for r in L[s]])
    P, _ = avg_probs("marbertv2", s); c, p = P.max(1)[idx], P.argmax(1)[idx]; hi = c >= tau
    a_hi = 100 * (p[hi] == y[hi]).mean(); e_lo = 100 * (p[~hi] == y[~hi]).mean() if (~hi).any() else float("nan")
    l_lo = 100 * (llm[~hi] == y[~hi]).mean() if (~hi).any() else float("nan")
    rows.append(f"{TN[s]} & {100 * hi.mean():.1f} & {a_hi:.1f} & {100 * (~hi).mean():.1f} & {e_lo:.1f} & {l_lo:.1f} \\\\")
    put(f"dec.{s}.acc_hi", a_hi); put(f"dec.{s}.enc_lo", e_lo); put(f"dec.{s}.llm_lo", l_lo)
open(f"{OUT}/tab_decomp.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Decomposition of the cascade (Eq.~\\eqref{eq:decomp}) on the 400-message subsamples: "
    "share of messages kept by the encoder and its accuracy on them, share routed to the LLM, and accuracy of the encoder and "
    "of the LLM on the routed messages (\\%).}\\label{tab:decomp}\n\\begin{tabular}{lrrrrr}\\toprule\n"
    "& \\multicolumn{2}{c}{Kept ($\\bar{c}\\ge\\tau$)} & \\multicolumn{3}{c}{Routed ($\\bar{c}<\\tau$)}\\\\\n"
    "Test & Share & Encoder & Share & Encoder & LLM \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---- agreement between systems (Cohen's kappa of seed-averaged predictions)
sysn = [("marbertv2", "MARBERTv2"), ("bert-base-arabertv02", "AraBERTv02"), ("bert-base-arabic-camelbert-mix", "CAMeLBERT"),
        ("xlm-roberta-base", "XLM-R"), ("knn", "kNN"), ("layace", "Laya (CE)")]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"wspace": 0.12})
for ax, s in zip(axes, ["test_saudi", "test_tunisian"]):
    preds = [avg_probs(f, s)[0].argmax(1) for f, _ in sysn]; K = np.array([[cohen_kappa_score(a, b) for b in preds] for a in preds])
    im = ax.imshow(K, vmin=0.4, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(sysn))); ax.set_xticklabels([n for _, n in sysn], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(sysn))); ax.set_yticklabels([n for _, n in sysn] if s == "test_saudi" else [], fontsize=7); ax.set_title(TN[s], fontsize=8)
    for i in range(len(sysn)):
        for j in range(len(sysn)): ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if K[i, j] < 0.75 else "black")
    put(f"kappa.{s}.mar_ara", K[0, 1], "{:.2f}"); put(f"kappa.{s}.mar_knn", K[0, 4], "{:.2f}")
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Cohen's $\\kappa$"); fig.savefig(f"{OUT}/fig_kappa.pdf", bbox_inches="tight")

# ---- rescue and harm examples (Arabic, XeLaTeX standalone)
esc = lambda t: t.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
rescue, harm = [], []
for s in ["test_moroccan", "test_tunisian", "test_saudi"]:
    X = xy(s)[0]; idx = np.array([r["i"] for r in L[s]]); y = np.array([r["y"] for r in L[s]]); llm = np.array([r["few"] for r in L[s]])
    P, _ = avg_probs("marbertv2", s); c, p = P.max(1)[idx], P.argmax(1)[idx]
    for j in np.argsort(c):
        if c[j] < tau and p[j] != y[j] and llm[j] == y[j] and len(X[idx[j]].split()) <= 12 and sum(r[0] == TN[s] for r in rescue) < 2:
            rescue.append((TN[s], X[idx[j]], EN[y[j]], EN[p[j]], c[j]))
        if c[j] < tau and p[j] == y[j] and llm[j] != y[j] and llm[j] >= 0 and len(X[idx[j]].split()) <= 12 and sum(r[0] == TN[s] for r in harm) < 1:
            harm.append((TN[s], X[idx[j]], EN[y[j]], EN[llm[j]], c[j]))
row = lambda d, t, g, w, cc: f"{d} & \\textarabic{{{esc(t)}}} & {esc(g)} & {esc(w)} & {cc:.2f}\\\\"
body = ("\\multicolumn{5}{@{}l}{\\textit{Rescued: encoder unsure and wrong, LLM correct}}\\\\\n"
        + "\n".join(row(*r) for r in rescue) + "\n\\midrule\n\\multicolumn{5}{@{}l}{\\textit{Harmed: encoder unsure but correct, LLM wrong}}\\\\\n"
        + "\n".join(row(*r) for r in harm))
open(f"{OUT}/rescue_src.tex", "w").write(
    "\\documentclass[border=2pt]{standalone}\n\\usepackage{fontspec}\n\\usepackage{polyglossia}\n\\setmainlanguage{english}\n"
    "\\setotherlanguage{arabic}\n\\newfontfamily\\arabicfont[Script=Arabic,Scale=1.05]{Amiri}\n\\usepackage{booktabs}\n"
    "\\begin{document}\\small\n\\begin{tabular}{@{}l r l l r@{}}\\toprule\nDialect & Message & Gold intent & Wrong answer & Enc. conf.\\\\\\midrule\n"
    + body + "\n\\bottomrule\\end{tabular}\n\\end{document}\n")

with open(f"{OUT}/extra_numbers.tex") as f: old = f.read().replace("\\makeatother\n", "")
open(f"{OUT}/extra_numbers.tex", "w").write(old + "\n".join(f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(M.items())) + "\n\\makeatother\n")
nj = json.load(open(f"{OUT}/numbers.json")); nj.update(M); json.dump(nj, open(f"{OUT}/numbers.json", "w"), indent=1, sort_keys=True)
print(len(M), "numbers"); [print(k, v) for k, v in sorted(M.items())]
