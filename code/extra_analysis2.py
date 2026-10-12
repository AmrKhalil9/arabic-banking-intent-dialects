"""Second set of journal-length analyses: lexical shift, message length, cascade component robustness, category heatmap,
selective-prediction table. Appends \\R{} numbers to <out_dir>/extra_numbers.tex. Usage: python extra_analysis2.py <out_dir>"""
import glob, json, os, re, sys
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, AR2EN, xy
from coarse import cat_of, CATS

R = str(__import__("pathlib").Path(__file__).resolve().parents[1]); OUT = sys.argv[1]
T = ["test_msa", "test_pal", "test_saudi", "test_moroccan", "test_tunisian"]
TN = {"test_msa": "MSA", "test_pal": "Pal.", "test_saudi": "Saudi", "test_moroccan": "Moroccan", "test_tunisian": "Tunisian"}
M = {}
def put(k, v, fmt="{:.1f}"): M[k] = fmt.format(v); return M[k]
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def runs(fam): return [d for d in sorted(glob.glob(f"{R}/runs/{fam}_s*")) if os.path.exists(f"{d}/meta.json")]
def avg_probs(fams, split):
    ps = [softmax(np.load(f"{d}/logits.npz")[split]) for f in fams for d in (runs(f) if f != "knn" else [f"{R}/runs/knn"])]
    d0 = runs(fams[0])[0] if fams[0] != "knn" else f"{R}/runs/knn"
    return np.mean(ps, 0), np.load(f"{d0}/logits.npz")[f"y_{split}"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
norm = lambda w: re.sub(r"[\u064B-\u0652\u0640]", "", w).replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
tok = lambda t: [norm(w) for w in re.findall(r"[\u0600-\u06FF]+", t)]

# ---- lexical shift
Xtr, _ = xy("train"); vocab = Counter(w for t in Xtr for w in tok(t))
rows, oov_rates, accs = [], [], []
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for s in T:
    X, _ = xy(s); P, y = avg_probs(["marbertv2"], s); ok = P.argmax(1) == y
    msg_oov = np.array([np.mean([vocab[w] == 0 for w in tok(t)]) if tok(t) else 0.0 for t in X])
    tok_oov = np.mean([vocab[w] == 0 for t in X for w in tok(t)])
    rows.append(f"{TN[s]} & {100*tok_oov:.1f} & {100*np.mean(msg_oov > 0):.1f} & {100*ok.mean():.1f} \\\\")
    put(f"lex.{s}.tokoov", 100 * tok_oov); put(f"lex.{s}.msgoov", 100 * np.mean(msg_oov > 0))
    oov_rates.append(tok_oov); accs.append(ok.mean())
    bins = [(0, 0), (1e-9, 0.2), (0.2, 0.4), (0.4, 1.01)]; xs, ys = [], []
    for i, (lo, hi) in enumerate(bins):
        m = (msg_oov == 0) if hi == 0 else (msg_oov >= lo) & (msg_oov < hi)
        if m.sum() >= 30: xs.append(i); ys.append(100 * ok[m].mean())
    ax.plot(xs, ys, marker="o", ms=3, lw=1, label=TN[s])
    m0, m1 = msg_oov == 0, msg_oov >= 0.4
    if m0.sum() >= 30: put(f"lex.{s}.acc_oov0", 100 * ok[m0].mean())
    if m1.sum() >= 30: put(f"lex.{s}.acc_oov40", 100 * ok[m1].mean())
ax.set_xticks(range(4)); ax.set_xticklabels(["0%", "0-20%", "20-40%", ">=40%"]); ax.set_xlabel("Share of message words unseen in training")
ax.set_ylabel("MARBERTv2 accuracy (%)"); ax.legend(fontsize=6, frameon=False); ax.grid(alpha=.3, lw=.4)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_oov.pdf")
put("lex.corr", float(np.corrcoef(oov_rates, accs)[0, 1]), "{:.2f}")
open(f"{OUT}/tab_lex.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Lexical shift from the training data: share of Arabic word tokens never seen in the "
    "training split (after removing diacritics and normalising alif, ta marbuta and alif maqsura), share of messages with at "
    "least one such word, and accuracy (\\%) of the seed-averaged MARBERTv2.}\\label{tab:lex}\n\\begin{tabular}{lrrr}\\toprule\n"
    "Test & Unseen tokens & Messages affected & Accuracy \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---- message length
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for s in T:
    X, _ = xy(s); P, y = avg_probs(["marbertv2"], s); ok = P.argmax(1) == y; L = np.array([len(t.split()) for t in X])
    edges = [(1, 4), (5, 7), (8, 10), (11, 15), (16, 200)]; xs, ys = [], []
    for i, (lo, hi) in enumerate(edges):
        m = (L >= lo) & (L <= hi)
        if m.sum() >= 30: xs.append(i); ys.append(100 * ok[m].mean())
    ax.plot(xs, ys, marker="o", ms=3, lw=1, label=TN[s])
    m = L <= 4; put(f"len.{s}.short", 100 * ok[m].mean() if m.sum() >= 30 else float("nan"))
ax.set_xticks(range(5)); ax.set_xticklabels(["1-4", "5-7", "8-10", "11-15", "16+"]); ax.set_xlabel("Message length (words)")
ax.set_ylabel("MARBERTv2 accuracy (%)"); ax.legend(fontsize=6, frameon=False); ax.grid(alpha=.3, lw=.4)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_length.pdf")

# ---- cascade with other components (threshold always chosen on validation with the same rule)
Lt = json.load(open(f"{R}/runs/llm_qwen/llm.json"))["rows"]; Lv = json.load(open(f"{R}/runs/llm_qwen/llm_val.json"))["rows"]["val"]
GRID = np.round(np.arange(0.30, 0.991, 0.01), 2)
def second(rows, kind, split):
    idx = np.array([r["i"] for r in rows]); y = np.array([r["y"] for r in rows])
    if kind in ("few", "zero"): return idx, y, np.array([r[kind] for r in rows])
    P, _ = avg_probs(["knn"], split); return idx, y, P.argmax(1)[idx]
def cascade(first, kind):
    idx, yv, sv = second(Lv, kind, "val"); P, _ = avg_probs(first, "val"); c, p = P.max(1)[idx], P.argmax(1)[idx]
    ok = [t for t in GRID if (c < t).any() and (sv[c < t] == yv[c < t]).mean() >= (p[c < t] == yv[c < t]).mean()]
    tau = float(max(ok)) if ok else float(GRID[0]); res = {}
    for s in T:
        idx, y, sv = second(Lt[s], kind, s); P, _ = avg_probs(first, s); c, p = P.max(1)[idx], P.argmax(1)[idx]
        res[s] = (100 * (np.where(c < tau, sv, p) == y).mean(), 100 * (c < tau).mean())
    return tau, res
configs = [(["marbertv2"], "few", "MARBERTv2 $\\to$ LLM (10 retrieved)"), (["bert-base-arabertv02"], "few", "AraBERTv02 $\\to$ LLM (10 retrieved)"),
           (["marbertv2", "bert-base-arabertv02"], "few", "MARBERTv2+AraBERTv02 $\\to$ LLM (10 retrieved)"),
           (["marbertv2"], "zero", "MARBERTv2 $\\to$ LLM (zero-shot)"), (["marbertv2"], "knn", "MARBERTv2 $\\to$ bge-m3 $k$NN")]
rows = []
for first, kind, name in configs:
    tau, res = cascade(first, kind)
    rows.append(f"{name} & {tau:.2f} & " + " & ".join(f"{res[s][0]:.1f}" for s in T) + f" & {res['test_tunisian'][1]:.1f} \\\\")
    key = "+".join(first) + "." + kind
    for s in T: put(f"casc2.{key}.{s}", res[s][0])
    put(f"casc2.{key}.tau", tau, "{:.2f}")
# encoder-only ensemble on the same subsamples, for reference
ens = []
for s in T:
    idx = np.array([r["i"] for r in Lt[s]]); y = np.array([r["y"] for r in Lt[s]]); P, _ = avg_probs(["marbertv2", "bert-base-arabertv02"], s)
    ens.append(100 * (P.argmax(1)[idx] == y).mean()); put(f"ens.{s}", ens[-1])
rows.append("MARBERTv2+AraBERTv02 ensemble (no LLM) & -- & " + " & ".join(f"{a:.1f}" for a in ens) + " & 0.0 \\\\")
open(f"{OUT}/tab_cascade2.tex", "w").write(
    "\\begin{table*}[t]\\centering\\small\\caption{Cascades built from other components, each with its threshold chosen on validation "
    "by Algorithm~\\ref{alg:tau}: accuracy (\\%) on the 400-message subsamples and share of Tunisian traffic sent to the second "
    "stage. The first row is the cascade of Table~\\ref{tab:cascade}.}\\label{tab:cascade2}\n"
    "\\begin{tabular}{lrrrrrrr}\\toprule\nFirst stage $\\to$ second stage & $\\tau$ & MSA & Pal. & Saudi & Moroccan & Tunisian & Routed (Tun.) \\\\\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

# ---- category-level heatmap (MARBERTv2 seed-averaged, probabilities summed per category)
cats = list(CATS); C = np.array([cats.index(cat_of(AR2EN[l])) for l in LABELS])
H = np.full((len(cats), len(T)), np.nan)
for j, s in enumerate(T):
    P, y = avg_probs(["marbertv2"], s); Pc = np.zeros((len(P), len(cats)))
    for k in range(len(cats)): Pc[:, k] = P[:, C == k].sum(1)
    pc, yc = Pc.argmax(1), C[y]
    for k in range(len(cats)):
        m = yc == k
        if m.sum() >= 20: H[k, j] = 100 * (pc[m] == k).mean()
keep = [k for k in range(len(cats)) if cats[k] != "other" and not np.isnan(H[k]).all()]
fig, ax = plt.subplots(figsize=(3.4, 3.4))
im = ax.imshow(H[keep], cmap="viridis", vmin=40, vmax=100, aspect="auto")
ax.set_xticks(range(len(T))); ax.set_xticklabels([TN[s] for s in T], rotation=30, ha="right")
ax.set_yticks(range(len(keep))); ax.set_yticklabels([cats[k].replace("_", " ") for k in keep])
for a in range(len(keep)):
    for b in range(len(T)):
        v = H[keep[a], b]
        if not np.isnan(v): ax.text(b, a, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white" if v < 75 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Category recall (%)"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_cathm.pdf")
worst = min(((H[k, 4], cats[k]) for k in keep if not np.isnan(H[k, 4])))
put("cathm.tun.worst", worst[0]); M["cathm.tun.worstname"] = worst[1].replace("_", " ")

# ---- selective prediction for every system
N = json.load(open(f"{R}/results/numbers.json"))
fams = [("marbertv2", "MARBERTv2"), ("bert-base-arabertv02", "AraBERTv02"), ("bert-base-arabic-camelbert-mix", "CAMeLBERT-mix"),
        ("xlm-roberta-base", "XLM-R base"), ("knn", "bge-m3 $k$NN"), ("laya", "Laya (RLCD)"), ("layace", "Laya (CE)")]
rows = [f"{n} & " + " & ".join(f"{N[f'{f}.{s}.aurc']:.1f}" for s in T) + " & " + " & ".join(f"{N[f'{f}.{s}.cov95']:.0f}" for s in T) + " \\\\"
        for f, n in fams if f"{f}.test_msa.aurc" in N]
open(f"{OUT}/tab_aurc.tex", "w").write(
    "\\begin{table*}[t]\\centering\\small\\caption{Selective prediction for every system. Left: area under the risk--coverage curve "
    "(AURC, \\%; lower is better). Right: share of traffic (\\%) that can be answered automatically at $\\geq$95\\% accuracy.}"
    "\\label{tab:aurc}\n\\begin{tabular}{l" + "r" * 10 + "}\\toprule\n& \\multicolumn{5}{c}{AURC} & \\multicolumn{5}{c}{Coverage at 95\\% accuracy}\\\\\n"
    "System & MSA & Pal. & Saudi & Moroccan & Tunisian & MSA & Pal. & Saudi & Moroccan & Tunisian \\\\\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

with open(f"{OUT}/extra_numbers.tex") as f: old = f.read().replace("\\makeatother\n", "")
open(f"{OUT}/extra_numbers.tex", "w").write(old + "\n".join(f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(M.items())) + "\n\\makeatother\n")
nj = json.load(open(f"{OUT}/numbers.json")); nj.update(M); json.dump(nj, open(f"{OUT}/numbers.json", "w"), indent=1, sort_keys=True)
print(len(M), "numbers"); [print(k, v) for k, v in sorted(M.items()) if k.startswith(("lex", "casc2", "ens", "cathm", "len"))]
