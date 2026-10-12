"""Tables, figures and \\R{} number macros for the paper, from results/numbers.json + results/cascade.json.
Usage: python make_tables.py <out_dir>"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = str(__import__("pathlib").Path(__file__).resolve().parents[1]); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
N = json.load(open(f"{R}/results/numbers.json")); C = json.load(open(f"{R}/results/cascade.json"))
T = ["test_msa", "test_pal", "test_saudi", "test_moroccan", "test_tunisian"]
TN = ["MSA", "Pal.", "Saudi", "Moroccan", "Tunisian"]
SYS = [("marbertv2", "MARBERTv2"), ("bert-base-arabertv02", "AraBERTv02"), ("bert-base-arabic-camelbert-mix", "CAMeLBERT-mix"),
       ("xlm-roberta-base", "XLM-R base"), ("knn", "bge-m3 $k$NN"), ("laya", "Laya (RLCD)"), ("layace", "Laya (CE)")]
SYS = [(f, n) for f, n in SYS if f"{f}.test_msa.acc" in N]
M = {}
def put(k, v, fmt="{:.1f}"): M[k] = fmt.format(v) if isinstance(v, (int, float, np.floating)) else str(v); return M[k]
for k, v in N.items():
    if isinstance(v, (int, float)): put(k, v, "{:.0f}" if k.startswith("leak.") and (k.endswith(".n") or "near_dup" in k) else "{:.2f}" if k.startswith("leak.") else "{:.1f}")
for s, d in C["tests"].items():
    for k, v in d.items(): put(f"cascade.{s}.{k}", v, "{}" if k == "n" else "{:.1f}")
for s in T:
    if f"laya_vs_best.{s}.d" in N:
        put(f"laya_gap.{s}.d", -N[f"laya_vs_best.{s}.d"]); put(f"laya_gap.{s}.lo", -N[f"laya_vs_best.{s}.hi"]); put(f"laya_gap.{s}.hi", -N[f"laya_vs_best.{s}.lo"])
put("cascade.tau", C["tau"], "{:.2f}"); put("cascade.val_bert", 100 * C["val_bert"]); put("cascade.val_llm", 100 * C["val_llm"])
for s in T:
    d = C["tests"][s]; gap = d["llm"] - d["bert"]
    if gap > 0: put(f"cascade.{s}.recovered", 100 * (d["cascade"] - d["bert"]) / gap, "{:.0f}")

def cell(f, s, k, sd=True):
    v = N.get(f"{f}.{s}.{k}"); e = N.get(f"{f}.{s}.{k}.sd")
    if v is None: return "--"
    return f"{v:.1f}" + (f"\\,{{\\scriptsize$\\pm${e:.1f}}}" if sd and e is not None and N.get(f"{f}.{s}.seeds", 1) > 1 else "")

def best_bold(rows, col):
    vals = [float(r[col].split("\\")[0]) for r in rows if r[col] != "--"]
    return max(vals) if vals else None

def table(name, caption, label, metric):
    rows = [[n] + [cell(f, s, metric) for s in T] for f, n in SYS]
    for j in range(1, 6):
        b = best_bold(rows, j)
        for r in rows:
            if r[j] != "--" and float(r[j].split("\\")[0]) == b: r[j] = "\\textbf{" + r[j].split("\\,")[0] + "}" + ("\\," + r[j].split("\\,", 1)[1] if "\\," in r[j] else "")
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    open(f"{OUT}/{name}.tex", "w").write(
        "\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\\caption{" + caption + "}\\label{" + label + "}\n\\begin{tabular}{l" + "r" * 5 + "}\\toprule\n"
        "System & " + " & ".join(TN) + " \\\\\\midrule\n" + body + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

table("tab_acc", "Accuracy (\\%) on the official test sets, mean $\\pm$ standard deviation over three seeds. All systems are trained on "
      "(or, for $k$NN, retrieve from) the MSA and Palestinian training split only; Saudi, Moroccan and Tunisian are unseen dialects.", "tab:acc", "acc")
table("tab_f1", "Macro-F1 (\\%), averaged over the intents present in each test set (Tunisian covers 27 of 77).", "tab:f1", "macro_f1")

rows = []
for s, n in zip(T, TN):
    d = C["tests"][s]
    v = [d['bert'], d['llm'], d['cascade']]; b = max(v)
    cells = [("\\textbf{%.1f}" if x == b else "%.1f") % x for x in v]
    rows.append(f"{n} & " + " & ".join(cells) + f" & {d['to_llm']:.1f} \\\\")
open(f"{OUT}/tab_cascade.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\\caption{Confidence cascade on the LLM subsample (" + str(C["tests"]["test_msa"]["n"]) +
    " items per test set): MARBERTv2 answers when its confidence is at least $\\tau=" + f"{C['tau']:.2f}" + "$ (chosen on validation), "
    "otherwise the retrieval-augmented LLM answers. Accuracy (\\%) and share of traffic routed to the LLM.}\\label{tab:cascade}\n"
    "\\begin{tabular}{lrrrr}\\toprule\nTest & Encoder & LLM & Cascade & Routed (\\%) \\\\\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\\end{tabular}\\end{table}\n")

cal = [f for f, _ in SYS if f in ("marbertv2", "bert-base-arabertv02", "knn", "laya", "layace")]
rows = []
for f, n in SYS:
    if f not in cal: continue
    rows.append(n + " & " + " & ".join(f"{N[f'{f}.{s}.ece_raw']:.1f}/{N[f'{f}.{s}.ece']:.1f}" for s in T) + " \\\\")
open(f"{OUT}/tab_calib.tex", "w").write(
    "\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\\caption{Expected calibration error (\\%, 15 bins) before / after temperature scaling fitted on the "
    "in-domain validation split.}\\label{tab:calib}\n\\begin{tabular}{l" + "r" * 5 + "}\\toprule\nSystem & " + " & ".join(TN) +
    " \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

rows = []
for f, n in SYS:
    if f not in cal: continue
    rows.append(n + " & " + " & ".join(f"{N[f'{f}.{s}.coarse_acc']:.1f}" for s in T) + " & " +
                " & ".join(f"{N[f'{f}.{s}.cov90']:.0f}" for s in T) + " \\\\")
if "layacoarse.test_msa.acc" in N:
    rows.append("Laya (12-way) & " + " & ".join(f"{N[f'layacoarse.{s}.acc']:.1f}" for s in T) + " & " +
                " & ".join(f"{N[f'layacoarse.{s}.cov90']:.0f}" for s in T) + " \\\\")
open(f"{OUT}/tab_routing.tex", "w").write(
    "\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\\caption{Routing view. Left: accuracy (\\%) at the level of 12 contact-centre categories (77-way "
    "predictions summed per category). Right: share of traffic (\\%) that can be answered automatically at $\\geq$90\\% accuracy "
    "(selective prediction by confidence).}\\label{tab:routing}\n\\begin{tabular}{l" + "r" * 10 + "}\\toprule\n"
    "& \\multicolumn{5}{c}{12-category accuracy} & \\multicolumn{5}{c}{Coverage at 90\\% accuracy}\\\\\n"
    "System & " + " & ".join(TN) + " & " + " & ".join(TN) + " \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

# figure 1: accuracy vs test set; figure 2: cascade curves
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for (f, n), mk in zip([s for s in SYS if s[0] in ("marbertv2", "knn", "laya", "layace")], "osv^"):
    ax.plot(TN, [N[f"{f}.{s}.acc"] for s in T], marker=mk, ms=3.5, lw=1, label=n.replace("$k$", "k"))
ax.plot(TN, [C["tests"][s]["llm"] for s in T], marker="D", ms=3.5, lw=1, ls="--", label="LLM 10-shot*")
ax.plot(TN, [C["tests"][s]["cascade"] for s in T], marker="*", ms=5, lw=1.4, color="black", label="Cascade*")
ax.set_ylabel("Accuracy (%)"); ax.grid(alpha=.3, lw=.4); ax.legend(fontsize=6, frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_dialects.pdf")
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for s, n in zip(["test_saudi", "test_moroccan", "test_tunisian"], ["Saudi", "Moroccan", "Tunisian"]):
    c = np.array(C["curve"][s]); ax.plot(c[:, 2], c[:, 1], lw=1, label=n)
    d = C["tests"][s]; ax.scatter([d["to_llm"]], [d["cascade"]], s=12, zorder=3)
ax.set_xlabel("Traffic sent to the LLM (%)"); ax.set_ylabel("Accuracy (%)"); ax.grid(alpha=.3, lw=.4)
ax.legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/fig_cascade.pdf")

lines = ["\\makeatletter"] + [f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(M.items())] + \
        ["\\makeatother", "\\newcommand{\\R}[1]{\\ifcsname r@#1\\endcsname\\csname r@#1\\endcsname\\else\\textbf{??#1}\\fi}"]
open(f"{OUT}/numbers.tex", "w").write("\n".join(lines) + "\n"); json.dump(M, open(f"{OUT}/numbers.json", "w"), indent=1, sort_keys=True)
print(len(M), "numbers;", sorted(os.listdir(OUT)))
