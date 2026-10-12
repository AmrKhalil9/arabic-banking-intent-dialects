"""Appendix tables: leakage-check statistics, fitted temperatures per run, per-intent Tunisian results.
Usage: python extra_analysis5.py <out_dir>"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, AR2EN
R = str(__import__("pathlib").Path(__file__).resolve().parents[1]); OUT = sys.argv[1]
TN = {"test_pal": "Pal. (in-domain)", "test_saudi": "Saudi", "test_moroccan": "Moroccan", "test_tunisian": "Tunisian"}
EN = [AR2EN[l] for l in LABELS]
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def runs(f): return [d for d in sorted(glob.glob(f"{R}/runs/{f}_s*")) if os.path.exists(f"{d}/meta.json")] if f != "knn" else [f"{R}/runs/knn"]

L = json.load(open(f"{R}/runs/leak_check.json")); rows = []
for s, n in TN.items():
    d = L[s]
    rows.append(f"{n} & {int(d['n']):,} & {d['median_sim_train']:.2f} & {int(d['near_duplicate_train_ge_0.95'])} & {100*d['near_duplicate_train_ge_0.95']/d['n']:.1f} & {d['nn_in_msa_test_share']:.2f} \\\\".replace(",", "{,}"))
open(f"{OUT}/tab_leak.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Leakage check with bge-m3 embeddings: median cosine similarity of each test message to "
    "its nearest training message, number and share of test messages with a training message at similarity $\\geq 0.95$, and share "
    "whose nearest neighbour among training and MSA-test messages is an MSA-test message.}\\label{tab:leak}\n"
    "\\begin{tabular}{lrrrrr}\\toprule\nTest & $n$ & Median sim. & $\\geq 0.95$ & \\% & NN in MSA test \\\\\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\\end{table}\n")

def fit_temp(z, y):
    zt, yt = torch.tensor(z, dtype=torch.float64), torch.tensor(y)
    lt = torch.zeros(1, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=200)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(zt / lt.exp(), yt); l.backward(); return l
    opt.step(cl); return float(lt.exp().detach())
fams = [("marbertv2", "MARBERTv2"), ("bert-base-arabertv02", "AraBERTv02"), ("bert-base-arabic-camelbert-mix", "CAMeLBERT-mix"),
        ("xlm-roberta-base", "XLM-R base"), ("knn", "bge-m3 $k$NN"), ("laya", "Laya (RLCD)"), ("layace", "Laya (CE)"), ("layacoarse", "Laya (12-way)")]
rows = []
for f, n in fams:
    ts = [fit_temp(np.load(f"{d}/logits.npz")["val"], np.load(f"{d}/logits.npz")["y_val"]) for d in runs(f)]
    rows.append(f"{n} & " + ", ".join(f"{t:.2f}" for t in ts) + " \\\\")
open(f"{OUT}/tab_temps.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Temperatures fitted on the validation split for each run (one value per seed). Values "
    "below one sharpen the predicted distribution, values above one soften it.}\\label{tab:temps}\n"
    "\\begin{tabular}{ll}\\toprule\nSystem & Fitted temperature(s) \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

Lq = {r["i"]: r for r in json.load(open(f"{R}/runs/llm_qwen/llm.json"))["rows"]["test_tunisian"]}
d = runs("marbertv2"); P = np.mean([softmax(np.load(f"{x}/logits.npz")["test_tunisian"]) for x in d], 0); y = np.load(f"{d[0]}/logits.npz")["y_test_tunisian"]
pm = P.argmax(1); rows = []
for c in np.unique(y):
    m = y == c; idx = [i for i in np.where(m)[0] if i in Lq]
    llm = f"{100*np.mean([Lq[i]['few'] == c for i in idx]):.0f} ({len(idx)})" if idx else "--"
    rows.append((100 * (pm[m] == c).mean(), f"{EN[c]} & {m.sum()} & {100*(pm[m] == c).mean():.1f} & {llm} \\\\"))
rows.sort()
open(f"{OUT}/tab_tun_intents.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Per-intent recall (\\%) on the Tunisian test set: seed-averaged MARBERTv2 on all messages, "
    "and the retrieval-augmented LLM on the subsampled messages of that intent (number in parentheses).}\\label{tab:tunintents}\n"
    "\\begin{tabular}{lrrr}\\toprule\nIntent & $n$ & MARBERTv2 & LLM \\\\\\midrule\n" + "\n".join(r for _, r in rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
print("ok")
