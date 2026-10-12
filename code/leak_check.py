"""Leakage check for the dialect test sets, whose question IDs are not released.
For each Saudi/Moroccan/Tunisian test sentence: nearest neighbour (bge-m3 cosine) among MSA *test* sentences vs among
*training* sentences. A clean split puts the parallel MSA version in the MSA test set, so the nearest neighbour should
be an MSA test sentence far more often than a training sentence. Runs on CPU.
"""
import json, os, sys
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, os.path.dirname(__file__))
from common import xy

torch.set_num_threads(32)
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3"); m = AutoModel.from_pretrained("BAAI/bge-m3").eval()


@torch.no_grad()
def emb(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], truncation=True, max_length=96, padding=True, return_tensors="pt")
        out.append(torch.nn.functional.normalize(m(**b).last_hidden_state[:, 0], dim=-1))
    return torch.cat(out)


Etr = emb(xy("train")[0]); Xm, ym = xy("test_msa"); Em = emb(Xm)
res = {}
for s in ["test_saudi", "test_moroccan", "test_tunisian", "test_pal"]:
    X, y = xy(s); E = emb(X)
    st, it = (E @ Etr.T).max(1); sm, im = (E @ Em.T).max(1)
    closer_test = (sm > st).float().mean().item()
    same_label_msa = np.mean([ym[j] == yy for j, yy in zip(im.tolist(), y)])
    res[s] = {"n": len(X), "nn_in_msa_test_share": round(closer_test, 4), "median_sim_train": round(st.median().item(), 4),
              "median_sim_msa_test": round(sm.median().item(), 4), "near_duplicate_train_ge_0.95": int((st >= 0.95).sum()),
              "msa_test_nn_same_label": round(float(same_label_msa), 4)}
    print(s, res[s], flush=True)
json.dump(res, open(str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/runs/leak_check.json", "w"), indent=1)
