"""Write per-item predictions of every finished run (predicted label and confidence on val + every test split) and
the LLM's per-item answers to predictions/. Usage: python code/export_predictions.py"""
import csv, glob, gzip, json, os, shutil
import numpy as np
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
os.makedirs(f"{R}/predictions", exist_ok=True)
for f in sorted(glob.glob(f"{R}/runs/*/logits.npz")):
    run = os.path.basename(os.path.dirname(f))
    if not os.path.exists(os.path.join(os.path.dirname(f), "meta.json")): continue
    d = np.load(f)
    with gzip.open(f"{R}/predictions/{run}.csv.gz", "wt", newline="") as out:
        w = csv.writer(out); w.writerow(["split", "item", "label", "pred", "confidence"])
        for s in [k for k in d.files if not k.startswith("y_")]:
            z = d[s] - d[s].max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            for i, (y, pr, c) in enumerate(zip(d[f"y_{s}"], p.argmax(1), p.max(1))): w.writerow([s, i, int(y), int(pr), f"{c:.4f}"])
for name in ("llm.json", "llm_val.json"):
    if os.path.exists(f"{R}/runs/llm_qwen/{name}"): shutil.copy(f"{R}/runs/llm_qwen/{name}", f"{R}/predictions/{name}")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, AR2EN
from coarse import cat_of
json.dump([{"index": i, "label_ar": l, "label_en": AR2EN[l], "category": cat_of(AR2EN[l])} for i, l in enumerate(LABELS)],
          open(f"{R}/predictions/labels.json", "w"), ensure_ascii=False, indent=1)
print(sorted(os.listdir(f"{R}/predictions")))
