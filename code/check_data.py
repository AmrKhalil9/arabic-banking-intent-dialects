import subprocess, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from data import load, SPLITS, D
tr = load("train"); trset = set(r["text"] for r in tr); labels = set(r["label"] for r in tr)
for s in SPLITS:
    r = load(s)
    raw = int(subprocess.run(["grep", "-cv", r"^\s*$", str(D / f"Banking77_Arabized_{SPLITS[s]}.csv")],
                             capture_output=True, text=True).stdout) - 1
    print(f"{s:14s} rows {len(r):6d} (raw {raw}) labels {len(set(x['label'] for x in r)):3d} "
          f"unseen-labels {len(set(x['label'] for x in r) - labels)} exact-overlap-train {sum(x['text'] in trset for x in r)}")
