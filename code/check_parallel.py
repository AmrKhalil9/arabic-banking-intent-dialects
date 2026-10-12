"""Parallel leakage: is a test sentence the dialect version of an English question whose MSA/PAL versions are in train?"""
import csv, sys, io
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from data import load, D, TESTS
txt = (D / "Banking77_full_corpus.csv").read_text(encoding="utf-8-sig")
reader = csv.reader(io.StringIO(txt))
hdr = next(reader); rows = []
for r in reader:
    if len(r) != len(hdr):
        continue
    rows.append(dict(zip(hdr, r)))
print("full corpus rows parsed:", len(rows), "cols:", [h for h in hdr if h.startswith("Question_")])
qcols = [h for h in hdr if h.startswith("Question_") and h != "Question_en"]
by_text = {}
for r in rows:
    for c in qcols:
        t = (r.get(c) or "").strip()
        if t and t != "NULL":
            by_text.setdefault(t, set()).add(r["QID"])
train = set(x["text"] for x in load("train"))
train_qids = set(q for t in train for q in by_text.get(t, ()))
print("train sentences mapped to a QID:", sum(t in by_text for t in train), "/", len(train), "| distinct QIDs:", len(train_qids))
for s in TESTS:
    te = load(s); mapped = [by_text.get(x["text"], set()) for x in te]
    n_map = sum(bool(m) for m in mapped); n_leak = sum(bool(m & train_qids) for m in mapped)
    print(f"{s:14s} mapped {n_map}/{len(te)}  same-English-question-in-train {n_leak}")
