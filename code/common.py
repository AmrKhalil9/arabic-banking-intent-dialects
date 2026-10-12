"""Shared label space and split loading (official ArBanking77 splits, 77 intents)."""
import csv
from data import load, D, TESTS

_rows = list(csv.DictReader(open(D / "Banking77_intents.csv", encoding="utf-8-sig")))
AR2EN = {r["label_ar"].strip(): r["label_en"].strip() for r in _rows}
LABELS = sorted(AR2EN)                      # Arabic label strings, fixed order -> class index
IDX = {l: i for i, l in enumerate(LABELS)}
EVAL_SPLITS = ["val"] + TESTS


def xy(split):
    rows = load(split)
    return [r["text"] for r in rows], [IDX[r["label"]] for r in rows]
