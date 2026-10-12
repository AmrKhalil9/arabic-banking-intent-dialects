"""Robust loader for the official ArBanking77 splits (SinaLab/ArBanking77 @2e3a563, CC BY-SA 4.0).

The released CSVs contain unescaped double quotes inside some sentences, so a standard CSV parser silently drops rows.
Every data line has the form  "label","text"  (text may contain quotes), so we split on the first '","'.
"""
from pathlib import Path

D = __import__("pathlib").Path(__file__).resolve().parents[1] / "ArBanking77" / "data"
SPLITS = {"train": "MSA_PAL_train", "val": "MSA_PAL_val", "test_msa": "MSA_test", "test_pal": "PAL_test",
          "test_saudi": "Saudi_test", "test_moroccan": "Moroccan_test", "test_tunisian": "Tunisian_test"}
TESTS = [s for s in SPLITS if s.startswith("test_")]


def load(split):
    rows = []
    lines = (D / f"Banking77_Arabized_{SPLITS[split]}.csv").read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].replace('"', "").strip() == "label,text", lines[0]
    for i, line in enumerate(lines[1:], 2):
        line = line.strip()
        if not line:
            continue
        if not (line.startswith('"') and '","' in line and line.endswith('"')):
            raise ValueError(f"{split} line {i}: {line[:80]}")
        label, text = line[1:-1].split('","', 1)
        rows.append({"label": label.strip(), "text": text.strip()})
    return rows
