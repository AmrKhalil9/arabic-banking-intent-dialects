# Cross-dialect Arabic banking intent detection

Code, per-item predictions and results for the paper
**"Encoders, Decision Models or LLMs? Cross-Dialect Arabic Banking Intent Detection with a Confidence Cascade"**
(Amr Abdelkhalik).

The study compares fine-tuned Arabic encoders (MARBERTv2, AraBERTv02, CAMeLBERT-mix, XLM-R), a bge-m3 nearest-neighbour
baseline, the Laya decision model and Qwen3.5-9B (zero-shot and with 10 retrieved examples) on the official ArBanking77
test sets: MSA, Palestinian, Saudi, Moroccan and Tunisian. All trained systems use only the official MSA and Palestinian
training split. It also evaluates a confidence cascade (encoder first, LLM when the encoder is unsure).

## Layout

| Path | Contents |
|---|---|
| `code/` | data loading, training, baselines, analysis, cascade, table/figure generation |
| `results/` | `metrics.csv` (every run, split and metric), `numbers.json` (every number used in the paper), `cascade.json`, `leak_check.json` |
| `predictions/` | per-item predictions and confidences of every run, and the LLM's per-item answers |
| `paper/paper_tables/` | the generated tables, figures and number macros used by the manuscript |
| `reproduce.sh` | runs the full pipeline end to end |

## Data

ArBanking77 is **not** redistributed here. `reproduce.sh` clones it from
[SinaLab/ArBanking77](https://github.com/SinaLab/ArBanking77) at commit `2e3a563` (CC BY-SA 4.0). Some lines of the
released CSV files contain unescaped quotation marks, so `code/data.py` splits each line at the first field separator
instead of using a CSV parser; this keeps every message.

## Reproducing

```bash
pip install -r requirements.txt
./reproduce.sh
```

One 24 GB GPU was used. The LLM baseline expects a local Ollama server serving Qwen3.5-9B (tag set by `LLM_MODEL`, default `qwen3.5:9b`)
(set `LLM_URL` to point elsewhere). Every number in the paper is produced by `code/analyze.py`, `code/cascade.py` and
`code/make_tables.py`; none is typed by hand.

## Licence

Code: MIT. Predictions and results are derived from ArBanking77 and are released under CC BY-SA 4.0.
