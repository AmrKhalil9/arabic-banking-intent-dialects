#!/usr/bin/env bash
# Reproduce every number in the paper. Needs one CUDA GPU (24 GB was used) and, for the LLM baseline,
# a local Ollama server serving Qwen3.5-9B (tag via LLM_MODEL, server via LLM_URL).
set -euo pipefail
cd "$(dirname "$0")"
[ -d ArBanking77 ] || git clone https://github.com/SinaLab/ArBanking77.git && git -C ArBanking77 checkout 2e3a563
python code/check_data.py && python code/leak_check.py

for seed in 1 2 3; do
  for m in UBC-NLP/MARBERTv2 aubmindlab/bert-base-arabertv02 CAMeL-Lab/bert-base-arabic-camelbert-mix FacebookAI/xlm-roberta-base; do
    python code/train_encoder.py "$m" "$seed" "runs/$(basename "$m" | tr 'A-Z' 'a-z')_s$seed"
  done
  python code/train_laya.py "$seed" "runs/laya_s$seed" 3                                   # released RLCD recipe
  LOSS=ce MODE=fine SELECT=1 python code/train_laya.py "$seed" "runs/layace_s$seed" 5     # encoders' protocol
  LOSS=rlcd MODE=coarse SELECT=1 python code/train_laya.py "$seed" "runs/layacoarse_s$seed" 3   # 12 categories
done
python code/knn_llm.py knn runs/knn
python code/knn_llm.py llm runs/llm_qwen "${LLM_MODEL:-qwen3.5:9b}" 400
LLM_SPLITS=val LLM_OUT=llm_val.json python code/knn_llm.py llm runs/llm_qwen "${LLM_MODEL:-qwen3.5:9b}" 400

python code/analyze.py && python code/cascade.py && python code/make_tables.py paper/paper_tables
