import json, os, re, sys
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_sequence
sys.path.insert(0, os.path.dirname(__file__))
from common import LABELS, AR2EN, xy
root = snapshot_download("convaiinnovations/laya"); mdir = os.path.join(root, "multilingual"); _fix_tokenizer_config(mdir)
tok = AutoTokenizer.from_pretrained(os.path.join(mdir, "tokenizer"))
key = lambda en: re.sub(r"[^a-z0-9]+", "_", en.lower()).strip("_")
text = max(xy("train")[0], key=len)
for name, crit in [("en+ar", {key(AR2EN[l]): f"{AR2EN[l]} ({l})" for l in LABELS}),
                   ("en", {key(AR2EN[l]): AR2EN[l] for l in LABELS})]:
    for ml in (512, 768, 1024, 1536):
        seq, mk = build_sequence(tok, text, {"t": "choice", "ins": "Which banking intent does this customer message express?", "crit": crit}, ml, ml)
        print(name, ml, "markers", len(mk), "seq len", len(seq))
