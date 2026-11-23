import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, json, numpy as np
sys.path.insert(0, os.getcwd())
import mlx.core as mx
import mk_kernels as kn
from mk_target import Target
from tokenizers import Tokenizer
MD = MODEL_DIR
tg = Target(MD, 4096)
tok = Tokenizer.from_file(MD + "/tokenizer.json")
texts = json.load(open(os.environ["TEXTS"]))
txt = "\n\n".join(t if isinstance(t, str) else json.dumps(t) for t in texts)
ids = tok.encode(txt).ids[:1536]
kn.EXP_STATS = []
tg.reset(); i = 0
while i < len(ids):
    t, _ = tg.prefill(ids[i:i + 256], 256); mx.eval(t, *tg.state()); i += 256
st = [(k, v.item()) for k, v in kn.EXP_STATS]
for kind in ("up", "down"):
    v = np.array([x for k, x in st if k == kind]); print(kind, "raw max over apps/chunks:", f"{v.max():.0f}", "median", f"{np.median(v):.0f}")
