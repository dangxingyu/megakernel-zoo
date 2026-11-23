"""float32 reference logits (eval/inkling2_ref.py, mode high, CPU) on one 9,600-token text, at positions on both
sides of the log-scaling floor (8,192): the check of the engine's long-context path."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, time, glob, json
from pathlib import Path
import numpy as np, torch
R = ZOO
sys.path.insert(0, R + "eval")
import inkling2_ref
from tokenizers import Tokenizer
MD = str(Path.home() / "models/Inkling-2-Mini"); out = Path.home() / "mk-ref/inkling-2-mini-m5max-long"
tok = Tokenizer.from_file(MD + "/tokenizer.json")
files = [R + "runs-mac/deviation/README.md", R + "eval/README.md", R + "mkserve/README.md", R + "README.md"] + sorted(glob.glob(R + "eval/*.py")) + sorted(glob.glob(R + "mkserve/*.py"))
N = 9600
ids = np.array(tok.encode("\n\n".join(open(f).read() for f in files)).ids[:N], dtype=np.int64)[None]
pos = np.concatenate([np.arange(1023, 8192, 1024), np.arange(8191, 9600, 64)])
np.save(out / "ids.npy", ids); np.save(out / "positions.npy", pos)
t0 = time.time()
model = inkling2_ref.Inkling2(MD, devices=[torch.device("cpu")], mode="fork")
model.p = inkling2_ref.Prec("high", torch.float32)
print(f"loaded in {time.time() - t0:.0f}s; {N} tokens, {len(pos)} positions", flush=True)
with torch.no_grad():
    h = model.hidden(torch.from_numpy(ids), {})
    lg = model.logits(h[:, torch.from_numpy(pos)]).float().cpu()[0].numpy()
np.save(out / "logits.npy", lg)
(out / "meta.json").write_text(json.dumps({"tokens": N, "positions": [int(p) for p in pos], "impl": "eval/inkling2_ref.py mode high float32 cpu", "seconds": round(time.time() - t0)}))
print(f"done in {time.time() - t0:.0f}s", flush=True)
