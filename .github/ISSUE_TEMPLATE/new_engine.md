---
name: Propose an engine, model or GPU
about: Before a pull request, say what you plan to add so that it does not collide with work in progress
labels: engine
---

**Model** (Hugging Face id) **and GPU**:

**Track** (A: the checkpoint's formats and data flow; B: lossless re-encoding, function unchanged in exact arithmetic):

**Written by** (an agent run with `autoresearch/start.py`, by hand, or ported from):

**Current status** (`eval/kernel_eval.py` result, time per token at batch size 1 and 8, general engines measured on the same GPU):
