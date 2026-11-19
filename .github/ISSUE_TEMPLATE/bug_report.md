---
name: Wrong output, crash or slow engine
about: An engine or mkserve gives a wrong answer, fails, or is slower than reported
labels: bug
---

**Model and engine** (`mkserve list`):

**GPU** (`nvidia-smi -L`), driver, CUDA toolkit (`nvcc --version`), PyTorch version:

**Command**:

```
```

**Output** (or the request and the response, for a wrong answer):

```
```

For a served engine, the output of `python scripts/mkserve_check.py --port <port> --engine <name> --model <model>`:

```
```
