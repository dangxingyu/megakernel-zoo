# Engines

One directory per engine, grouped by model and GPU:

```
engines/<model>_<gpu>/<name>/engine/
    api.py      the Python binding: class Engine (the contract below); loads the weights, launches the kernel
    build.sh    builds everything from source on the machine that serves it (nvcc); must be repeatable
    TRACK       A, B or C, then optional track-B storage options (e.g. "B W12U L0T")
    *.cu, *.h   the kernel source
    README.md   optional: how the engine is organized, what it measured
```

- **`<model>`**: the Hugging Face name in lower case without the organization (`qwen3-8b`, `north-mini-code-1.0`).
- **`<gpu>`**: the registry key of the GPU (`h100`, `b200`), with `x<N>` for an engine that spans N GPUs of one node
  (`b200x4`).
- **`<name>`**: `offline-a`, `offline-b`, `offline-c` (steady-decode engines, track A, B or C) and `online-a`,
  `online-b` (serving engines). A second engine of the same kind and track gets a suffix (`offline-a2`).

Nothing large is committed: weights are read from the checkpoint, and tables (12-bit codes, layer-0 token tables, head
sketches) are built at load time. Build products (`*.so`, `build/`, `__pycache__/`) are ignored.

## The two contracts

**Offline (steady decoding)**: [`starter/engine/api.py`](../starter/engine/api.py) is the reference implementation.

```python
class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int): ...
    def reset(self) -> None                          # forget every sequence
    def step(self, tokens: torch.Tensor) -> torch.Tensor
        # tokens: int64 [B] on the GPU, one per sequence; returns next-token logits [B, vocab] on the GPU
    def generate(self, tokens, n_steps) -> torch.Tensor   # optional: greedy, n_steps tokens per sequence in one launch
    def stats(self) -> dict                          # optional: launches, barriers, formats that differ from the checkpoint
```

**Online (serving)**: the same class with a request interface; admission, batching, prefill and KV allocation are the
engine's own (full contract in [`eval/_online_runner.py`](../eval/_online_runner.py)).

```python
    def submit(self, rid: int, prompt: list[int], max_new_tokens: int) -> None   # returns at once
    def poll(self) -> list[tuple[int, list[int], bool]]                          # (rid, new tokens, finished)
    def idle(self) -> bool
    def kv_pages(self) -> tuple[int, int]                                        # (free, total)
    def drop_cache(self) -> None
```

## Tracks

- **Track A**: the checkpoint's weights, number formats and data flow; every operator runs for every token, one token
  per pass. Only the organization of the computation on the GPU differs from the reference implementation.
- **Track B**: any change of storage or load-time precomputation that leaves the function unchanged in exact
  arithmetic (lossless 12-bit weight codes, certified head sketches with exact rescoring, per-token tables for layer 0).
  `mkserve list` and the registry say, per engine, whether its logits are bit-identical to track A.
- **Track C**: exact speculative decoding. The engine drafts several tokens (for example with the checkpoint's own
  draft head) and verifies them with its track-A decode step in one pass; with greedy decoding its tokens through
  `submit`/`poll` equal its own `step()`-by-`step()` greedy tokens for every sequence. `step()` itself stays the
  one-token track-A step, which every numeric gate checks. Its time per token is counted in tokens received.

## The engines in this repository

| Model | GPU | Engines |
|---|---|---|
| Qwen3-8B | H100 | `offline-a`, `offline-b`, `online-a`, `online-b` |
| North Mini Code 1.0 | H100 | `offline-a`, `offline-b` |
| gpt-oss-20b | H100 | `offline-a`, `offline-b` |
| flint | B200 | `offline-a`, `offline-b` |
| Qwen3.8-27B | B200 | `offline-a`, `offline-b` |
| Qwen3-32B | B200 | `offline-a`, `offline-b` |
| gpt-oss-120b | B200 | `offline-a`, `offline-b` |
| Inkling-2-Small | 4 × B200 | `offline-a`, `offline-b`, `online-a` |
| Inkling-2-Mini | B200 | `offline-a`, `offline-b`, `offline-c` |

Where each one came from (run and submission), its limits and what it was checked with are in
[`mkserve/registry.json`](../mkserve/registry.json). Before opening a pull request, run:

```bash
python scripts/check_engine.py engines/<model>_<gpu>/<name>    # layout, contract, imports, registry, sha256
```
