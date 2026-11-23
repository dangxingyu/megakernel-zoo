# Inkling-2-Mini on Apple silicon: the general engines at batch size 1

Measured on one Apple M5 Max (40-core GPU, 48 GB, macOS 26.6.2) with the B200 track-C protocol: `eval/client.py steady`
over HTTP, client clock, greedy, one sequence. Results: [`eval/baselines_inkling-2-mini_m5max_c.json`](../../eval/baselines_inkling-2-mini_m5max_c.json).

| System | Weights | synth tok/s | natural tok/s | chat tok/s |
|---|---|---:|---:|---:|
| mlx-lm #2086 | the checkpoint as stored (NVFP4 / MXFP8 / BF16) | 32.9 | 32.8 | 34.6 |
| llama.cpp master | Q4_K_M | 39.8 | 40.0 | 41.1 |
| llama.cpp #32218 + DSpark | Q4_K_M, BF16 draft from `mtp.safetensors` | 48.6 | 61.3 | 80.0 |
| oMLX v0.8.3 | oQ4e (NVFP4 experts 0-19 as stored, rest 4/8-bit affine) | 59.7 | 59.6 | 60.5 |
| **oMLX v0.8.3 + DSpark** | oQ4e, bundled BF16 draft | **72.6** | **120.4** | **82.4** |

For comparison, one B200 with vLLM + DSpark: 647 / 1148 / 769 tok/s (`eval/baselines_inkling-2-mini_b200_c.json`).

## Workloads

- **synth**: the evaluator's synthetic 64-token prompts, 1,024 new tokens, `ignore_eos`; token-based decode rate.
- **natural**: `eval/natural_prompts.py` (64 tokens: Gutenberg prose, Python source, GSM8K), 1,024 new tokens,
  `ignore_eos`; same rate. Gutenberg answered 503 here; the generator now falls back to Project Gutenberg's mirror.
- **chat**: 48 GSM8K questions through the chat template (thinking on), until the model's own end token (at most 512;
  175-181 tokens on median); rate after the first token. `ignore_eos` is not used: past the answer the model emits
  special tokens that stream as nothing, so the last token's time would be early (llama.cpp: 6.4 s by its own log,
  4.8 s by the client).

All systems get text prompts (oMLX's completions endpoint takes no token ids; the natural prompts decode and encode
back to the same 64 ids, 48 of 48). Each system: 2 server starts in turns, 1 warm-up + 3 repeats per start, median.

## Reproduce

```bash
bash baselines/mac/serve.sh SYSTEM            # one system on 127.0.0.1:18600
ROUNDS=2 bash baselines/mac/bench.sh omlx-oq4e-ds omlx-oq4e llamacpp-q4-ds llamacpp-q4 mlxlm-native
python baselines/mac/summarize.py eval/baselines_inkling-2-mini_m5max_c.json runs-mac/<dirs>...
```

Setup (paths are `serve.sh`'s defaults):

- llama.cpp master in `~/src/llama.cpp` and ggml-org/llama.cpp#32218 (branch `pwilkin/inkling2-dspark`, the Inkling 2
  DSpark head) in `~/src/llama.cpp-dspark`, both `cmake -DGGML_METAL=ON`. The draft:
  `python convert_hf_to_gguf.py ~/models/Inkling-2-Mini --dspark --outtype bf16 --outfile ~/models/gguf-dspark-pr/Inkling-2-Mini-DSpark-bf16.gguf`
  (from #32218). Main model: `bartowski/thinkingmachines_Inkling-2-Mini-GGUF` Q4_K_M. Unsloth's GGUFs load in neither
  build (arch `inkling2` with 48 blocks; DSpark head without `markov_w2`, padded vocabulary).
- oMLX v0.8.3 (`git clone --branch v0.8.3 https://github.com/jundot/omlx`, `pip install -e .` in its own venv; no
  Xcode on this machine, so no native kernels, which Inkling does not use), model `mlx-community/Inkling-2-Mini-oQ4e-mtp`.
  oMLX renames its process to `omlx-server`: stop it by port.
- mlx-lm from ml-explore/mlx-lm#2086 in `.venv-mac`, on the checkpoint itself.

## What the numbers say

- At batch size 1 the M5 Max decodes 26 B / 5.4 B-effective-active Inkling-2-Mini at 33-60 tok/s without speculation,
  depending on the weight format; DSpark adds 1.2x on synthetic text and 1.5-2x on natural text and chat.
- Nothing general runs the checkpoint's own formats with speculation: mlx-lm (the stored formats) has no DSpark, and
  the two engines that speculate do it on re-quantized weights. A track-C engine on the stored formats starts 2.2x
  behind oMLX + DSpark on synthetic text (30.4 vs 13.8 ms per token) and 3.7x on natural text.
- oMLX streams a chunk every ~128 ms (9-15 tokens) and reports no per-chunk token counts; llama.cpp streams one token
  per chunk.
