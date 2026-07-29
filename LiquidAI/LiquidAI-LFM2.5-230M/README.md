# LFM2.5 (LiquidAI) LLM Demo

Interactive text chat with LiquidAI **LFM2.5-230M** using a Torq VMFB.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

Enter the demo directory. Install its dependencies. Jump back to the repo root.

```sh
cd LiquidAI/LiquidAI-LFM2.5-230M
pip install -r requirements.txt
cd ..
```

From the repo root — downloads the 230M model from HuggingFace:

```sh
python setup_demos.py LiquidAI-LFM2.5-230M
```

This fetches the artifacts from
[`Synaptics/LiquidAI-LFM2.5-230M`](https://huggingface.co/Synaptics/LiquidAI-LFM2.5-230M)
into `models/Synaptics/LiquidAI-LFM2.5-230M/`:

```
models/Synaptics/LiquidAI-LFM2.5-230M/
├── body.vmfb             ← decoder minus lm_head (hidden output, bf16, 256-token KV cache)
├── lm_head.vmfb          ← standalone lm_head (hidden -> logits; skipped during prefill)
├── token_embeddings.npy  ← CPU-side embedding LUT (bf16)
├── config.json
└── tokenizer.json
```

## Running

The decoder is split into a **body** (decoder minus lm_head → hidden state) and a
standalone **lm_head** (hidden → logits). The `[1024, 65536]` lm_head MatMul
(~134 MB bf16) only produces logits, so it runs only when a token is sampled (the
last prefill token + each decode step) and is **skipped during prefill**:

```sh
cd LiquidAI/LiquidAI-LFM2.5-230M
python src/infer.py \
  -m ../../models/Synaptics/LiquidAI-LFM2.5-230M/body.vmfb \
  --lm-head ../../models/Synaptics/LiquidAI-LFM2.5-230M/lm_head.vmfb \
  --instruct-model
```

`-m` is the **body** vmfb; `--lm-head` is the standalone lm_head. Multi-turn chat
loop — type `exit` or `quit` to stop; press <kbd>Ctrl</kbd>+<kbd>C</kbd> /
<kbd>Ctrl</kbd>+<kbd>D</kbd> to interrupt an in-flight answer. Stats print per
answer as `(<total_ms>, TTFT: <ms>, <tok/s>)` — the 230M runs ~6.3 tok/s on the
SL2619, TTFT ~1.6 s (skipping the lm_head in prefill is ~34% faster than the
fused model). `--instruct-model` enables the ChatML chat format + system-prompt
warm-up (drop it for a base/completion model). Run `python src/infer.py -h` for
all options.

> [!TIP]
> The `body.vmfb` + `lm_head.vmfb` pair is produced by the exporter's
> `--split-decoder` flag:
> `torq-export-model liquid --model-size 230m --convert-dtypes --extract-embeddings --split-decoder`.
> (A fused single-file `model.vmfb` — run with just `-m`, no `--lm-head` — is also
> in the HF repo if you prefer the monolithic path.)

## Model notes

LFM2.5-230M is a hybrid conv + attention model with **14 layers** (8 depthwise-conv
+ 6 attention). Each layer is either a depthwise 1D conv block (sliding
`past_conv.N` state `[1, 1024, 3]`) or an attention block (per-layer KV cache,
8 KV heads × 64 head-dim, 256-token window).

The runner (`src/runner.py`) is a thin subclass of the shared
[`DecoderOnlyLLMRunner`](../../utils/llm.py); it only supplies the LFM2.5 ChatML
chat format, system-prompt warm-up, and stop conditions. Its
`ManagedSelfAttnCacheRunner` cache manager is agnostic to what each cached
tensor is — it zero-inits every per-layer cache from the model's input-shape
metadata and shuttles each present output back to its past input — so the
*mixed* conv/KV caches thread correctly with no special-casing (board-verified
bit-coherent against the fp32 reference).

The token-embedding lookup is done on the CPU from `token_embeddings.npy`; the
VMFB takes the embedded vector `[1, 1, 1024]` as input 0 (not `input_ids`).

This example is a redistribution of a model created by **Liquid AI, Inc.**,
licensed under the **LFM Open License v1.0**.

> [!NOTE]
> The sliding-window KV shift (`--kv-cache-window` / `shift_kv`) is left off for
> LFM2.5: it slices a KV sequence axis, which is not safe for the rank-3 conv
> caches. Generation stops at the 256-token cache limit instead of shifting;
> normal short chat turns are unaffected.
