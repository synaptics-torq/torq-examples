# LFM2.5 (LiquidAI) LLM Demo

Interactive text chat with LiquidAI **LFM2.5** using a Torq VMFB.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

Enter the demo directory. Install its dependencies. Jump back to the repo root.

```sh
cd LiquidAI/LiquidAI-LFM2.5
pip install -r requirements.txt
cd ..
```

From the repo root — downloads the model from HuggingFace (230M by default;
`350m` and the w8a8 SL2619 builds `230m-w8a8`/`350m-w8a8` are also available
via the demo's own `setup_demo.py`):

```sh
python setup_demos.py LiquidAI-LFM2.5
```

The model version defaults to the one matching this repo's `VERSION` file (see the
[repo README](../../README.md) for the versioning scheme); `setup_demos.py` and the demo's
`setup_demo.py` both support `--model-version` to pin a specific release tag and `--no-update`
to skip model tracking entirely. By default setup does **not** download the optional
batched prefill model; pass `--with-prefill` to also fetch and track
`transformer_prefill.vmfb` (an extra model, so it uses more memory, with a fixed
64-token prompt chunk size; a no-op with a warning for repos that have no prefill
model). Re-running setup without the flag removes it from tracking (the local file
is kept but no longer checked or refreshed).

This fetches the artifacts from
[`Synaptics/LiquidAI-LFM2.5-230M`](https://huggingface.co/Synaptics/LiquidAI-LFM2.5-230M)
into `models/Synaptics/LiquidAI-LFM2.5-230M/`:

```
models/Synaptics/LiquidAI-LFM2.5-230M/
├── transformer.vmfb           ← decoder body (hidden output, 256-token KV cache)
├── transformer_prefill.vmfb   ← optional: batched prefill build (setup only downloads it with the --with-prefill flag)
├── lm_head.vmfb               ← standalone lm_head (hidden -> logits; skipped during prefill)
├── token_embeddings.npy       ← CPU-side embedding LUT
├── config.json
└── tokenizer.json
```

> [!NOTE]
> Legacy deployments of `body.vmfb` + `lm_head.vmfb` (or the fused
> `model.vmfb`) still run; the demo picks whichever file set is present.

> [!NOTE]
> The `230m-w8a8`/`350m-w8a8` models come from the
> [`LiquidAI-LFM2.5-*-w8a8-torq`](https://huggingface.co/Synaptics/models?search=LFM2.5-w8a8)
> repos (the new exporter's SL2619 w8a8 builds). Besides `transformer.vmfb`
> and `lm_head.vmfb` they publish a batched prefill model in
> `transformer_prefill.vmfb`, which setup downloads with the `--with-prefill` flag.

## Running

The decoder is split into a **body** (decoder minus lm_head → hidden state) and a
standalone **lm_head** (hidden → logits). The `[1024, 65536]` lm_head MatMul
only produces logits, so it runs only when a token is sampled (the
last prefill token + each decode step) and is **skipped during prefill**:

```sh
cd LiquidAI/LiquidAI-LFM2.5
python src/infer.py \
  -m ../../models/Synaptics/LiquidAI-LFM2.5-230M/transformer.vmfb \
  --instruct-model
```

`-m` is the **body** vmfb; the standalone `lm_head.vmfb` next to it is picked
up automatically (`--lm-head PATH` to point elsewhere, `--no-lm-head` to run a
fused build that emits logits directly). Multi-turn chat loop — type `exit` or
`quit` to stop; press <kbd>Ctrl</kbd>+<kbd>C</kbd> / <kbd>Ctrl</kbd>+<kbd>D</kbd>
to interrupt an in-flight answer. Stats print per answer as
`(<total_ms>, TTFT: <ms>, <tok/s>)`. `--instruct-model` enables the ChatML chat
format + system-prompt warm-up (drop it for a base/completion model). Run
`python src/infer.py -h` for all options.

> The demo defaults to the DMA/dmabuf allocator with device I/O enabled. Use
> `--tda cpu` to run with the CPU allocator, or `--no-device-io` to pass user
> inputs as NumPy arrays.

> [!TIP]
> When the model directory contains a `transformer_prefill.vmfb` (a fixed-size
> batched prefill model exported alongside `transformer.vmfb`), the demo picks
> it up automatically: complete prompt chunks run through the prefill model and
> only the final prompt unit samples a token, so time-to-first-token drops
> sharply. The prefill build has the lm_head baked in for its final position.
> Any prompt remainder and all generated tokens still use `transformer.vmfb`,
> and both paths share the same KV cache. Use `--prefill-model PATH` to point
> at a specific prefill model, or `--no-prefill-model` to force single-token
> prompt prefill.

## Model notes

LFM2.5 is a hybrid conv + attention model: each layer is either a depthwise 1D
conv block (sliding `past_conv.N` state `[1, 1024, 3]`) or an attention block
(per-layer combined KV cache `[1, 16, 256, 64]` — 8 KV heads × 64 head-dim,
256-token window). The 230M build has 14 layers (8 conv + 6 attention); the
350M build has 16 layers (9 conv + 7 attention).

The new exporter's body VMFB takes three non-cache inputs — `token_embedding`
`[1, 1, 1024]`, `position_ids` `[1, 1]`, and a fixed all-ones
`attention_mask` `[1, 256]` sized at the compiled KV-cache window —
followed by the per-layer cache inputs. The shared runner builds all of them
(shapes, dtypes and the mask value) from the model's reflection metadata, so
legacy two-input exports keep working unchanged.

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
