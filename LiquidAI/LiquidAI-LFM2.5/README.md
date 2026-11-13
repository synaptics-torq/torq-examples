# LFM2.5 (LiquidAI) LLM Demo

Interactive text chat with LiquidAI **LFM2.5** using a Torq VMFB.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

Enter the demo directory. Install its dependencies. Jump back to the repo root.

```sh
cd LiquidAI/LiquidAI-LFM2.5
pip install -r requirements.txt
cd ../..
```

From the repo root — downloads the model from HuggingFace (230M by default;
`350m`, the w8a8 SL2619 builds `230m-w8a8`/`350m-w8a8` and the 4-bit
`350m-qad-q4_0` build are also available via the demo's own `setup_demo.py`):

```sh
python setup_demos.py LiquidAI-LFM2.5
```

The model version defaults to the one matching this repo's `VERSION` file (see the
[repo README](../../README.md) for the versioning scheme); `setup_demos.py` and the demo's
`setup_demo.py` both support `--model-version` to pin a specific release tag and `--no-update`
to skip model tracking entirely. By default setup does **not** download the optional
batched prefill model; pass `--with-batch-prefill` to also fetch and track
`transformer_prefill.vmfb` (an extra model, so it uses more memory, with fixed
prompt chunk sizes; a no-op with a warning for repos that have no prefill
model). Re-running setup without the flag removes it from tracking (the local file
is kept but no longer checked or refreshed).

This fetches the artifacts from
[`Synaptics/LiquidAI-LFM2.5-230M`](https://huggingface.co/Synaptics/LiquidAI-LFM2.5-230M)
into `models/Synaptics/LiquidAI-LFM2.5-230M/`:

```
models/Synaptics/LiquidAI-LFM2.5-230M/
├── transformer.vmfb           ← decoder body (hidden output, 256-token KV cache)
├── transformer_prefill.vmfb   ← optional: batched prefill build (setup only downloads it with the --with-batch-prefill flag)
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
> `transformer_prefill.vmfb`, which setup downloads with the `--with-batch-prefill` flag.

### LFM2.5-350M QAD Q4_0

`350m-qad-q4_0` is Liquid AI's
[quantization-aware-distilled Q4_0 checkpoint](https://huggingface.co/blog/LiquidAI/qad)
(`LFM2.5-350M-QAD-Q4_0.gguf` from
[`LiquidAI/LFM2.5-350M-GGUF`](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)),
running on the NPU with the GGUF's own weights, from
[`Synaptics/LiquidAI-LFM2.5-350M-QAD-Q4_0-torq`](https://huggingface.co/Synaptics/LiquidAI-LFM2.5-350M-QAD-Q4_0-torq).

> [!IMPORTANT]
> This build needs a Torq runtime **newer than 2.3.0**. The repo's `requirements.txt`
> installs `torq-runtime==2.3.0`, which rejects these files with `required module
> features [Ch] are not available`. After the repo setup, install a newer runtime wheel
> into the same virtual environment (tested with the development build
> `torq-runtime 0.dev0+20261023t220624z.g79394a3d`) and check the version:
>
> ```sh
> pip install --force-reinstall --no-deps torq_runtime-<version>-cp312-cp312-manylinux_2_39_aarch64.whl
> pip show torq-runtime | grep Version   # must be newer than 2.3.0
> ```

From the repo root, after the [repo setup](../../README.md#setup) (virtual environment,
`pip install -r requirements.txt`, and `python setup_demos.py LiquidAI-LFM2.5` once):

```sh
cd LiquidAI/LiquidAI-LFM2.5
pip install -r requirements.txt
python setup_demo.py 350m-qad-q4_0:main --with-batch-prefill
python src/infer.py \
  -m ../../models/Synaptics/LiquidAI-LFM2.5-350M-QAD-Q4_0-torq/transformer.vmfb \
  --instruct-model
```

The model repo has no `v2.3.0` tag, so pin `:main` as above; without it setup stops on
the missing tag. `--with-batch-prefill` is what fetches `transformer_prefill.vmfb`;
without it every prompt token runs through the decode model and the first answer
token takes several times longer.

Setup downloads into `models/Synaptics/LiquidAI-LFM2.5-350M-QAD-Q4_0-torq/`:

```
├── transformer.vmfb           ← decode, one token per step (read-only KV cache, 256 tokens)
├── transformer_prefill.vmfb   ← prompt prefill: 32-, 16-, 8- and 4-token chunks, one weight copy
├── lm_head.vmfb               ← lm_head (hidden -> logits)
├── token_embeddings.npy       ← CPU-side embedding table
├── config.json
└── tokenizer.json
```

At start-up the demo logs `Batched prefill chunk sizes: 32, 16, 8, 4`. Each prompt is
split greedily into those chunks; a leftover of 1-3 tokens runs through
`transformer.vmfb`. Nothing is re-quantized: every Q4_0 matrix of the decoder keeps the
GGUF's 4-bit codes and fp16 block scales (INT4, block 32), and the Q6_K tied embedding /
lm_head keeps its 6-bit codes and per-16 scales (INT8, block 16).

Measured on SL2619 (2 GHz CPU, 1 GHz NPU), greedy, against llama.cpp running the same
GGUF on the board's CPU (2 threads; prefill counts one lm_head per chunk, as
`llama-bench` does):

| tok/s | Torq NPU | llama.cpp CPU |
|---|---|---|
| Prefill, 4-token chunk | 45 | 43 |
| Prefill, 8-token chunk | 82 | 51 |
| Prefill, 16-token chunk | 140 | 65 |
| Prefill, 32-token chunk | 198 | 65 |
| Decode | 14.0 (demo 13.8) | 14.1 |

Time to first token in the demo is about 0.22 s for a short question and 0.75 s for a
150-token prompt. Next-token top-1 agreement with llama.cpp over 841 chat-answer
positions is 97.2-97.9% (the mismatches are near-ties). The demo uses about 500 MB of
memory (the three networks' NPU buffers, 421 MB, plus the Python process).

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
> sharply. A prefill model that also exports smaller chunk sizes as `main_<B>`
> functions (the `350m-qad-q4_0` build does) is used with all of them, largest
> first. The prefill build has the lm_head baked in for its final position.
> Any prompt remainder and all generated tokens still use `transformer.vmfb`,
> and both paths share the same KV cache. Use `--batch-prefill-model PATH` to point
> at a specific prefill model, or `--no-batch-prefill-model` to force single-token
> prompt prefill.

## Model notes

LFM2.5 is a hybrid conv + attention model: each layer is either a depthwise 1D
conv block (sliding `past_conv.N` state `[1, 1024, 3]`) or an attention block
(per-layer combined KV cache `[1, 256, 16, 64]` — 256-token window × 8 KV heads
for K and V × 64 head-dim; models before v2.4.0 use `[1, 16, 256, 64]`). The 230M build has 14 layers (8 conv + 6 attention); the
350M build has 16 layers (10 conv + 6 attention).

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
bit-coherent against the fp32 reference). From v2.4.0 the attention layers only
read their KV caches and output the step's new rows (`[1, 1, 16, 64]` per
decode step, `[1, 64, 16, 64]` per prefill chunk); the cache manager copies them
into the cache at the step position, and still replaces the conv states whole.

The token-embedding lookup is done on the CPU from `token_embeddings.npy`; the
VMFB takes the embedded vector `[1, 1, 1024]` as input 0 (not `input_ids`).

This example is a redistribution of a model created by **Liquid AI, Inc.**,
licensed under the **LFM Open License v1.0**.

> [!NOTE]
> The sliding-window KV shift (`--kv-cache-window` / `shift_kv`) is left off for
> LFM2.5: it slices a KV sequence axis, which is not safe for the rank-3 conv
> caches. Generation stops at the 256-token cache limit instead of shifting;
> normal short chat turns are unaffected.
