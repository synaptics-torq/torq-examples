# LFM2.5 (LiquidAI) LLM Demo

Interactive text chat with LiquidAI **LFM2.5-230M** using a Torq VMFB.

## Setup

From the repo root — downloads the 230M model from HuggingFace:

```sh
python setup_demos.py liquid
```

This fetches the artifacts from
[`Synaptics/liquidAI-LFM2p5-230M-LLM`](https://huggingface.co/Synaptics/liquidAI-LFM2p5-230M-LLM)
into `models/Synaptics/liquidAI-LFM2p5-230M-LLM/`:

```
models/Synaptics/liquidAI-LFM2p5-230M-LLM/
├── model.vmfb            ← full decoder (logits output, bf16, 256-token KV cache)
├── body.vmfb             ← decoder minus lm_head (hidden output) — for the split path
├── lm_head.vmfb          ← standalone lm_head (hidden -> logits)
├── token_embeddings.npy  ← CPU-side embedding LUT (bf16)
├── config.json
└── tokenizer.json
```

## Running

```sh
cd liquid
python src/infer.py -m ../models/Synaptics/liquidAI-LFM2p5-230M-LLM/model.vmfb --instruct-model
```

Multi-turn chat loop. Type `exit` or `quit` to stop; press <kbd>Ctrl</kbd>+<kbd>C</kbd>
/ <kbd>Ctrl</kbd>+<kbd>D</kbd> to interrupt an in-flight answer. Stats print per
answer as `(<total_ms>, TTFT: <ms>, <tok/s>)` — the 230M runs ~6.3 tok/s on the
SL2619. `--instruct-model` enables the ChatML chat format + system-prompt warm-up
(drop it for a base/completion model).

Run `python src/infer.py -h` for all options.

## Faster TTFT: split body + lm_head

The decoder runs one token per NPU invocation, and the `[1024, 65536]` lm_head
MatMul (~134 MB bf16) produces logits — needed **only** when a token is sampled.
Split the decoder into a **body** (decoder minus lm_head → hidden state) and a
standalone **lm_head** (hidden → logits): the body runs every step; the lm_head
runs only when sampling (the last prefill token + each decode step), so
**prefill skips the lm_head entirely**.

```sh
cd liquid
python src/infer.py \
  -m ../models/Synaptics/liquidAI-LFM2p5-230M-LLM/body.vmfb \
  --lm-head ../models/Synaptics/liquidAI-LFM2p5-230M-LLM/lm_head.vmfb \
  --instruct-model
```

- `-m` is the **body** vmfb; `--lm-head` is the standalone lm_head vmfb.
- The body's hidden state (`[1,1,1024]`, 2 KB) is handed to the lm_head via the host.
- Measured on the SL2619: TTFT 2381 ms → **1578 ms** (−34%); decode 6.3 tok/s in
  both modes. The win scales with prompt length (more prefill tokens skip the lm_head).

> [!TIP]
> The `body.vmfb` + `lm_head.vmfb` pair is produced by the exporter's
> `--split-decoder` flag:
> `torq-export-model liquid --model-size 230m --convert-dtypes --extract-embeddings --split-decoder`.

## Model notes

LFM2.5-230M is a hybrid conv + attention model with **14 layers** (8 depthwise-conv
+ 6 attention). Each layer is either a depthwise 1D conv block (sliding
`past_conv.N` state `[1, 1024, 3]`) or an attention block (per-layer KV cache,
8 KV heads × 64 head-dim, 256-token window).

The runner threads these per-layer conv/KV caches **manually** (present → past,
kept on the torq device). The generic `ManagedSelfAttnCacheRunner` mishandles
this model's *mixed* conv/KV cache and yields degenerate output (repeated/garbage
tokens); the explicit manual threading is bit-exact against the fp32 reference.

The token-embedding lookup is done on the CPU from `token_embeddings.npy`; the
VMFB takes the embedded vector `[1, 1, 1024]` as input 0 (not `input_ids`).

> [!NOTE]
> The manual-cache runner does not implement the sliding-window KV cache
> (`--kv-cache-window` / `shift_kv`): generation stops at the 256-token cache
> limit instead of shifting. Normal short chat turns are unaffected.
