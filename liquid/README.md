# LFM2.5 (LiquidAI) LLM Demo

Interactive chat with LiquidAI LFM2.5-350M using Torq VMFB models.

## Model files

```
models/Synaptics/LFM2.5-350M-torq/
├── model.vmfb            ← full decoder (logits output, bf16, 256-token KV cache)
├── body.vmfb            ← decoder minus lm_head (hidden output) — for the split path
├── lm_head.vmfb         ← standalone lm_head (hidden -> logits)
├── token_embeddings.npy  ← CPU-side embedding LUT (bf16)
├── config.json
└── tokenizer.json
```

## Running

From the demo directory:

```sh
cd liquidAI-LLM
python src/infer.py -m ../models/Synaptics/LFM2.5-350M-torq/model.vmfb --instruct-model
```

Multi-turn chat loop. Type `exit` or `quit` to stop; press <kbd>Ctrl</kbd>+<kbd>C</kbd> / <kbd>Ctrl</kbd>+<kbd>D</kbd> to interrupt an in-flight answer. Stats print per answer as `(<total_ms>, TTFT: <ms>, <tok/s>)`. `--instruct-model` enables the ChatML chat format + system-prompt warm-up (drop it for a base/completion model).

Run `python src/infer.py -h` for all options.

## Faster TTFT: split body + lm_head

The decoder runs one token per NPU invocation, and the `[1024, 65536]` lm_head
MatMul (~134 MB bf16) is computed on **every** prefill token even though only the
last one needs logits. You can split the decoder into a **body** (decoder minus
lm_head → hidden state) and a standalone **lm_head** (hidden → logits). The body
runs every step; the lm_head runs only when sampling (the last prefill token +
each decode step), so prefill skips the lm_head entirely:

```sh
cd liquidAI-LLM
python src/infer.py \
  -m ../models/Synaptics/LFM2.5-350M-torq/body.vmfb \
  --lm-head ../models/Synaptics/LFM2.5-350M-torq/lm_head.vmfb \
  --instruct-model
```

- `-m` is the **body** vmfb; `--lm-head` is the standalone lm_head vmfb.
- The body's hidden state (`[1,1,1024]`, 2 KB) is handed to the lm_head via the
  host (the two vmfbs are compiled independently, so their on-device layouts
  differ; the host normalize is negligible). The lm_head is preloaded — no
  per-call weight faulting.
- Measured: TTFT 2769 ms → **2101 ms** (−24%), decode 4.3 tok/s in both modes.
  The win scales with prompt length (more prefill tokens skipping the lm_head).

> [!IMPORTANT]
> Body (~577 MB) + lm_head (~134 MB) both preloaded sit at the edge of the NPU's
> device memory. Run the split as the **first NPU op after a clean (cold) boot**;
> if it reports `Cannot allocate memory`, the device is fragmented from a prior
> run — power-cycle and retry, or fall back to the monolithic `model.vmfb`.

## Model notes

LFM2.5 is a hybrid conv + attention model: each of the 16 layers is either a
depthwise 1D conv block (sliding `past_conv.N` state `[1, 1024, 3]`) or an
attention block (combined `past_key_values.X.key_value` `[1, 16, 256, 64]`).

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

## Building the split vmfbs

From the bf16 decoder ONNX, strip the lm_head (new output = the final-norm hidden
`[1,1,1024]`) and compile it as `body.vmfb`; compile the lone
`MatMul(hidden, embed_tokens.weight_T)` (ONNX graph named `main`) as
`lm_head.vmfb`. Both use the standard liquid compile flags
(`--torq-hw=SL2610 --torq-disable-slicing --torq-enable-transpose-optimization
--torq-convert-dtypes --torq-enable-annotate-tied-operands --torq-convert-io-dtype
--torq-enable-split-constants-optimization`).
