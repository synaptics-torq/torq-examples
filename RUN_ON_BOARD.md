# Running the LFM2-VL / LFM2.5 models on the SL2619 board

Handoff for a fresh session. The board is `root@10.3.10.55` (ssh key already set up, no password). Board Python venv (has `iree.runtime`, `torq.runtime`, `onnxruntime`, `numpy`, `ml_dtypes`, `tokenizers`, `pillow`):

```
~/torq/torq-examples/.venv/bin/python
```

## Setup + when to reboot

After a boot, run **`~/cpu.sh` once**: `ssh root@10.3.10.55 '~/cpu.sh'` — sets the CPU performance governor + NPU clock. **Required for correct compute** (without it the NPU can produce garbage/empty output). You don't need to re-run it before every model.

**Reboot only when stuck** — you don't need to reboot for every run. Reboot (then re-run `~/cpu.sh`) if you hit:
- `Failed to start network via IOCTL: Cannot allocate memory` / `failed to acquire hardware` (NPU device memory got too fragmented — heavy multi-model configs can do this after a few runs), or
- garbage / empty output (NPU wedged).

So: normally just run your command; **if it errors or garbles, reboot → `~/cpu.sh` → retry.**

Filter the NPU warm-up noise from output with: `2>&1 | grep -viE "Failed to wait|IOCTL"`.

## VLM — LFM2-VL-450M (image → caption)

Dir: `~/torq/torq-examples/liquidAI-VLM/`. Model dir: `../models/Synaptics/LFM2-VL-450M-torq/`.
`--image PATH` = one-shot; omit it for the interactive loop. `--prompt "..."` optional (default is image-only). `--native-res` is **on by default** (128-res image → 16 tokens; processes at native resolution, no padding).

**Most reliable — monolithic decoder + CPU vision (onnxruntime):**
```sh
cd ~/torq/torq-examples/liquidAI-VLM
python src/infer.py \
  -m ../models/Synaptics/LFM2-VL-450M-torq/decoder_main.vmfb \
  --vision ../models/Synaptics/LFM2-VL-450M-torq/vision_encoder.onnx \
  --image ../data/vlm/two-dogs-128.jpg --prompt "Describe the image."
```

**Lower TTFT — split body + lm_head (decoder), CPU vision:**
```sh
python src/infer.py \
  -m ../models/Synaptics/LFM2-VL-450M-torq/decoder_nolm.vmfb \
  --lm-head ../models/Synaptics/LFM2-VL-450M-torq/lm_head.vmfb \
  --vision ../models/Synaptics/LFM2-VL-450M-torq/vision_encoder.onnx \
  --image ../data/vlm/two-dogs-128.jpg
```

**Vision encoder on the NPU (128-res, faster encode ~0.53 s):** swap `--vision` to the vmfb.
The runner runs vision, **frees it, then loads the decoder** (they don't co-fit) — so this is **one-shot** (`--image` only, not the interactive loop):
```sh
python src/infer.py \
  -m ../models/Synaptics/LFM2-VL-450M-torq/decoder_nolm.vmfb \
  --lm-head ../models/Synaptics/LFM2-VL-450M-torq/lm_head.vmfb \
  --vision ../models/Synaptics/LFM2-VL-450M-torq/vision_encoder.vmfb \
  --image ../data/vlm/two-dogs-128.jpg
```

- `vision_encoder.vmfb` = 128-res (16 tokens). `vision_encoder_256.vmfb` = 256-res (64 tokens), used only with the image-prefill decoder (see below).
- Stats line: `(vision: <ms>, prefill: <N tok>, TTFT: <ms>, gen: <N tok @ tok/s>)`. `prefill` = input tokens (image + text); `gen` = generated; TTFT excludes model load.
- 256-res images for the 64-token path: `../data/vlm/two-dogs-256.jpg`, `cats-and-dogs-256.jpg`, `dogs-256.jpg`.

## LLM — LFM2.5-230M (text chat)

Dir: `~/torq/torq-examples/liquidAI-LLM/`. Model dir:
`../models/Synaptics/liquidAI-LFM2p5-230M-LLM/` — fetch with
`python setup_demos.py liquid`, or download the vmfbs +
`token_embeddings.npy` / `config.json` / `tokenizer.json` from
[`Synaptics/liquidAI-LFM2p5-230M-LLM`](https://huggingface.co/Synaptics/liquidAI-LFM2p5-230M-LLM).
Interactive chat loop; `--instruct-model` enables the ChatML/system-prompt; type
`exit` to quit. ~6.3 tok/s on the SL2619.

**Monolithic:**
```sh
cd ~/torq/torq-examples/liquidAI-LLM
python src/infer.py -m ../models/Synaptics/liquidAI-LFM2p5-230M-LLM/model.vmfb --instruct-model
```
**Split (lower TTFT — lm_head skipped during prefill, ~2381 → 1578 ms):**
```sh
python src/infer.py \
  -m ../models/Synaptics/liquidAI-LFM2p5-230M-LLM/body.vmfb \
  --lm-head ../models/Synaptics/liquidAI-LFM2p5-230M-LLM/lm_head.vmfb --instruct-model
```

## Key gotchas

- **Garbage / empty output** → you forgot `~/cpu.sh` after reboot (or the NPU is wedged → reboot).
- **`Cannot allocate memory` / `failed to acquire hardware`** → 2nd+ run since boot, or too many NPU models at once → reboot and run as the first op.
- **busybox board**: no `head -N` (use `awk 'NR<=N'`), no `pkill` (use `pgrep`+`kill`), plain `ps`.
- The decoder threads the conv/KV caches **manually** in `liquidAI-VLM/src/runner.py` (the generic `ManagedSelfAttnCacheRunner` garbles this model's hybrid conv+KV cache — do not switch back to it).
- Compare a vmfb vs its ONNX numerically with `liquidAI-VLM/scripts/compare_vision.py` (load both, feed same input, compare features).

## In-progress (not working yet)

**One-shot image prefill** (`decoder_image_2part_A/B.vmfb` → seed `decoder_nolm` caches, then text autoregressively): the runner integration is designed but **blocked** — the image-decoder vmfb is numerically broken on the NPU (produces NaN/overflow from layer 0; the per-token decoder over the same features is fine). Needs a recompile of `decoder_image_2part_A/B.vmfb`. Re-validate after recompile with `liquidAI-VLM/scripts/img_prefill_diag.py` (reboot → cpu.sh → run; caches must match the per-token reference, cosine ≈ 1.0).
