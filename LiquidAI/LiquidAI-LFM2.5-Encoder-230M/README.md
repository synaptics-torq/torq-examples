# LFM2.5-Encoder-230M — prompt routing on Torq

[LFM2.5-Encoder-230M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M) is
LiquidAI's bidirectional encoder built on the LFM2 hybrid backbone (gated
short-conv + grouped-query attention) with a masked-LM head. This demo runs
the encoder body on the Torq NPU (static 256-token sequence, bf16) and uses
it for **zero-shot prompt routing** in the spirit of the
[LiquidAI prompt-routing space](https://huggingface.co/spaces/LiquidAI/prompt-routing):
free-text routing lanes, one encoder pass per decision.

Because the 230M model is the *base* encoder (the fine-tuned Prompt-Router
exists only at 350M), routing here is done zero-/few-shot through the MLM
head: the prompt is wrapped in a small template ending in
`Category: <|mask|>` (with one in-context example per lane), and the mask
logits of each lane's first token are compared. The mask-position logits are
computed on the host as `hidden[mask] @ token_embeddings.T` over just the
lane tokens — the 65536-row lm_head matmul never runs on chip.

## Files

```
models/Synaptics/LiquidAI-LFM2.5-Encoder-230M/
├── body_s256.vmfb          encoder body, [1,256,1024] bf16 embeds + [1,256] mask -> hidden
├── token_embeddings.npy    65536 x 1024 bf16 LUT (tied MLM head)
├── tokenizer.json / config.json / encoder_manifest.json
```

Produced by `torq-export-model liquid-encoder -s 230m --seq-len 256` in
torq-tools.

## Run

```sh
python setup_demo.py                 # download model files
python src/demo.py -m ../../models/Synaptics/LiquidAI-LFM2.5-Encoder-230M/body_s256.vmfb
```

Interactive commands: type a prompt to route it; `/add <name> :: <example>`
adds a lane, `/del <name>` removes one, `/mask <text with <|mask|>>` runs a
one-off fill-mask query, `/routes` lists lanes, Ctrl-D exits.

One-shot: `python src/demo.py -m <model> -p "How do I fix this segfault?"`.

For host-side development the runner also accepts the fp32 ONNX export
(`-m body_s256.onnx`, needs `onnxruntime`).
