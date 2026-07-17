# LFM2-VL-450M Demo

Image + prompt → caption/answers with LiquidAI **LFM2-VL-450M** on the Torq NPU.
The SigLIP vision encoder and the LFM2 decoder both run on the NPU (bf16); the image
is encoded once and cached, so you can ask multiple questions about it in one session.

## Setup

From the repo root, run:

```sh
python setup_demos.py LiquidAI-LFM2-VL-450M
```

This downloads the model files to:

```sh
models/Synaptics/LiquidAI-LFM2-VL-450M/
```

(~1.6 GB: `vision_encoder_256.vmfb`, `decoder_image_2part_{A,B}.vmfb`,
`decoder_nolm.vmfb`, `lm_head.vmfb`, `token_embeddings.npy`, `config.json`,
`tokenizer.json`, plus a sample `cats-and-dogs-256.jpg`.)

## Running

Run the demo from the `LiquidAI/LiquidAI-LFM2-VL-450M` directory:

```sh
cd LiquidAI/LiquidAI-LFM2-VL-450M
MODELS=../../models/Synaptics/LiquidAI-LFM2-VL-450M
python src/infer.py \
  -m              $MODELS/decoder_nolm.vmfb \
  --lm-head       $MODELS/lm_head.vmfb \
  --vision        $MODELS/vision_encoder_256.vmfb \
  --image-decoder $MODELS/decoder_image_2part_ \
  --image         $MODELS/cats-and-dogs-256.jpg
```

This encodes the image once, then drops into an interactive prompt so you can ask
several questions about it — the cached image-prefill keeps every follow-up fast:

```
Ask questions about the image ('exit'/'quit' to stop).
Q: What is the breed of the dog?
Agent: The dog appears to be a Beagle mix, with a tricolor black/white/brown coat ...
Q: How many animals are there?
Q: exit
```

Press <kbd>Ctrl</kbd>+<kbd>C</kbd>/<kbd>D</kbd> to interrupt a generation. Add
`--prompt "..."` to ask a single question and exit instead of looping.

Run `python src/infer.py -h` for all options (`--cpu-lm-head`, `--image-decoder`
splits, `--native-res`, etc.).
