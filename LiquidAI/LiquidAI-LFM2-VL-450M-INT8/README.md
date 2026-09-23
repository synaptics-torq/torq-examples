# LFM2-VL-450M INT8 — Image Q&A on the Torq NPU

Ask questions about an image and get an answer, using LiquidAI's **LFM2-VL-450M**
vision-language model running entirely on the Synaptics Torq NPU — including the
vision encoder. No CPU inference, no onnxruntime.

This demo uses the **INT8 (W8)** weights: the five model files total ~750 MB, about half the
size of the bf16 set, which is what lets every part of the model stay loaded in
memory at once. Switching images and asking follow-up questions is fast because
nothing has to be reloaded.

## What it does

Point it at a photo and ask:

```
image:  cats-and-dogs-256.jpg   (a beagle puppy and a tabby kitten)
you:    What animals are in the picture?
model:  The picture shows a small beagle puppy and a gray tabby kitten ...
```

You can ask several questions about the same image, and load a new image at any
time without restarting.

## The models

The model is split into five compiled artifacts (`.vmfb`) that the demo loads and
keeps resident. `W8` = 8-bit integer weights; `bf16` = the original 16-bit
floating-point set from the same
[HF repo](https://huggingface.co/Synaptics/LiquidAI-LFM2-VL-450M) (see its model
card for architecture and licensing details).

| File (W8)                      | Role                                          | W8 size | bf16 size |
|--------------------------------|-----------------------------------------------|--------:|----------:|
| `vision_encoder_256_W8.vmfb`   | image encoder (256×256 → 64 image tokens)      | 106 MB  | 199 MB  |
| `decoder_image_2part_A_W8.vmfb`| image-prefill decoder, part A (layers 0–7)     | 150 MB  | 346 MB  |
| `decoder_image_2part_B_W8.vmfb`| image-prefill decoder, part B (layers 8–15)    | 132 MB  | 304 MB  |
| `decoder_nolm_W8.vmfb`         | text decoder body (generates the answer)       | 290 MB  | 577 MB  |
| `lm_head_W8.vmfb`              | output head (hidden → next-token scores)       | 67 MB   | 134 MB  |
| **Total**                      |                                               | **746 MB** | **1561 MB** |

The two `decoder_image_2part_*` files run back-to-back to fold the 64 image
tokens into the decoder's context in one shot. The download also includes the
shared `token_embeddings.npy` (the text-embedding table, kept at bf16 in both
sets), `config.json`, and `tokenizer.json`. The W8 reduction above is in the
compiled weights; the embedding table is identical in both sets and not counted.

## Setup

See the repo [README.md](../README.md) for installing the virtual environment and
base dependencies. Then, from the repo root:

```sh
cd LiquidAI/LiquidAI-LFM2-VL-450M-INT8
pip install -r requirements.txt
cd ../..
python setup_demos.py LiquidAI-LFM2-VL-450M-INT8
```

`setup_demos.py` downloads the model set (~884 MB: the five W8 vmfbs plus the
shared 134 MB embedding table and tokenizer) from
[`Synaptics/LiquidAI-LFM2-VL-450M`](https://huggingface.co/Synaptics/LiquidAI-LFM2-VL-450M)
into `models/Synaptics/LiquidAI-LFM2-VL-450M/`.

## Running

From the `LiquidAI/LiquidAI-LFM2-VL-450M-INT8` directory:

```sh
MODELS=../../models/Synaptics/LiquidAI-LFM2-VL-450M
python src/infer.py \
  --model-dir $MODELS \
  --image $MODELS/cats-and-dogs-256.jpg
```

The image is encoded once, then you get an interactive prompt:

```
Ask questions about the image.
  /image <path>  encode a new image   /quit  exit
Q: What animals are in the picture?
Agent: The picture shows a small beagle puppy and a gray tabby kitten ...
  (TTFT 1900 ms | prompt 82 tok, gen: 83 tok @ 5.8 tok/s)
Q: /image /path/to/another.jpg
  image loaded: /path/to/another.jpg
Q: exit
```

Use `--prompt "..."` to answer a single question and exit instead of looping.
<kbd>Ctrl</kbd>+<kbd>C</kbd> interrupts a generation. `python src/infer.py -h`
lists all options (`--max-new`, `--temperature`, `--top-p`, `--top-k`, ...).

## Python API

```python
import sys
sys.path.insert(0, "<repo>/LiquidAI/LiquidAI-LFM2-VL-450M-INT8/src")
from lfm2vl import load_models

vl = load_models("<model dir>")
vl.encode_image(img_bgr)            # any HxWx3 BGR numpy array
answer = vl.infer("What is this?")  # -> str
for chunk in vl.infer_stream("..."):  # or stream the text
    print(chunk, end="", flush=True)
vl.close()
```

`encode_image` accepts a raw array on purpose, so any image source — a file, a
screenshot, or a camera frame — plugs straight in.

## Reference performance

On an SL2619 board with all models loaded: first token ~4 s (the image is encoded
and the question prefilled), then ~6 tokens/second while generating.

---

This example is a redistribution of a model created by **Liquid AI, Inc.**,
licensed under the **LFM Open License v1.0**.
