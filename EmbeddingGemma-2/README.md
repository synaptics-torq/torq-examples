# EmbeddingGemma-2 on Torq (text, image, video, audio)

[EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2) maps text, images,
video and audio into one 768-dimensional embedding space. This demo runs all four on the
SL2619 NPU (text backbone + vision encoder + audio encoder, about 740M parameters).

## How it runs

| Stage | Where | Notes |
|---|---|---|
| Tokenizer, prompt prefix | CPU | `tokenizer.json`; task prefixes such as `task: search result \| query: ` |
| Token-embedding lookup | CPU | `token_embeddings.npy` (262144 x 512 bf16, memory-mapped from eMMC) |
| Image / frame preprocessing | CPU | resize to 384x384, cut 576 patches of 16x16 |
| Vision encoder | NPU | `vision_384x384.vmfb`: 576 patches -> 64 soft tokens (512-d) |
| Audio front end | CPU | 16 kHz mono, 128-bin log-mel (`mel_filters.npy`, `mel_window.npy`), padded to the encoder window |
| Audio encoder | NPU | `audio_<frames>.vmfb` for 1.4 / 2.8 / 5.6 / 11.2 s windows (140 / 280 / 560 / 1120 frames) -> up to 35 / 70 / 140 / 280 soft tokens (512-d), one per 40 ms |
| Text body | NPU | `text_body_s128.vmfb` or `text_body_s512.vmfb`: the whole token sequence, text and soft tokens, -> 768-d per token |
| Mean pooling, Matryoshka truncation, L2 norm | CPU | |

The encoders do not produce the embedding themselves. Their soft tokens are already in the text
model's embedding space (the 512-d projection is part of each encoder), and they go through the
text body like words:

1. The host builds the token sequence, for example `<bos> <|image> 64 x <|image|> <image|> <eos>`
   for an image, and looks up every token in `token_embeddings.npy`.
2. The encoder's soft tokens replace the `<|image|>`, `<|video|>` or `<|audio|>` placeholder rows,
   in order. The surrounding marker tokens keep their text embeddings.
3. The sequence is right-padded to the text bucket and runs through the text body (24
   bidirectional layers) with a padding mask.
4. The host takes the mean over the real tokens, optionally truncates it to 512 / 256 / 128
   dimensions, and L2-normalises it.

So every image, video or sound costs one encoder pass and one text pass, and its embedding is
directly comparable with a text query's.

Each image becomes `<bos> <|image> 64 x <|image|> <image|> <eos>` (68 tokens).
A video is sampled at 1 fps (at most 7 frames), each frame runs through the vision encoder,
and the frames become one block each: `<bos> 7 x (<|image> 64 x <|video|> <image|>) <eos>`
(464 tokens).

Audio becomes `<bos> <|audio> n x <|audio|> <audio|> <eos>` with n = 25 tokens per second.
A session loads one audio encoder window. By default (`--audio-window auto`) that is the smallest
window that holds the longest clip. Shorter clips are zero-padded and only their n valid tokens
are used (identical to the unpadded result); longer clips are truncated to the window.
Smaller windows are faster (see the timings below).

The static 384x384 / 64-token geometry is what the HF processor produces for a square image
at the smallest token budget (70). Inputs are resized to a square, so non-square images and
video frames are squashed.

### Text body: two sequence lengths

The text body is compiled twice from the same weights, for 128 and for 512 tokens. Each input
runs in the smallest one that holds its token sequence; shorter sequences are padded and masked,
so the embedding does not depend on the bucket:

| Input | Tokens | Text body |
|---|---|---|
| Text query or short document | up to 128 | S=128 |
| Longer document | 129-512 | S=512 |
| Image | 68 | S=128 |
| Audio up to ~4.9 s | n + 4 <= 128 (n = 25 per second) | S=128 |
| Audio longer than ~4.9 s (up to 11.2 s: 284 tokens) | up to 512 | S=512 |
| Video (up to 7 frames x 66 + 2 = 464) | up to 512 | S=512 |

- S=128 is about 5.6x faster than S=512, and most queries and every image fit in it.
- 512 is the largest bucket because the model's sliding-window layers reach 512 tokens: up to
  513 tokens every layer is plain full attention. Longer inputs are rejected, not truncated.
- Each bucket holds its own copy of the weights (8-bit: 165 MB + 186 MB). If the requested
  networks do not fit in the NPU's mapping limit (the bf16 build with all modalities), the
  S=128 bucket is not loaded and everything runs at S=512, which is slower but gives the same
  embeddings.

## Inference time (SL2610 series)

Measured on an SL2619 board, one network per process, NPU time per call after a warm-up run.
The 8-bit build, the default, is compiled with the newer compiler optimisations (slicing,
prefetch and K-chunked matmuls); the 4-bit and bf16 builds will be rebuilt with them.

| Network | 8-bit (default) | 4-bit (`main`) | bf16 |
|---|---|---|---|
| text S=128 | 389 ms | 549 ms | 440 ms |
| text S=512 | 2185 ms | 2825 ms | 2128 ms |
| vision 384x384 (per image or frame) | 2734 ms | 3310 ms | 3909 ms |
| audio 1.4 s window | 302 ms | 279 ms | 332 ms |
| audio 2.8 s window | 463 ms | 600 ms | 527 ms |
| audio 5.6 s window | 935 ms | 1337 ms | 955 ms |
| audio 11.2 s window | 2041 ms | 2670 ms | 1886 ms |

| Network | 8-bit RAM / file | 4-bit RAM / file | bf16 RAM / file |
|---|---|---|---|
| text S=128 | 198 / 165 MB | 122 / 87 MB | 304 / 278 MB |
| text S=512 | 220 / 186 MB | 140 / 103 MB | 316 / 288 MB |
| vision 384x384 | 330 / 303 MB | 157 / 122 MB | 364 / 339 MB |
| audio 1.4 s | 352 / 328 MB | 210 / 180 MB | 601 / 589 MB |
| audio 2.8 s | 355 / 331 MB | 211 / 181 MB | 602 / 591 MB |
| audio 5.6 s | 361 / 337 MB | 217 / 186 MB | 605 / 593 MB |
| audio 11.2 s | 381 / 357 MB | 227 / 196 MB | 610 / 598 MB |

End to end per input with the 8-bit build (encoder + text body; host preprocessing adds about
50 ms per image, 25 ms per sound, and 1-2 s to decode a video):

| Input | Networks | Time |
|---|---|---|
| Text query | text S=128 | 0.39 s |
| Long document | text S=512 | 2.2 s |
| Image | vision + text S=128 | 3.1 s |
| Video, 7 frames | 7 x vision + text S=512 | 21.3 s |
| Sound up to 1.4 s | audio 1.4 s + text S=128 | 0.69 s |
| Sound up to 2.8 s | audio 2.8 s + text S=128 | 0.85 s |
| Sound up to 4.9 s | audio 5.6 s + text S=128 | 1.3 s |
| Sound 4.9-5.6 s | audio 5.6 s + text S=512 | 3.1 s |
| Sound 5.6-11.2 s | audio 11.2 s + text S=512 | 4.2 s |

The 8-bit timings are the same when all the networks run in one session (`--examples`).

## Usage

The model files come from [Synaptics/Google-EmbeddingGemma-2](https://huggingface.co/Synaptics/Google-EmbeddingGemma-2).
The default build has 8-bit weights (int8 in blocks of 32, bf16 activations), about 2.3 GB. Its
embeddings are much closer to the original model's than the 4-bit build's (cosine to the fp32
model over a media library: 0.997, against 0.976 for 4-bit), and it is as fast or faster on the
NPU (389 ms against 549 ms for a short query). `--model-version main` downloads the 4-bit build
(weights of the onnx-community `*_q4` export, about 1.1 GB), and `--model-version bf16` the bf16
build (about 3.6 GB, closest to the original model, with more memory per network). Each build
replaces the files of the previous one.

```sh
python setup_demos.py EmbeddingGemma-2      # download the model files and the examples
cd EmbeddingGemma-2
python src/infer.py --examples              # run the example media
```

`--examples` scores 14 example files against their 14 descriptions:
- six subjects, each with a photo and a sound: cat, dog, bird, piano, train and police car;
- two short videos: a waterfall and a sea turtle.

Each file ranks its own description first, well ahead of the other subjects' descriptions (checked
with the 8-bit build on the NPU). For example, the dog's bark matches "a dog barking and growling",
not "a cat meowing loudly". The files are CC0 or public domain; credits are in `examples/README.md`
next to them.

Your own queries and media work the same way, and can be combined with `--examples`:

```sh
python src/infer.py \
    --queries "a fluffy tabby and white cat with bright blue eyes" "a dog barking and growling" \
    --images my_cat.jpg --videos my_video.mp4 --audios my_dog.wav
```

`-m/--model-dir` defaults to the files `setup_demos.py` downloaded.

The board's NPU driver maps at most ~1.5 GB of networks at a time, and releasing a network
within a process is not reliable on the current driver, so a session keeps all the networks it
needs mapped. The 8-bit build (~1.3 GB) and the 4-bit build (~0.6 GB) fit all of them. With the
bf16 build, the vision and audio encoders plus both text buckets need ~1.7 GB. In that case
`infer.py` asks you to run `--audios` in a separate session; embeddings from both sessions live in
the same space and compare directly. For the examples, run `--examples visual` and then
`--examples audio`.

Options: `--backend ort` runs the fp32 ONNX graphs on the CPU with the same host code,
`--dim 256` truncates the embeddings (Matryoshka; renormalised), `--documents` adds text
documents.

### Media search: one index for text, images, video and sound

`src/media_search.py` builds a searchable index of a media folder on the NPU and answers
queries against it. It indexes every image, sound clip and video, and every second of each
video ("moments"), so a result can point to a time in a video. `--fetch-samples` downloads a
small library (20 photos, 5 videos, 10 sounds) from the transformers.js documentation assets.

```sh
python src/media_search.py index --media-dir media --fetch-samples
python src/media_search.py search --text "a dog barking"
python src/media_search.py search --image media/beach.png
```

```
query: text 'a dog barking'

  images:
     0.689  corgi.jpg
     0.603  pikachu.png
  sounds:
     0.724  dog_barking.wav
     0.697  cat_meow.wav
  ...
```

- **Text queries** search everything: "piano music" finds `piano.wav`, "a mountain lake" finds
  `moraine-lake.png`, "a turtle swimming in the ocean" finds `sea-turtle.mp4` and its first
  seconds.
- **An image as the query** finds similar images and video moments.
- Matching a sound directly against photos is not reliable: those scores are close to noise,
  so the default results stay within the query's own media type. `--kinds` shows the others.
- Indexing runs the vision and audio encoders in two processes (see the NPU mapping limit
  above). The index is one `.npz` file of 768-d embeddings plus their file names and times.

### Backend comparison

```sh
python src/compare_backends.py run -m <vmfb_dir> --backend torq -o torq.npz
python src/compare_backends.py run -m <onnx_dir> --backend ort --ort-threads 2 -o ort.npz
python src/compare_backends.py report torq.npz ort.npz -o report.md
# audio: separate sessions as well
python src/compare_backends.py run -m <vmfb_dir> --backend torq --modalities audio -o torq_audio.npz
```

The eval set (`eval/eval_set.json`) has 53 texts, 20 images, 5 videos and 10 audio clips
(pass `--audio` to `run` to include them); media files are downloaded from
`Xenova/transformers.js-docs` on first use.
