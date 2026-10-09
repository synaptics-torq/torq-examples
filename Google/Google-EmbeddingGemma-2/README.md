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
| Text body | NPU | `text_body_s128.vmfb` (queries, images) or `text_body_s512.vmfb` (documents, video) |
| Mean pooling, Matryoshka truncation, L2 norm | CPU | |

Each image becomes `<bos> <|image> 64 x <|image|> <image|> <eos>` (68 tokens).
A video is sampled at 1 fps (at most 7 frames), each frame runs through the vision encoder,
and the frames become one block each: `<bos> 7 x (<|image> 64 x <|video|> <image|>) <eos>`
(464 tokens).

Audio becomes `<bos> <|audio> n x <|audio|> <audio|> <eos>` with n = 25 tokens per second.
A session loads one audio encoder window. By default (`--audio-window auto`) that is the smallest
window that holds the longest clip. Shorter clips are zero-padded and only their n valid tokens
are used (identical to the unpadded result); longer clips are truncated to the window.
Smaller windows are faster (board, NPU):

| Window | Audio encoder |
|---|---|
| 1.4 s | 337 ms |
| 2.8 s | 530 ms |
| 5.6 s | 1161 ms |
| 11.2 s | 1973 ms |

The text pass that follows takes 481 ms (S=128) for clips up to ~4.9 s (124 audio tokens), or
2278 ms (S=512) for longer ones: an 11 s clip takes 4.3 s end to end.

The static 384x384 / 64-token geometry is what the HF processor produces for a square image
at the smallest token budget (70). Inputs are resized to a square, so non-square images and
video frames are squashed.

## Usage

The model files come from [Synaptics/Google-EmbeddingGemma-2](https://huggingface.co/Synaptics/Google-EmbeddingGemma-2).
The default build keeps the 4-bit weights of the onnx-community `*_q4` export, about 1.1 GB.
`--model-version bf16` downloads the bf16 build instead, about 3.6 GB: closer to the original
model, but with more memory per network.

```sh
python setup_demos.py Google-EmbeddingGemma-2      # download the model files
cd Google/Google-EmbeddingGemma-2
python src/infer.py \
    --queries "cats sleeping on a couch" "a turtle swimming in the ocean" "a dog barking" \
    --images cats.jpg --videos sea-turtle.mp4 --audios dog_barking.wav
```

`-m/--model-dir` defaults to the files `setup_demos.py` downloaded.

The board's NPU driver maps at most ~1.5 GB of networks at a time, and releasing a network
within a process is not reliable on the current driver, so a session keeps all the networks it
needs mapped. The default 4-bit build fits all of them (~0.6 GB). With the bf16 build, the vision
and audio encoders plus both text buckets need ~1.7 GB. In that case `infer.py` asks you to run
`--audios` in a separate session; embeddings from both sessions live in the same space and
compare directly.

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
