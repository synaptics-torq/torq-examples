# EmbeddingGemma-2 on Torq for text, image, video and audio

[EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2) maps text, images, video and audio into one 768-dimensional embedding space. This demo runs the text backbone, vision encoder and audio encoder on the SL2619 NPU, about 740M parameters in total.

## How it runs

| Stage | Where | Notes |
|---|---|---|
| Tokenizer, prompt prefix | CPU | `tokenizer.json`; task prefixes such as `task: search result \| query: ` |
| Token-embedding lookup | CPU | `token_embeddings.npy`: 262144 x 512 bf16, memory-mapped from eMMC |
| Image / frame preprocessing | CPU | resize to 384x384, cut 576 patches of 16x16 |
| Vision encoder | NPU | `vision_384x384.vmfb`: 576 patches -> 64 soft tokens of 512 dimensions |
| Audio front end | CPU | 16 kHz mono, 128-bin log-mel using `mel_filters.npy` and `mel_window.npy`, padded to the encoder window |
| Audio encoder | NPU | `audio_<frames>.vmfb` for 1.4 / 2.8 / 5.6 / 11.2 s windows of 140 / 280 / 560 / 1120 frames -> up to 35 / 70 / 140 / 280 soft tokens of 512 dimensions, one per 40 ms |
| Text body | NPU | `text_body_s128.vmfb` or `text_body_s512.vmfb`: the whole token sequence, text and soft tokens, -> 768-d per token |
| Mean pooling, Matryoshka truncation, L2 norm | CPU | |

The encoders do not produce the embedding themselves. Each encoder includes a 512-dimensional projection into the text model's embedding space. Its soft tokens go through the text body like words:

1. The host builds the token sequence, for example `<bos> <|image> 64 x <|image|> <image|> <eos>` for an image, and looks up every token in `token_embeddings.npy`.
2. The encoder's soft tokens replace the `<|image|>`, `<|video|>` or `<|audio|>` placeholder rows, in order. The surrounding marker tokens keep their text embeddings.
3. The sequence is right-padded to the text bucket and runs through the text body's 24 bidirectional layers with a padding mask.
4. The host takes the mean over the real tokens, optionally truncates it to 512 / 256 / 128 dimensions, and L2-normalises it.

So every image, video or sound costs one encoder pass and one text pass, and its embedding is directly comparable with a text query's.

Each image becomes `<bos> <|image> 64 x <|image|> <image|> <eos>`, a total of 68 tokens. A video is sampled at 1 fps with at most 7 frames. Each frame runs through the vision encoder and becomes one block in `<bos> 7 x (<|image> 64 x <|video|> <image|>) <eos>`, a total of 464 tokens.

Audio becomes `<bos> <|audio> n x <|audio|> <audio|> <eos>` with n = 25 tokens per second. A session loads one audio encoder window. The default `--audio-window auto` selects the smallest window that holds the longest clip. Shorter clips are zero-padded and only their n valid tokens are used, giving the same result as an unpadded clip; longer clips are truncated to the window. Smaller windows are faster, as shown in the timings below.

The static 384x384 / 64-token geometry is what the HF processor produces for a square image at the smallest token budget of 70. Inputs are resized to a square, so non-square images and video frames are squashed.

### Text body: two sequence lengths

The text body is compiled twice from the same weights, for 128 and for 512 tokens. Each input runs in the smallest one that holds its token sequence; shorter sequences are padded and masked, so the embedding does not depend on the bucket:

| Input | Tokens | Text body |
|---|---|---|
| Text query or short document | up to 128 | S=128 |
| Longer document | 129-512 | S=512 |
| Image | 68 | S=128 |
| Audio up to ~4.9 s | n + 4 <= 128, with n = 25 per second | S=128 |
| Audio from ~4.9 s to 11.2 s, up to 284 tokens | up to 512 | S=512 |
| Video: up to 7 frames x 66 + 2 = 464 tokens | up to 512 | S=512 |

- S=128 is about 5.6x faster than S=512, and most queries and every image fit in it.
- 512 is the largest bucket because the model's sliding-window layers reach 512 tokens: up to 513 tokens every layer is plain full attention. Longer inputs are rejected, not truncated.
- Each bucket holds its own copy of the weights, 165 MB + 186 MB.

## Inference time on the SL2610 series

Measured on an SL2619 board: time per call after a warm-up run, one network per process. The NPU runs the vmfbs; ONNX Runtime 1.27 runs the same 8-bit weights using `MatMulNBits` on the board's 2 CPU cores.

| Network | NPU | CPU with ONNX Runtime | NPU speed-up | NPU RAM / file |
|---|---|---|---|---|
| text S=128 | 389 ms | 4774 ms | 12.3x | 198 / 165 MB |
| text S=512 | 2185 ms | 17546 ms | 8.0x | 220 / 186 MB |
| vision 384x384 per image or frame | 2734 ms | 18612 ms | 6.8x | 330 / 303 MB |
| audio 1.4 s window | 302 ms | 5793 ms | 19.2x | 352 / 328 MB |
| audio 2.8 s window | 463 ms | 7859 ms | 17.0x | 355 / 331 MB |
| audio 5.6 s window | 935 ms | 12067 ms | 12.9x | 361 / 337 MB |
| audio 11.2 s window | 2041 ms | 20613 ms | 10.1x | 381 / 357 MB |

End-to-end timings per input include the encoder and text body. Host preprocessing adds about 50 ms per image, 25 ms per sound, and 1-2 seconds to decode a video:

| Input | Networks | NPU | CPU with ONNX Runtime |
|---|---|---|---|
| Text query | text S=128 | 0.39 s | 4.8 s |
| Long document | text S=512 | 2.2 s | 17.5 s |
| Image | vision + text S=128 | 3.1 s | 23.4 s |
| Video, 7 frames | 7 x vision + text S=512 | 21.3 s | 148 s |
| Sound up to 1.4 s | audio 1.4 s + text S=128 | 0.69 s | 10.6 s |
| Sound up to 2.8 s | audio 2.8 s + text S=128 | 0.85 s | 12.6 s |
| Sound up to 4.9 s | audio 5.6 s + text S=128 | 1.3 s | 16.8 s |
| Sound 4.9-5.6 s | audio 5.6 s + text S=512 | 3.1 s | 29.6 s |
| Sound 5.6-11.2 s | audio 11.2 s + text S=512 | 4.2 s | 38.2 s |

The NPU timings are the same when all the networks run in one session.

## Usage

The model files come from [Synaptics/Google-EmbeddingGemma-2](https://huggingface.co/Synaptics/Google-EmbeddingGemma-2). They have int8 weights in blocks of 32 and take about 2.3 GB.

```sh
python setup_demos.py EmbeddingGemma-2      # download the model files and the example media
cd EmbeddingGemma-2
python src/media_search.py index            # index the example media on the NPU (~2 min)
python src/media_search.py search startup   # load the model once (~10 s), then type queries
```

`-m/--model-dir` defaults to the files `setup_demos.py` downloaded.

### Media search: one index for text, images, video and sound

`src/media_search.py index` embeds every image, sound clip and video of a media folder once, and every second of each video as a "moment", and saves the embeddings with their file names and times to `media_index.npz`. `search` then embeds only the query and ranks the index with one matrix product, so a result can also point to a time in a video.

`search startup` starts an interactive prompt. Loading the index and text model takes about 10 seconds, mostly to start the two text networks on the NPU. Each typed query then takes about 0.4 seconds. Type `q` or press Ctrl-D or Ctrl-C to quit. Before the process exits, the networks are unmapped from the NPU and memory use falls from about 660 MB to 50 MB. `search --text "..."`, `--image` or `--audio` answers one query and exits, taking about 12 seconds in total because each invocation loads the model.

By default `index` reads the 14 example files that `setup_demos.py` downloads into `<model dir>/examples`: a photo and sound for each of six subjects, cat, dog, bird, piano, train and police car, plus short videos of a waterfall and a sea turtle. They are CC0 or public domain; credits are in `examples/README.md` next to them. `--media-dir <folder>` indexes your own media instead.

```
$ python src/media_search.py search startup
[search] 25 entries from media_index.npz, model ready in 10 s. Type a query; 'q', Ctrl-D or Ctrl-C to quit.

search> a dog barking
query: text 'a dog barking'  (390 ms)

  images:
     0.696  dog_labrador.jpg
     0.611  bird_sparrow.jpg
     0.582  piano_keyboard.jpg
  sounds:
     0.784  dog_barking.wav
     0.675  police_siren.wav
     0.673  cat_meowing.wav
  videos:
     0.589  waterfall.ogv
     0.584  sea_turtle.ogv
  ...

search> q
[search] model released
```

- **Text queries** search everything: "a piano" finds `piano_keyboard.jpg` with a score of 0.748 and `piano_arpeggio.wav` with 0.668; "a turtle swimming in the ocean" finds `sea_turtle.ogv` from its first second with a score of 0.767.
- NPU checks show each example ranks first for a description of its own subject, well ahead of the other subjects: the dog's bark matches "a dog barking and growling", not "a cat meowing loudly".
- **An image query** with `--image photo.jpg` finds similar images and video moments. It is most useful on your own library, where there are several photos of a kind.
- Matching a sound directly against photos is not reliable: those scores are close to noise, so the default results stay within the query's own media type. `--kinds` shows the others.
- Indexing the 14 examples takes 2 minutes on the board, mostly in the vision encoder at 2.7 seconds per photo or video second. Each modality group is indexed in its own process.

### Embed your own files

`src/infer.py` embeds text, images, videos and sounds given on the command line and lists, for each query, the best matches first. With some of the example files:

```sh
E=../models/Synaptics/Google-EmbeddingGemma-2/examples
python src/infer.py --queries "a dog barking and growling" "a cat" \
    --images $E/dog_labrador.jpg $E/cat_tabby.jpg --audios $E/dog_barking.wav $E/cat_meowing.wav
```

```
query: 'a dog barking and growling'
   0.790  audio:dog_barking.wav
   0.681  audio:cat_meowing.wav
   0.675  image:dog_labrador.jpg
   0.557  image:cat_tabby.jpg
```

Options include `--videos`, `--documents` for text documents, `--dim 256` for Matryoshka truncation with renormalisation, and `--top N`. Use `--backend ort` to run the ONNX graphs of the same networks on the CPU with the same host code.

### Backend comparison

```sh
python src/compare_backends.py run -m <vmfb_dir> --backend torq -o torq.npz
python src/compare_backends.py run -m <onnx_dir> --backend ort --ort-threads 2 -o ort.npz
python src/compare_backends.py report torq.npz ort.npz -o report.md
# audio: separate sessions as well
python src/compare_backends.py run -m <vmfb_dir> --backend torq --modalities audio -o torq_audio.npz
```

The eval set in `data/eval/eval_set.json` at the repository root has 53 texts, 20 images, 5 videos and 10 audio clips. Pass `--audio` to `run` to include the audio clips. Media files are downloaded from `Xenova/transformers.js-docs` on first use.
