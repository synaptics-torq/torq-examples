# PP-OCR Demo

PP-OCRv6-tiny optical character recognition using Torq VMFB models: DBNet text
detection followed by CTC text recognition.

The pipeline runs in two stages. Detection finds text-line quads at one static
input shape. Recognition then reads each line, using one vmfb per width
"bucket" so a line is padded only to the narrowest width that fits it — short
labels do not pay for the widest model.

## Setup

From the repo root, run:

```sh
cd ppocr
pip install -r requirements.txt
cd ..
python setup_demos.py ppocr
```

This verifies Python dependencies for the demo and downloads the PP-OCR assets from Hugging Face.

Downloaded assets are stored at:

```sh
models/Synaptics/paddle-paddle-tiny/
```

The setup downloads:
- `ppocr_det_800x608.vmfb` — detection, static 800×608 input
- `rec_buckets/rec_w{320,640,1280,2432}.vmfb` — recognition, one per width bucket
- `ppocr_rec.yml` — recognizer character dictionary
- everything under `samples/`: `sample.jpg`, a 10-line café menu card, and
  `sample.png`, a dense 99-line paper page for a heavier run

## Running

Run the demo from the `ppocr` directory.

If you want on-device display output, set:

```sh
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
```

### Image inference

```sh
cd ppocr
python src/infer.py \
  --image ../models/Synaptics/paddle-paddle-tiny/samples/sample.jpg \
  --models ../models/Synaptics/paddle-paddle-tiny \
  --device torq
```

`--models` points at the directory holding the assets; the detection vmfb,
bucket directory and character dictionary are found inside it by name. Override
any of them individually with `--det-vmfb`, `--rec-bucket-dir` and `--rec-yml`.

To save or display the annotated image:

```sh
cd ppocr
python src/infer.py \
  --image ../models/Synaptics/paddle-paddle-tiny/samples/sample.jpg \
  --models ../models/Synaptics/paddle-paddle-tiny \
  --device torq \
  --save-image \
  --display
```

Output lists one line per recognized text box with its confidence:

```
Time: detection 519.8ms, recognition 1181.8ms (10 boxes detected)

[4/5] Text:
  1   [0.991] BLUE DOOR CAFE
  2   [0.996] all day breakfast
  3   [0.999] BREAKFAST
  4   [0.996] Avocado Toast  6.50
  ...
```

Recognition time scales with the number of detected lines, since each line is a
separate invocation — the dense `sample.png` takes ~22 s for its 99 lines.

`--tda` selects the allocator backing Torq device buffers — it does not change
where the model runs; both stages execute on the NPU either way.

> **Note:** this demo defaults to `--tda cpu`, which is also the Torq runtime's
> own default; the other demos override it to `dmabuf`. On current firmware the
> detection model fails under `dmabuf` with `INTERNAL; failed to writeXram()`.

Image inference options:
- `--device`: Torq device URI, defaults to `torq`
- `--tda {cpu,dmabuf}`: allocator backing Torq device buffers, defaults to `cpu`
- `--device-io`: preallocate input buffers and keep outputs as device arrays
- `--drop-score`: minimum recognition confidence to keep a line, defaults to `0.5`
- `--save-image`: save the annotated output image as `output_ocr.jpg`
- `--display`: show the annotated image with GStreamer/Wayland
- `--no-refresh`: skip the Hugging Face freshness check (offline/airgapped runs)

### Comparing against the CPU

Either stage can run on ONNX Runtime instead of the NPU, which is useful for
checking NPU accuracy against a CPU reference:

```sh
cd ppocr
python src/infer.py \
  --image ../models/Synaptics/paddle-paddle-tiny/samples/sample.jpg \
  --models ../models/Synaptics/paddle-paddle-tiny \
  --rec-backend ort --rec-onnx ppocr_rec_dynamic.onnx
```

- `--det-backend {npu,ort}` with `--det-onnx` for the detector
- `--rec-backend {npu,ort}` with `--rec-onnx` for the recognizer

The ONNX path needs `onnxruntime`, listed in `requirements.txt`.

### Model geometry

- `--det-hw H W`: static input the detection vmfb was compiled for, defaults to `800 608`
- `--rec-buckets W ...`: widths available as bucket vmfbs, defaults to `320 640 1280 2432`
- `--rec-vmfb` with `--rec-width`: use a single fixed-width recognizer instead of buckets

Use `python src/infer.py -h` to see all options.
