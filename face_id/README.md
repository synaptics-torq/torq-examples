# Face ID Demo

Face detection on an image file using a Torq VMFB model.

This Torq example runs the detector only.
## Setup

See repo [README.md](../README.md) for installing the virtual environment and
base dependencies.

Enter the demo directory. Install its dependencies. Jump back to the repo root.

```sh
cd face_id
pip install -r requirements.txt
cd ..
```

From the repo root, run:

```sh
python setup_demos.py face_id
```

This verifies the Python dependencies and downloads the detector model from
Hugging Face.

Downloaded assets are stored at:

```text
models/Synaptics/face-id-torq/
```

The Torq example downloads only:

- `face_detection.vmfb`
- `face.jpg`


## Running

Run the demo from the `face_id` directory.

```sh
cd face_id
```

### Image inference

```sh
python src/infer.py \
  --model ../models/Synaptics/face-id-torq/face_detection.vmfb \
  --image ../models/Synaptics/face-id-torq/face.jpg \
  --device torq
```

The command writes an annotated image to `face_detection.jpg` and detection
data to `face_detection_results.json`.

The JSON bounding box uses pixel coordinates in the original image, in this
format:

```text
[x, y, width, height]
```

where `x` and `y` are the top-left corner of the face.

### Options

- `--model`: path to `face_detection.vmfb` (default: the one setup_demo.py downloaded)
- `--image`: input image path (required)
- `--output`: annotated output image, default `face_detection.jpg`
- `--json-results`: detection JSON output, default `face_detection_results.json`
- `--device`: Torq device URI, default `torq`
- `--tda {cpu,dmabuf}`: allocator backing Torq buffers, default `dmabuf`
- `--device-io` / `--no-device-io`: allocate inputs and keep outputs device-backed (default: enabled; forced with `--tda dmabuf`)
- `--runtime-flags ...`: extra Torq runtime flags, must be specified last
- `--confidence-threshold`: minimum detection confidence, default `0.6`

Use `python src/infer.py -h` to see all options.

## Validation

`src/validate.py` runs the face detector on 50 photos of Labeled Faces in the Wild ([bitmind/lfw](https://huggingface.co/datasets/bitmind/lfw)), crops centred on one person's face; a photo passes when a detected face box contains its centre:

```sh
cd face_id
python src/validate.py
```

The rows are downloaded from Hugging Face on the first run and cached under `$MODELS/datasets/` (copy that folder to a board without network access). The data is not part of this repository: check the dataset card for its license and terms of use. `--dataset`, `--config`, `--split` and `--samples` select other data (the defaults are the `validation` list in `info.json`).

The script prints each item's result and the overall accuracy; `--metrics-out` records it as the `accuracy` metric next to the performance metrics of `src/infer.py`. There is no pass mark: the score is there so you can weigh accuracy against speed (e.g. between model variants). Run `python src/validate.py -h` to see all available validation options.
