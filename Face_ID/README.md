# Face ID Demo

Face detection on an image file using a Torq VMFB model.

This Torq example runs the detector only.
## Setup

See repo [README.md](../README.md) for installing the virtual environment and
base dependencies.

Enter the demo directory. Install its dependencies. Jump back to the repo root.

```sh
cd Face_ID
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

Run the demo from the `Face_ID` directory.

```sh
cd Face_ID
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

- `--model`: path to `face_detection.vmfb` (required)
- `--image`: input image path (required)
- `--output`: annotated output image, default `face_detection.jpg`
- `--json-results`: detection JSON output, default `face_detection_results.json`
- `--device`: Torq device URI, default `torq`
- `--tda {cpu,dmabuf}`: allocator backing Torq buffers, default `dmabuf`
- `--device-io`: allocate inputs and keep outputs device-backed; enabled automatically with `--tda dmabuf`
- `--confidence-threshold`: minimum detection confidence, default `0.6`

Use `python src/infer.py -h` to see all options.


