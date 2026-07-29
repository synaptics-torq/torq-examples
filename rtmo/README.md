# RTMO Pose Demo

RTMO tiny multi-person pose estimation on Torq. Runs a **hybrid** model — an
int8 conv backbone, a **bf16** AIFI transformer neck, and an int8 detection head,
compiled to three NSS-only VMFBs and chained on the NPU. The bf16 transformer
keeps int8 speed (~56 ms on the SL2619 board) while removing the full-int8 false
positives, matching the fp32 detections. The decode/NMS + DCC pose classifier run
host-side.

## Setup

From the repo root, run:

```sh
cd rtmo
pip install -r requirements.txt
cd ..
python setup_demos.py rtmo
```

This verifies the demo's Python dependencies and downloads the assets from
Hugging Face ([`Synaptics/RTMO_pose`](https://huggingface.co/Synaptics/RTMO_pose)).

Downloaded assets are stored at:

```sh
models/Synaptics/RTMO_pose/
├── vmfb/rtmo_hyb_backbone_int8.vmfb
├── vmfb/rtmo_hyb_transformer_bf16.vmfb
├── vmfb/rtmo_hyb_head_int8.vmfb
└── calib/{person,people}.jpg
```

The host-side DCC pose-decode weights ship with the demo
(`rtmo_core/postprocess_weights.npz`), so only the VMFBs and sample images
download.

## Running

Run the demo from the `rtmo` directory. The first run downloads the assets
automatically; pass `--no-refresh` for fully offline runs afterwards.

```sh
cd rtmo

# default: the person.jpg sample -> person_rtmo.jpg (one person)
python src/infer.py

# the multi-person sample -> people_rtmo.jpg (five people)
python src/infer.py --image ../models/Synaptics/RTMO_pose/calib/people.jpg

# any image
python src/infer.py --image /path/to/photo.jpg --output out.jpg
```

Each run prints the per-part timing (backbone / transformer / head), the number
of people drawn, and the annotated output image path. Boxes are drawn in green
with a `person <score>` label; the 17-keypoint COCO skeleton is overlaid in blue.

Options:

- `--model-dir DIR` — asset dir (default `models/Synaptics/RTMO_pose`).
- `--image PATH` — input image (default the downloaded `calib/person.jpg`).
- `--output PATH` — annotated output (default `<input-stem>_rtmo.jpg`).
- `--device URI` — IREE device (default `torq`).
- `--no-refresh` — skip the Hugging Face update check (offline).

## How it runs

`HybridRunner` ([`rtmo_core/hybrid.py`](./rtmo_core/hybrid.py)) chains the three
VMFBs, requantizing at the seams exactly as on-device:

```
image(int8) -> [backbone] -> P3, P4, P5 (int8)
                              P5 -{dequant}-> bf16 -> [transformer] -> P5' (bf16)
P3, P4 -{requant to head scales}-\
                                  P5' -{NHWC, requant}--+-> [head] -> 8 heads (int8)
```

The P3/P4 FPN skip connections pass straight through (identical
backbone-out/head-in scales); only the neck-transformed P5 is requantized. The
eight int8 head tensors are dequantized and handed to
[`rtmo_core/postprocess.py`](./rtmo_core/postprocess.py) for the NMS + DCC/SimCC
pose decode.
