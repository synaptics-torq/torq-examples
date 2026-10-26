# Object Detection Demo

YOLOv8n and the nano and small variants of YOLO26 object detection using Torq VMFB models.

## Setup

See repo [README.md](../README.md) for installing the virtual environment and base dependencies.

Enter the demo directory. Install its dependencies. Jump back to the repo root.

```sh
cd object_detection
pip install -r requirements.txt
cd ..
```

From the repo root, run:

```sh
python setup_demos.py object_detection
```

This verifies Python dependencies for the demo and downloads the object detection assets from Hugging Face.

By default the demo downloads the model version matching this repo's `VERSION` file. The [repo README](../README.md) describes the versioning scheme. To pin a specific release tag instead, use:

```sh
cd object_detection
python setup_demo.py --model-version v2.2.0
```

A pinned version is kept in sync with its own tag but never upgraded. `--no-update` skips model tracking and does not write a `.manifest.json`, at your own risk.

The YOLO26 repo has no version tags yet, so its models are downloaded at the repo's latest HEAD revision and left untracked: re-runs only repair missing files rather than following a tag.

Downloaded YOLOv8n assets are stored at `models/Synaptics/yolov8-od-nano-320-int8-torq/`. YOLO26n and YOLO26s assets are stored at `models/Synaptics/yolov26n_od/`.

The setup downloads, for each model:

- the model VMFBs: `yolo_od.vmfb`, or `yolo26n_npu.vmfb` and `yolo26s_npu.vmfb`
- `labels.json`
- sample media from `samples/` in the Hugging Face repo or from its root, such as `bus.jpg`

## Running

Run the demo from the `object_detection` directory.

```sh
cd object_detection
```

If you want on-device display output, set:

```sh
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
```

For video or camera display with `--display`, the target also needs PyGObject/GStreamer Python bindings, typically via the system package `python3-gi` and the corresponding GStreamer introspection packages.

For portrait displays, you can also set:

```sh
export ORIENTATION=portrait
export DISPLAY_HEIGHT=800
export DISPLAY_WIDTH=480
```

### Image inference

`--model` is optional: it defaults to the model file setup downloaded for `--variant`. The default is the YOLOv8n model.

```sh
python src/infer.py \
  --image ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/dog_bike_car.jpg \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io
```

To save or display the annotated image:

```sh
python src/infer.py \
  --image ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/dog_bike_car.jpg \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io \
  --save-image \
  --display
```

For YOLO26, pass `--variant yolo26n`, or `--variant yolo26s` with `yolo26s_npu.vmfb`, so the NMS-free head is decoded with the right quantization parameters:

```sh
python src/infer.py \
  --image ../models/Synaptics/yolov26n_od/samples/soccer.jpg \
  --labels ../models/Synaptics/yolov26n_od/labels.json \
  --variant yolo26n \
  --device torq \
  --device-io
```

On `samples/soccer.jpg` both sizes detect all five players. YOLO26s also picks up the sports ball.

Measured on an SL2610 board with 320x320 int8 input and 10 repetitions of `iree-benchmark-module`: YOLO26n ~13 ms, YOLO26s ~33 ms per inference.

`--tda` selects the Torq buffer allocator and lets you choose `dmabuf` or `cpu`. The default is `dmabuf`.

Image inference options:

- `--model`: model VMFB path, defaults to the one setup_demo.py downloaded for `--variant`
- `--variant {yolo26n,yolo26s,yolov8}`: model head variant, defaults to `yolov8`
- `--labels`: label JSON file to map class IDs to names
- `--device`: Torq device URI, defaults to `torq`
- `--tda {cpu,dmabuf}`: allocator backing Torq buffers, defaults to `dmabuf`
- `--device-io`: preallocate input buffers and keep outputs as device arrays
- `--save-image`: save the annotated output image as `output_yolo.jpg`
- `--display`: show the annotated image with GStreamer/Wayland

### Video, USB camera, or RTSP inference

```sh
python src/infer_video.py \
  --video ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/object_detection.mp4 \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io \
  --rotate 0
```

#### USB camera

```sh
python src/infer_video.py \
  --camera-device auto \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io \
  --display \
  --rotate 0
```

Example with explicit camera controls:

```sh
python src/infer_video.py \
  --camera-device /dev/video0 \
  --camera-control-device /dev/v4l-subdev2 \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io \
  --display \
  --exposure-auto 0
```

#### RTSP stream

```sh
python src/infer_video.py \
  --rtsp-url rtsp://user:pass@host:port/stream \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io \
  --rotate 0 \
  --display
```

#### Profiling

```sh
python src/infer_video.py \
  --video ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/object_detection.mp4 \
  --device torq \
  --device-io \
  --profile
```

For camera input, use `--camera-device auto` or a specific `/dev/video*` path. For video files and RTSP streams, you will usually want `--rotate 0`.

Video inference options:

- `--model`: model VMFB path, defaults to the one setup_demo.py downloaded for `--variant`
- `--variant {yolo26n,yolo26s,yolov8}`: model head variant, defaults to `yolov8`
- `--output`: save annotated video to a file
- `--json-results`: JSON output path for detections, default `detection_results.json`
- `--display`: show annotated frames live
- `--display-sink`: GStreamer sink element, default `waylandsink`
- `--rotate {0,90,180,270}`: rotate frames before inference and display
- `--camera-width`, `--camera-height`, `--camera-fps`: configure USB camera capture
- `--camera-control-device`: V4L2 control device for camera settings
- `--brightness`, `--contrast`, `--saturation`, `--sharpness`, `--gain`, `--exposure-auto`, `--exposure-absolute`: camera controls
- `--runtime-flags ...`: extra Torq runtime flags, must be specified last
- `--profile`: print Torq resource profiling info and exit

Use `python src/infer.py -h` and `python src/infer_video.py -h` to see all options.
