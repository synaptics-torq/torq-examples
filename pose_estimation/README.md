# Pose Estimation Demo

YOLOv8n pose estimation using a Torq VMFB model.

## Setup

From the repo root, run:

```sh
cd pose_estimation
pip install -r requirements.txt
cd ..
python setup_demos.py pose_estimation
```

This verifies Python dependencies for the demo and downloads the pose estimation assets from Hugging Face.

By default the demo downloads the current `latest` model revision from the HF repo. To pin a specific release tag instead,
use:

```sh
cd pose_estimation
python setup_demo.py --model-version v2.1.0
```

The `--model-version` flag accepts a Hugging Face revision or tag name, such as `latest` or a release like `v2.1.0`.

Downloaded assets are stored at:

```sh
models/Synaptics/yolov8-pose-nano-320-int8-torq/
```

The setup downloads:
- `yolo_pose.vmfb`
- any files present under `samples/` in the Hugging Face repo

## Running

Run the demo from the `pose_estimation` directory.

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

```sh
cd pose_estimation
python src/infer.py \
  --model ../models/Synaptics/yolov8-pose-nano-320-int8-torq/yolo_pose.vmfb \
  --image ../models/Synaptics/yolov8-pose-nano-320-int8-torq/samples/pose.jpg \
  --device torq \
  --device-io
```

To save or display the annotated image:

```sh
cd pose_estimation
python src/infer.py \
  --model ../models/Synaptics/yolov8-pose-nano-320-int8-torq/yolo_pose.vmfb \
  --image ../models/Synaptics/yolov8-pose-nano-320-int8-torq/samples/pose.jpg \
  --device torq \
  --device-io \
  --save-image \
  --display
```

`--tda` selects the Torq buffer allocator and lets you choose `dmabuf` (default) or `cpu`.

Image inference options:
- `--device`: Torq device URI, defaults to `torq`
- `--tda {cpu,dmabuf}`: allocator backing Torq buffers, defaults to `dmabuf`
- `--device-io`: preallocate input buffers and keep outputs as device arrays
- `--save-image`: save the annotated output image as `output_pose.jpg`
- `--display`: show the annotated image with GStreamer/Wayland

### Video, USB camera, or RTSP inference

```sh
cd pose_estimation
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-pose-nano-320-int8-torq/yolo_pose.vmfb \
  --video ../models/Synaptics/yolov8-pose-nano-320-int8-torq/samples/body_pose.mp4 \
  --device torq \
  --device-io \
  --rotate 0
```

#### USB camera

```sh
cd pose_estimation
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-pose-nano-320-int8-torq/yolo_pose.vmfb \
  --camera-device auto \
  --device torq \
  --device-io \
  --display \
  --rotate 0
```

Example with explicit camera controls:

```sh
cd pose_estimation
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-pose-nano-320-int8-torq/yolo_pose.vmfb \
  --camera-device /dev/video0 \
  --camera-control-device /dev/v4l-subdev2 \
  --device torq \
  --device-io \
  --display \
  --exposure-auto 0
```

#### RTSP stream

```sh
cd pose_estimation
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-pose-nano-320-int8-torq/yolo_pose.vmfb \
  --rtsp-url rtsp://user:pass@host:port/stream \
  --device torq \
  --device-io \
  --rotate 0 \
  --display
```

#### Profiling

```sh
cd pose_estimation
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-pose-nano-320-int8-torq/yolo_pose.vmfb \
  --video ../models/Synaptics/yolov8-pose-nano-320-int8-torq/samples/body_pose.mp4 \
  --device torq \
  --device-io \
  --profile
```

For camera input, use `--camera-device auto` or a specific `/dev/video*` path. For video files and RTSP streams, you will usually want `--rotate 0`.

Video inference options:
- `--output`: save annotated video to a file
- `--json-results`: JSON output path for detections, default `pose_results.json`
- `--display`: show annotated frames live
- `--display-sink`: GStreamer sink element, default `waylandsink`
- `--rotate {0,90,180,270}`: rotate frames before inference and display
- `--camera-width`, `--camera-height`, `--camera-fps`: configure USB camera capture
- `--camera-control-device`: V4L2 control device for camera settings
- `--brightness`, `--contrast`, `--saturation`, `--sharpness`, `--gain`, `--exposure-auto`, `--exposure-absolute`: camera controls
- `--runtime-flags ...`: extra Torq runtime flags, must be specified last
- `--profile`: print Torq resource profiling info and exit

Use `python src/infer.py -h` and `python src/infer_video.py -h` to see all options.
