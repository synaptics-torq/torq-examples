# Object Detection Demo

YOLOv8n object detection using a Torq VMFB model.

## Setup

From the repo root, run:

```sh
cd object_detection
pip install -r requirements.txt
cd ..
python setup_demos.py object_detection
```

This verifies Python dependencies for the demo and downloads the object detection assets from Hugging Face.

Downloaded assets are stored at:

```sh
models/Synaptics/yolov8-od-nano-320-int8-torq/
```

The setup downloads:
- `yolo_8n_2.0.0_npu.vmfb`
- `labels.json`
- any files present under `samples/` in the Hugging Face repo

## Running

Run the demo from the `object_detection` directory.

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
cd object_detection
python src/infer.py \
  --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
  --image ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/dog_bike_car.jpg \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io
```

To save or display the annotated image:

```sh
cd object_detection
python src/infer.py \
  --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
  --image ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/dog_bike_car.jpg \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io \
  --save-image \
  --display
```

`--tda` selects the Torq buffer allocator and lets you choose `dmabuf` (default) or `cpu`.

Image inference options:
- `--labels`: label JSON file to map class IDs to names
- `--device`: Torq device URI, defaults to `torq`
- `--tda {cpu,dmabuf}`: allocator backing Torq buffers, defaults to `dmabuf`
- `--device-io`: preallocate input buffers and keep outputs as device arrays
- `--save-image`: save the annotated output image as `output_yolo.jpg`
- `--display`: show the annotated image with GStreamer/Wayland

### Video, USB camera, or RTSP inference

```sh
cd object_detection
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
  --video ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/object_detection.mp4 \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io \
  --rotate 0
```

#### USB camera

```sh
cd object_detection
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
  --camera-device auto \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io \
  --display \
  --rotate 0
```

Example with explicit camera controls:

```sh
cd object_detection
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
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
cd object_detection
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
  --rtsp-url rtsp://user:pass@host:port/stream \
  --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
  --device torq \
  --device-io \
  --rotate 0 \
  --display
```

#### Profiling

```sh
cd object_detection
python src/infer_video.py \
  --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
  --video ../models/Synaptics/yolov8-od-nano-320-int8-torq/samples/object_detection.mp4 \
  --device torq \
  --device-io \
  --profile
```

For camera input, use `--camera-device auto` or a specific `/dev/video*` path. For video files and RTSP streams, you will usually want `--rotate 0`.

Video inference options:
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
