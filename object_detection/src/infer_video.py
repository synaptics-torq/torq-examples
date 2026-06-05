# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time

from object_detection.setup_demo import ensure_object_detection_models
from utils.npu import enable_npu_clock
from torq.runtime import profile_vmfb_resources
from utils.inference import SimpleVMFBInferenceRunner
from utils.object_detection import (
    FrameGrabber,
    RotatingJsonArrayWriter,
    annotate_frame,
    build_runtime_flags,
    configure_camera,
    create_display_pipeline,
    dequantize_out,
    draw_ui,
    letterbox_frame,
    postprocess,
    preprocess_frame_cv,
    push_display_frame,
    resolve_camera_device,
    shutdown_display_pipeline,
)

MAX_DETECTIONS_TO_KEEP = 60


def run_with_opencv(args, runner, labels):
    import cv2

    Gst = None
    display_pipeline = None
    display_appsrc = None
    display_fps = 0
    if args.display:
        try:
            import gi
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "--display requires PyGObject/GStreamer bindings. Install the system package 'python3-gi' "
                "and the GStreamer introspection packages for Gst 1.0 on the target."
            ) from e

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst as _Gst

        _Gst.init(None)
        Gst = _Gst

    if args.display and Gst is None:
        raise RuntimeError("Failed to initialize GStreamer display")

    if args.rtsp_url:
        cap = cv2.VideoCapture(args.rtsp_url)
        source_desc = args.rtsp_url
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        source_desc = args.video
    else:
        dev = args.camera_device
        try:
            cam_index = int(dev)
        except (ValueError, TypeError):
            cam_index = dev
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if args.camera_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        source_desc = dev

    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source_desc}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.camera_fps or 15
    out_fps = int(src_fps) if src_fps > 0 else 15
    display_fps = out_fps if out_fps > 0 else 15

    orientation = os.environ.get("ORIENTATION", "landscape")
    disp_w, disp_h = (480, 800) if orientation == "portrait" else (800, 480)

    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, out_fps, (width, height))

    all_detections = deque(maxlen=MAX_DETECTIONS_TO_KEEP)
    last_detection_labels = []
    json_writer = RotatingJsonArrayWriter(args.json_results, MAX_DETECTIONS_TO_KEEP)

    grabber = None
    if args.camera_device or args.rtsp_url:
        grabber = FrameGrabber(cap)

    print(f"Processing {source_desc} with Torq Python runtime... Press Ctrl+C to stop.")
    frame_count = 0
    fps = 0.0
    fps_time = time.time()

    try:
        while True:
            if grabber is not None:
                while True:
                    ret, bgr_frame = grabber.read()
                    if ret:
                        break
                    time.sleep(0.005)
            else:
                ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                break

            if args.rotate == 90:
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotate == 180:
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)
            elif args.rotate == 270:
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            input_data, pad_info, orig_shape = preprocess_frame_cv(bgr_frame)

            raw_out = runner.infer(input_data)

            infer_time = runner.infer_time_ms

            out_scale = 0.004194467328488827
            out_zp = -128
            outputs = dequantize_out(raw_out, out_scale, out_zp, int8=True)
            detections = postprocess(outputs, orig_shape, pad_info, labels)

            detection_labels = [d[0] for d in detections]
            objects_changed = set(detection_labels) != set(last_detection_labels)

            if objects_changed:
                print("\n", end="", flush=True)
            else:
                print("\r" + " " * 50 + "\r", end="", flush=True)
            print(f"{frame_count} ({infer_time:.3f} ms)", end="", flush=True)
            for label, conf, box in detections:
                print(f" {label} {conf:.2f}", end="", flush=True)
            last_detection_labels = detection_labels

            annotated, frame_detections = annotate_frame(bgr_frame, detections)

            frame_result = {"frame": frame_count, "detections": frame_detections}
            all_detections.append(frame_result)
            json_writer.append(frame_result)

            if args.display:
                assert Gst is not None
                if display_pipeline is None:
                    display_pipeline, display_appsrc = create_display_pipeline(
                        Gst,
                        disp_w,
                        disp_h,
                        display_fps,
                        args.display_sink,
                    )
                    display_pipeline.set_state(Gst.State.PLAYING)

                display_frame_bgr, video_rect = letterbox_frame(annotated, (disp_w, disp_h))
                ui_stats = {
                    "fps": fps,
                    "npu": infer_time,
                    "count": len(detections),
                }
                draw_ui(display_frame_bgr, "Object Detection", ui_stats, video_rect)

                rendered_frame = cv2.cvtColor(display_frame_bgr, cv2.COLOR_BGR2BGRA)
                ret = push_display_frame(
                    Gst,
                    display_appsrc,
                    rendered_frame,
                    frame_count,
                    display_fps,
                )
                if ret != Gst.FlowReturn.OK:
                    print(f"Warning: failed to display frame: {ret}")

            if out_writer is not None:
                out_writer.write(annotated)

            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - fps_time)
                fps_time = now

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if grabber is not None:
            grabber.stop()
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if args.display:
            assert Gst is not None
            shutdown_display_pipeline(Gst, display_pipeline, display_appsrc)
        json_writer.close()
        print(f"Done. Processed {frame_count} frames. Output: {args.output if args.output else 'not saved'}")
        print(
            f"Detection results saved to: {args.json_results} "
            f"(previous file: {json_writer.rotated_path if os.path.exists(json_writer.rotated_path) else 'none'})"
        )
        print(f"Kept the last {len(all_detections)} detections in memory.")


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on video, RTSP, or camera input.")
    parser.add_argument("--model", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--rtsp-url", help="RTSP stream URL")
    source_group.add_argument("--video", help="Path to video file")
    source_group.add_argument(
        "--camera-device",
        help="USB camera device, for example /dev/video0, or 'auto'",
    )
    parser.add_argument("--labels")
    parser.add_argument("--device", default="torq")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        default=False,
        help="Skip the Hugging Face check for updated models (offline/airgapped runs)",
    )
    parser.add_argument(
        "--tda", type=str, choices=["cpu", "dmabuf"], default="cpu",
        help="Allocator backing Torq device buffers (default: %(default)s)",
    )
    parser.add_argument(
        "--device-io", action="store_true",
        help="Preallocate inputs and keep outputs as device arrays",
    )
    parser.add_argument(
        "--runtime-flags",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help="[Advanced] Extra flags for the Torq runtime. Must be specified last; all remaining arguments are forwarded.",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Profile resource usage and exit",
    )
    parser.add_argument("--output", default=None, help="Output video file (optional)")
    parser.add_argument("--json-results", default="detection_results.json", help="Output JSON file for detections")
    parser.add_argument("--camera-width", type=int, default=640, help="USB camera width")
    parser.add_argument("--camera-height", type=int, default=480, help="USB camera height")
    parser.add_argument("--camera-fps", type=int, default=30, help="USB camera frame rate")
    parser.add_argument("--display", action="store_true", help="Display annotated frames live")
    parser.add_argument("--display-sink", default="waylandsink", help="GStreamer video sink for live display")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=180, help="Rotate camera feed (degrees clockwise)")

    cam_group = parser.add_argument_group("Camera Config")
    cam_group.add_argument("--camera-control-device", help="V4L2 device for controls (e.g. /dev/v4l-subdev2)")
    cam_group.add_argument("--brightness", type=int, help="V4L2 brightness")
    cam_group.add_argument("--contrast", type=int, help="V4L2 contrast")
    cam_group.add_argument("--saturation", type=int, help="V4L2 saturation")
    cam_group.add_argument("--sharpness", type=int, help="V4L2 sharpness")
    cam_group.add_argument("--gain", type=int, help="V4L2 gain")
    cam_group.add_argument("--exposure-auto", type=int, help="V4L2 auto exposure")
    cam_group.add_argument("--exposure-absolute", type=int, help="V4L2 absolute exposure time")

    args = parser.parse_args()

    ensure_object_detection_models(Path(args.model).parent, refresh=not args.no_refresh)

    runtime_flags = build_runtime_flags(args.tda, args.runtime_flags)

    if args.profile:
        results = profile_vmfb_resources(
            args.model,
            device=args.device,
            n_iters=1,
            n_threads=None,
            function="main",
            runtime_flags=runtime_flags,
            device_io=args.device_io,
            do_warmup=True,
        )
        print(results.summary())
        return

    ok, message = enable_npu_clock()
    print(f"[NPU] {message}")

    if args.display:
        os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
        os.environ["WAYLAND_DISPLAY"] = "wayland-1"

    if args.camera_device and args.camera_device != "auto":
        ctrl_device = args.camera_control_device or args.camera_device
        cam_ctrls = {
            "brightness": args.brightness,
            "contrast": args.contrast,
            "saturation": args.saturation,
            "sharpness": args.sharpness,
            "gain": args.gain,
            "exposure_auto": args.exposure_auto,
            "exposure_absolute": args.exposure_absolute,
        }
        configure_camera(ctrl_device, cam_ctrls)

    runner = SimpleVMFBInferenceRunner(
        args.model,
        device_uri=args.device,
        runtime_flags=runtime_flags,
        device_io=args.device_io,
    )
    if args.camera_device:
        try:
            args.camera_device = resolve_camera_device(args.camera_device)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    labels = {}
    if args.labels:
        with open(args.labels, encoding="utf-8") as handle:
            data = json.load(handle)
            if "names" in data:
                labels = {str(k): v for k, v in data["names"].items()}
            else:
                labels = data

    run_with_opencv(args, runner, labels)


if __name__ == "__main__":
    main()
