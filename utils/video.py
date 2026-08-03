# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared camera, GStreamer, and JSON utilities for vision inference scripts."""

from __future__ import annotations

import json
import os
import subprocess
import threading


def configure_camera(device, controls):
    if not controls:
        return

    name_maps = {
        "brightness": ["brightness"],
        "contrast": ["contrast"],
        "saturation": ["saturation"],
        "sharpness": ["sharpness"],
        "gain": ["gain", "analogue_gain"],
        "exposure_auto": ["exposure_auto", "auto_exposure"],
        "exposure_absolute": ["exposure_absolute", "exposure"],
        "exposure_auto_priority": ["exposure_auto_priority"],
    }

    for logical_name, value in controls.items():
        if value is None:
            continue

        success = False
        errors = []
        for v4l2_name in name_maps.get(logical_name, [logical_name]):
            command = ["v4l2-ctl", "-d", device, "-c", f"{v4l2_name}={value}"]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    print(f"[Camera] Success: {' '.join(command)}")
                    success = True
                    break
                errors.append(f"{v4l2_name}: {result.stderr.strip()}")
            except Exception as exc:
                errors.append(f"{v4l2_name}: {exc}")

        if not success:
            print(f"[Camera] Warning: Failed to set {logical_name}={value} on {device}. Tried: {' | '.join(errors)}")


def find_working_camera():
    try:
        result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, timeout=2, check=False)
        if result.returncode == 0:
            in_usb_device = False
            for line in result.stdout.splitlines():
                if "usb-" in line.lower():
                    in_usb_device = True
                    continue
                if in_usb_device and "/dev/video" in line:
                    device = line.strip()
                    if os.path.exists(device):
                        return device
                if line.strip() == "":
                    in_usb_device = False
    except Exception:
        pass

    for index in range(10):
        device = f"/dev/video{index}"
        if os.path.exists(device):
            return device
    return None


def resolve_camera_device(camera_device):
    if camera_device == "auto":
        resolved = find_working_camera()
        if resolved is None:
            raise RuntimeError("No USB camera device found")
        return resolved

    if not os.path.exists(camera_device):
        raise RuntimeError(f"Camera device not found: {camera_device}")
    return camera_device


def create_display_pipeline(Gst, width, height, fps, sink_name, disp_width=None, disp_height=None):
    disp_width = disp_width or width
    disp_height = disp_height or height
    pipeline_str = (
        "appsrc name=display_src format=time is-live=true block=true ! "
        f"video/x-raw,format=BGRA,width={width},height={height},framerate={fps}/1 ! "
        "synavideoconvertscale ! "
        f"video/x-raw,width={disp_width},height={disp_height} ! "
        f"{sink_name} sync=false"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("display_src")
    return pipeline, appsrc


def push_display_frame(Gst, appsrc, frame_bgra, frame_index, fps):
    data = frame_bgra.tobytes()
    gst_buffer = Gst.Buffer.new_allocate(None, len(data), None)
    gst_buffer.fill(0, data)
    if fps > 0:
        frame_duration = Gst.SECOND // fps
        gst_buffer.pts = frame_index * frame_duration
        gst_buffer.duration = frame_duration
    return appsrc.emit("push-buffer", gst_buffer)


class RotatingJsonArrayWriter:
    def __init__(self, path, max_entries):
        self.path = path
        self.max_entries = max_entries
        self.rotated_path = self._build_rotated_path(path)
        self.file = None
        self.first_entry = True
        self.current_entries = 0
        self._open_new_file()

    @staticmethod
    def _build_rotated_path(path):
        base, ext = os.path.splitext(path)
        return f"{base}.1{ext or '.json'}"

    def _open_new_file(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")
        self.file.write("[\n")
        self.file.flush()
        self.first_entry = True
        self.current_entries = 0

    def _close_current_file(self):
        if self.file is None:
            return
        if not self.first_entry:
            self.file.write("\n")
        self.file.write("]\n")
        self.file.flush()
        self.file.close()
        self.file = None

    def _rotate(self):
        self._close_current_file()
        if os.path.exists(self.rotated_path):
            os.remove(self.rotated_path)
        if os.path.exists(self.path):
            os.replace(self.path, self.rotated_path)
        self._open_new_file()

    def append(self, record):
        if self.current_entries >= self.max_entries:
            self._rotate()

        prefix = "" if self.first_entry else ",\n"
        self.file.write(prefix)
        self.file.write(json.dumps(record, separators=(",", ":")))
        self.file.flush()
        self.first_entry = False
        self.current_entries += 1

    def close(self):
        self._close_current_file()


class FrameGrabber:
    def __init__(self, cap):
        self._cap = cap
        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                self._stopped = True
                break
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1.0)


def shutdown_display_pipeline(Gst, pipeline, appsrc):
    if pipeline is None:
        return

    if appsrc is not None:
        try:
            appsrc.emit("end-of-stream")
        except Exception:
            pass

    bus = pipeline.get_bus()
    if bus is not None:
        bus.timed_pop_filtered(Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)

    pipeline.set_state(Gst.State.READY)
    pipeline.get_state(Gst.SECOND)
    pipeline.set_state(Gst.State.NULL)
    pipeline.get_state(Gst.SECOND)


def build_video_argparser(description, default_json_results="results.json"):
    """Build the shared argparse parser for video inference scripts.

    Each script calls this, then adds any model-specific args (e.g. --labels).
    """
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True)

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--rtsp-url", help="RTSP stream URL")
    source_group.add_argument("--video", help="Path to video file")
    source_group.add_argument(
        "--camera-device",
        help="USB camera device, for example /dev/video0, or 'auto'",
    )

    parser.add_argument("--device", default="torq")
    parser.add_argument(
        "--no-refresh", action="store_true", default=False,
        help="Skip model freshness check (offline/airgapped runs)",
    )
    parser.add_argument(
        "--tda", type=str, choices=["cpu", "dmabuf"], default="dmabuf",
        help="Allocator backing Torq device buffers (default: %(default)s)",
    )
    parser.add_argument("--device-io", action="store_true",
                        help="Preallocate inputs and keep outputs as device arrays")
    parser.add_argument(
        "--runtime-flags", nargs=argparse.REMAINDER, default=None, metavar="FLAG",
        help="[Advanced] Extra flags for the Torq runtime. Must be specified last.",
    )
    parser.add_argument("--profile", action="store_true", help="Profile resource usage and exit")
    parser.add_argument("--output", default=None, help="Output video file (optional)")
    parser.add_argument("--json-results", default=default_json_results,
                        help="Output JSON file for detections")
    parser.add_argument("--camera-width", type=int, default=640, help="USB camera width")
    parser.add_argument("--camera-height", type=int, default=480, help="USB camera height")
    parser.add_argument("--camera-fps", type=int, default=30, help="USB camera frame rate")
    parser.add_argument("--display", action="store_true", help="Display annotated frames live")
    parser.add_argument("--display-sink", default="waylandsink",
                        help="GStreamer video sink for live display")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=180,
                        help="Rotate camera feed (degrees clockwise)")

    cam_group = parser.add_argument_group("Camera Config")
    cam_group.add_argument("--camera-control-device",
                           help="V4L2 device for controls (e.g. /dev/v4l-subdev2)")
    cam_group.add_argument("--brightness", type=int, help="V4L2 brightness")
    cam_group.add_argument("--contrast", type=int, help="V4L2 contrast")
    cam_group.add_argument("--saturation", type=int, help="V4L2 saturation")
    cam_group.add_argument("--sharpness", type=int, help="V4L2 sharpness")
    cam_group.add_argument("--gain", type=int, help="V4L2 gain")
    cam_group.add_argument("--exposure-auto", type=int, help="V4L2 auto exposure")
    cam_group.add_argument("--exposure-absolute", type=int, help="V4L2 absolute exposure time")

    return parser


def run_video_inference_loop(args, process_fn, ui_title):
    """Run the main video inference loop.

    Args:
        args: parsed argparse namespace from build_video_argparser
        process_fn: callable(bgr_frame) ->
            (annotated_frame, frame_detections, infer_time_ms, log_str)
        ui_title: title string shown in the on-screen display overlay
    """
    import sys
    import time
    from collections import deque
    import cv2
    from utils.draw import draw_ui, letterbox_frame

    _MAX_DETECTIONS_TO_KEEP = 60

    Gst = None
    display_pipeline = None
    display_appsrc = None
    display_fps = 0

    if args.display:
        try:
            import gi
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "--display requires PyGObject/GStreamer bindings. "
                "Install 'python3-gi' and GStreamer 1.0 introspection packages on the target."
            ) from exc
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
        # Resolve 'auto' (and validate explicit paths) here so this loop is safe
        # even when callers skip separate runtime camera setup.
        dev = resolve_camera_device(args.camera_device)
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

    all_detections = deque(maxlen=_MAX_DETECTIONS_TO_KEEP)
    json_writer = RotatingJsonArrayWriter(args.json_results, _MAX_DETECTIONS_TO_KEEP)

    grabber = None
    if args.camera_device or args.rtsp_url:
        grabber = FrameGrabber(cap)

    print(f"Processing {source_desc} with Torq Python runtime... Press Ctrl+C to stop.")
    frame_count = 0
    fps = 0.0
    fps_time = time.time()
    last_log_str = ""

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

            annotated, frame_detections, infer_time, log_str = process_fn(bgr_frame)

            if log_str != last_log_str:
                print("\n", end="", flush=True)
            else:
                print("\r" + " " * 60 + "\r", end="", flush=True)
            print(f"{frame_count} ({infer_time:.3f} ms) {log_str}", end="", flush=True)
            last_log_str = log_str

            frame_result = {"frame": frame_count, "detections": frame_detections}
            all_detections.append(frame_result)
            json_writer.append(frame_result)

            if args.display:
                assert Gst is not None
                if display_pipeline is None:
                    display_pipeline, display_appsrc = create_display_pipeline(
                        Gst, disp_w, disp_h, display_fps, args.display_sink,
                    )
                    display_pipeline.set_state(Gst.State.PLAYING)

                display_frame_bgr, video_rect = letterbox_frame(annotated, (disp_w, disp_h))
                draw_ui(display_frame_bgr, ui_title,
                        {"fps": fps, "npu": infer_time, "count": len(frame_detections)},
                        video_rect)
                rendered_frame = cv2.cvtColor(display_frame_bgr, cv2.COLOR_BGR2BGRA)
                ret = push_display_frame(Gst, display_appsrc, rendered_frame, frame_count, display_fps)
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
            f"Results saved to: {args.json_results} "
            f"(previous: {json_writer.rotated_path if os.path.exists(json_writer.rotated_path) else 'none'})"
        )
        print(f"Kept the last {len(all_detections)} detections in memory.")

