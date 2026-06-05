# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

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


def push_display_frame(Gst, appsrc, frame_rgb, frame_index, fps):
    data = frame_rgb.tobytes()
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
