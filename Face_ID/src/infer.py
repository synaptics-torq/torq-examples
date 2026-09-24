# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Stage 1 face detection VMFB smoke test using a USB camera."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torq.runtime import VMFBInferenceRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.npu import configure_npu_userspace_frequency, enable_npu_clock
from utils.object_detection.draw import annotate_frame
from utils.download import download_from_hf
from utils.download import default_models_dir
from utils.runtime import build_runtime_flags, cleanup_npu_after_inference
from utils.video import run_video_inference_loop

from Face_ID.src.postprocess import decode_face_outputs
from Face_ID.src.keypoints import (
    decode_keypoints,
    map_keypoints_to_frame,
    preprocess_keypoint_roi,
)
from Face_ID.src.embedding import decode_embedding, preprocess_embedding
from Face_ID.src.recognition import FaceDatabase, RecognitionTracker
from Face_ID.src.box_smoother import BoxSmoother
from Face_ID.src.qt_ui import QtEnrollmentUI
from Face_ID.src.batch_enrollment import BatchEnrollmentSession


MODEL_WIDTH = 1280
MODEL_HEIGHT = 704
FACE_DETECTION_DIR = Path(__file__).resolve().parents[1]
FACE_DATABASE_PATH = FACE_DETECTION_DIR / "face_embeddings.json"
FACE_MODEL_REPO = "Synaptics/face-id-torq"
FACE_MODEL_DIR = default_models_dir() / "Synaptics" / "face-id-torq"
FACE_MODEL_FILENAMES = (
    "face_detection.vmfb",
    "face_keypoint_static.vmfb",
    "face_embeddings_static.vmfb",
)


def ensure_face_models() -> dict[str, Path]:
    """Download the Face ID VMFB files from Hugging Face when missing."""
    FACE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename in FACE_MODEL_FILENAMES:
        path = FACE_MODEL_DIR / filename
        if not path.exists():
            print(f"Downloading {filename} from {FACE_MODEL_REPO}...")
            download_from_hf(FACE_MODEL_REPO, filename, base_dir=default_models_dir())
        paths[filename] = path
    return paths


@dataclass(frozen=True)
class AppConfig:
    """Fixed configuration for the board-side Qt application."""

    model: str = str(FACE_MODEL_DIR / "face_detection.vmfb")
    keypoint_model: str = str(FACE_MODEL_DIR / "face_keypoint_static.vmfb")
    embedding_model: str = str(FACE_MODEL_DIR / "face_embeddings_static.vmfb")
    face_database: str = str(FACE_DATABASE_PATH)
    camera_device: str = "/dev/video0"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    rtsp_url: str | None = None
    video: str | None = None
    tda: str = "cpu"
    device: str = "torq"
    runtime_flags: list[str] | None = None
    confidence_threshold: float = 0.6
    iou_threshold: float = 0.4
    box1_scale: float = 6.7147956
    box1_zero_point: int = -61
    box2_scale: float = 6.6746836
    box2_zero_point: int = -128
    score_scale: float = 1.0 / 256.0
    score_zero_point: int = -128
    keypoint_input_zero_point: int = -128
    keypoint_output_scale: float = 0.2108401
    keypoint_output_zero_point: int = -128
    embedding_output_scale: float = 0.1855205
    embedding_output_zero_point: int = -2
    match_threshold: float = 0.50
    match_hysteresis: float = 0.15
    match_margin: float = 0.06
    switch_confirm_frames: int = 2
    acquire_confirm_frames: int = 4
    recognition_debug: bool = os.environ.get("FACE_ID_RECOGNITION_DEBUG", "0") == "1"
    enroll_frames: int = 10
    recognition_window: int = 5
    recognition_hold_frames: int = 16
    display: bool = False
    raw_yuyv: bool = True
    rotate: int = 0
    display_sink: str = "waylandsink"
    output: str | None = None
    json_results: str = "face_detection_results.json"


def preprocess_frame(
    bgr_frame: np.ndarray,
    raw_yuyv: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int, float]]:
    """Letterbox camera frame to model input, preserving aspect ratio."""
    gray_frame = raw_yuyv[:, :, 0] if raw_yuyv is not None else cv2.cvtColor(
        bgr_frame, cv2.COLOR_BGR2GRAY,
    )
    source_height, source_width = gray_frame.shape

    # Calculate scale to fit camera frame into model dimensions while preserving aspect ratio
    scale = min(MODEL_WIDTH / source_width, MODEL_HEIGHT / source_height)
    new_width = int(source_width * scale)
    new_height = int(source_height * scale)

    # Resize preserving aspect ratio
    resized = cv2.resize(gray_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create letterboxed frame (pad to MODEL_WIDTH x MODEL_HEIGHT)
    letterboxed = np.full((MODEL_HEIGHT, MODEL_WIDTH), 0, dtype=np.uint8)
    offset_x = (MODEL_WIDTH - new_width) // 2
    offset_y = (MODEL_HEIGHT - new_height) // 2
    letterboxed[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized

    quantized = np.clip(letterboxed.astype(np.int16) - 128, -128, 127).astype(np.int8)
    return quantized[np.newaxis, :, :, np.newaxis], (offset_x, offset_y, new_width, new_height, scale)


def map_detections_to_frame(detections, letterbox_info, frame_shape):
    """Map model-space xywh detections back to original camera frame."""
    offset_x, offset_y, letterbox_width, letterbox_height, scale = letterbox_info
    frame_height, frame_width = frame_shape[:2]
    mapped = []
    for label, confidence, box in detections:
        x, y, width, height = box
        # Remove letterbox offset and scale back to original camera coordinates
        mapped_box = np.array([
            (x - offset_x) / scale,
            (y - offset_y) / scale,
            width / scale,
            height / scale,
        ], dtype=np.float32)
        # Clip to frame bounds
        mapped_box[[0, 2]] = np.clip(
            [mapped_box[0], mapped_box[0] + mapped_box[2]], 0, frame_width - 1,
        )
        mapped_box[[1, 3]] = np.clip(
            [mapped_box[1], mapped_box[1] + mapped_box[3]], 0, frame_height - 1,
        )
        mapped_box[2] -= mapped_box[0]
        mapped_box[3] -= mapped_box[1]
        mapped.append((label, confidence, mapped_box))
    return mapped


def make_process_fn(
    runner, args, keypoint_runner=None, embedding_runner=None, face_database=None,
    batch_session=None,
):
    recognition_tracker = RecognitionTracker(
        args.recognition_window,
        hold_frames=args.recognition_hold_frames,
        hysteresis=args.match_hysteresis,
        switch_confirm_frames=args.switch_confirm_frames,
        acquire_confirm_frames=args.acquire_confirm_frames,
        min_margin=args.match_margin,
        debug=args.recognition_debug,
    )
    box_smoother = BoxSmoother()

    def process_fn(bgr_frame, raw_yuyv=None):
        recognition_tracker.begin_frame()
        recognized_names = []
        model_input, letterbox_info = preprocess_frame(bgr_frame, raw_yuyv)
        outputs = runner.infer([model_input])
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        detections = decode_face_outputs(
            outputs,
            box1_scale=args.box1_scale,
            box1_zero_point=args.box1_zero_point,
            box2_scale=args.box2_scale,
            box2_zero_point=args.box2_zero_point,
            score_scale=args.score_scale,
            score_zero_point=args.score_zero_point,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            image_width=MODEL_WIDTH,
            image_height=MODEL_HEIGHT,
        )
        detections = map_detections_to_frame(detections, letterbox_info, bgr_frame.shape)
        if batch_session is not None:
            batch_session.update_faces(detections)
            detections = batch_session.order_detections(detections)
        if not detections:
            recognition_tracker.clear()
            box_smoother.clear()
        # Smoothed boxes are only for the drawn overlay (and anything derived
        # from frame_detections, like the card thumbnail crop) - keypoint and
        # embedding extraction below still use the raw `detections` boxes.
        annotated, frame_detections = annotate_frame(
            bgr_frame, box_smoother.smooth(detections), show_labels=False,
        )
        if keypoint_runner is not None:
            batch_embeddings = []
            for _label, _confidence, box in detections:
                keypoint_input, roi = preprocess_keypoint_roi(
                    bgr_frame, box, args.keypoint_input_zero_point,
                )
                keypoint_outputs = keypoint_runner.infer([keypoint_input])
                if not isinstance(keypoint_outputs, (list, tuple)):
                    keypoint_outputs = [keypoint_outputs]
                keypoints = decode_keypoints(
                    keypoint_outputs,
                    args.keypoint_output_scale,
                    args.keypoint_output_zero_point,
                )
                keypoints = map_keypoints_to_frame(keypoints, roi)
                if embedding_runner is not None:
                    embedding_input, landmarks = preprocess_embedding(bgr_frame, keypoints)
                    embedding_outputs = embedding_runner.infer([embedding_input])
                    if not isinstance(embedding_outputs, (list, tuple)):
                        embedding_outputs = [embedding_outputs]
                    embedding = decode_embedding(
                        embedding_outputs[0],
                        args.embedding_output_scale,
                        args.embedding_output_zero_point,
                    )
                    if batch_session is not None:
                        batch_embeddings.append(embedding)
                    if face_database is not None:
                        matched_name, similarity = recognition_tracker.update(
                            box, embedding, face_database, args.match_threshold,
                        )
                        recognition_text = matched_name if matched_name is not None else "unknown"
                        recognized_names.append(recognition_text)
                        if recognition_text:
                            cv2.putText(
                                annotated, recognition_text, (int(box[0]), max(int(box[1]) - 28, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                            )
            if batch_session is not None:
                batch_session.update_preview_embeddings(batch_embeddings)
                batch_session.submit(batch_embeddings)
        log_str = f"faces: {len(detections)}"
        if recognized_names:
            log_str += " " + ",".join(recognized_names)
        return annotated, frame_detections, runner.infer_time_ms, log_str or "no faces"

    return process_fn


def main() -> None:
    args = AppConfig()
    default_log_level = "DEBUG" if args.recognition_debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, os.environ.get("FACE_ID_LOG_LEVEL", default_log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_face_models()
    runtime_flags = build_runtime_flags(args.tda, args.runtime_flags)

    if args.display:
        os.environ.setdefault("XDG_RUNTIME_DIR", "/var/run/user/0")
        os.environ.setdefault("WAYLAND_DISPLAY", "wayland-1")

    ok, message = enable_npu_clock()
    print(f"[NPU] {message}")
    ok, message = configure_npu_userspace_frequency("max")
    print(f"[NPU] {message}")

    runner = VMFBInferenceRunner(
        args.model,
        device_uri=args.device,
        function="main",
        runtime_flags=runtime_flags,
        device_outputs=False,
    )
    keypoint_runner = None
    if args.keypoint_model:
        keypoint_runner = VMFBInferenceRunner(
            args.keypoint_model,
            device_uri=args.device,
            function="main",
            runtime_flags=runtime_flags,
            device_outputs=False,
        )
    embedding_runner = None
    if args.embedding_model:
        embedding_runner = VMFBInferenceRunner(
            args.embedding_model,
            device_uri=args.device,
            function="main",
            runtime_flags=runtime_flags,
            device_outputs=False,
        )
    face_database = None
    face_database = FaceDatabase(args.face_database)
    batch_session = BatchEnrollmentSession(face_database, args.enroll_frames)
    qt_ui = QtEnrollmentUI(batch_session)

    try:
        run_video_inference_loop(
            args, make_process_fn(
                runner, args, keypoint_runner, embedding_runner, face_database,
                batch_session,
            ), "Face Detection Stage 2", qt_ui,
        )
    finally:
        qt_ui.stop()
        cleanup_npu_after_inference()


if __name__ == "__main__":
    main()