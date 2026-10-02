# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for the demos' default-model resolution helpers (T5).

Each demo's ``setup_demo.py`` exposes a ``local_*`` helper that points the
inference scripts' ``-m``/``--model`` at whatever the setup downloaded. The
helpers check file presence only (no manifest requirement), so untracked
(``--no-update``) copies work as defaults too.
"""

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_LIQUID_REQUIRED = ("token_embeddings.npy", "config.json", "tokenizer.json")


def _load_setup(path_parts):
    path = _REPO_ROOT.joinpath(*path_parts)
    spec = importlib.util.spec_from_file_location("_default_test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


liquid_setup = _load_setup(("LiquidAI", "LiquidAI-LFM2.5", "setup_demo.py"))
vl_setup = _load_setup(("LiquidAI", "LiquidAI-LFM2-VL-450M", "setup_demo.py"))
face_setup = _load_setup(("Face_ID", "setup_demo.py"))

from moonshine import setup_demo as moonshine_setup  # noqa: E402
from moonshine_streaming import setup_demo as moonshine_streaming_setup  # noqa: E402
from pose_estimation import setup_demo as pose_setup  # noqa: E402


def _make_dir(base: Path, repo_id: str, filenames) -> Path:
    model_dir = base / repo_id
    for filename in filenames:
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
    return model_dir


# ── LiquidAI-LFM2.5 ───────────────────────────────────────────────────────────


def _liquid_files(model_files=("transformer.vmfb", "lm_head.vmfb")):
    return tuple(model_files) + _LIQUID_REQUIRED


def test_liquid_default_picks_first_complete_repo_in_map_order(tmp_path):
    # The 230M dir is incomplete; the 350M-w8a8 repo is complete -> it wins.
    _make_dir(tmp_path, liquid_setup._HF_REPO_MAP["230m"], _liquid_files()[:2])
    w8a8 = _make_dir(tmp_path, liquid_setup._HF_REPO_MAP["350m-w8a8"], _liquid_files())

    assert liquid_setup.local_liquid_model_path(base_dir=tmp_path) == w8a8 / "transformer.vmfb"


def test_liquid_main_model_filename_preference(tmp_path):
    repo_id = liquid_setup._HF_REPO_MAP["350m-w8a8"]
    model_dir = tmp_path / repo_id

    # Monolithic layout: the default resolves to model.vmfb.
    _make_dir(tmp_path, repo_id, _liquid_files(("model.vmfb",)))
    assert liquid_setup.local_liquid_model_path(base_dir=tmp_path) == model_dir / "model.vmfb"

    # body.vmfb + lm_head.vmfb: the body layout is preferred over the monolith.
    (model_dir / "body.vmfb").write_text("body.vmfb")
    (model_dir / "lm_head.vmfb").write_text("lm_head.vmfb")
    assert liquid_setup.local_liquid_model_path(base_dir=tmp_path) == model_dir / "body.vmfb"

    # transformer.vmfb: the newest layout wins.
    (model_dir / "transformer.vmfb").write_text("transformer.vmfb")
    assert liquid_setup.local_liquid_model_path(base_dir=tmp_path) == model_dir / "transformer.vmfb"


def test_liquid_default_requires_the_full_file_set(tmp_path):
    # A dir with only the model files (missing config.json) is not a complete
    # local model and must not be defaulted to.
    _make_dir(
        tmp_path,
        liquid_setup._HF_REPO_MAP["350m-w8a8"],
        ("transformer.vmfb", "lm_head.vmfb"),
    )
    assert liquid_setup.local_liquid_model_path(base_dir=tmp_path) is None


def test_liquid_explicit_unknown_repo_passes_through(tmp_path):
    # A raw HF repo id not in the map resolves through unchanged.
    _make_dir(tmp_path, "org/custom-liquid", _liquid_files())
    assert liquid_setup.local_liquid_model_path(model="org/custom-liquid", base_dir=tmp_path) == (
        tmp_path / "org" / "custom-liquid" / "transformer.vmfb"
    )


def test_liquid_missing_returns_none(tmp_path):
    assert liquid_setup.local_liquid_model_path(base_dir=tmp_path) is None
    assert liquid_setup.local_liquid_model_path(model="350m", base_dir=tmp_path) is None


# ── LFM2-VL-450M INT8 (model dirs, not files) ─────────────────────────────────


def test_vl_int8_default_returns_the_model_dir(tmp_path):
    model_dir = _make_dir(
        tmp_path, vl_setup._HF_REPO_MAP["default"], vl_setup._LFM2VL_W8_REQUIRED_FILES
    )
    assert vl_setup.local_lfm2vl_model_dir(base_dir=tmp_path) == model_dir


def test_vl_int8_incomplete_dir_returns_none(tmp_path):
    _make_dir(
        tmp_path,
        vl_setup._HF_REPO_MAP["default"],
        vl_setup._LFM2VL_W8_REQUIRED_FILES[:4],
    )
    assert vl_setup.local_lfm2vl_model_dir(base_dir=tmp_path) is None


def test_vl_int8_name_aliases_resolve_to_the_same_repo(tmp_path):
    model_dir = _make_dir(
        tmp_path, vl_setup._HF_REPO_MAP["w8"], vl_setup._LFM2VL_W8_REQUIRED_FILES
    )
    for name in ("default", "w8", "int8"):
        assert vl_setup.local_lfm2vl_model_dir(model=name, base_dir=tmp_path) == model_dir


def test_vl_int8_missing_returns_none(tmp_path):
    assert vl_setup.local_lfm2vl_model_dir(base_dir=tmp_path) is None


# ── moonshine (model dirs, not files) ─────────────────────────────────────────


def test_moonshine_default_returns_the_model_dir(tmp_path):
    model_dir = _make_dir(
        tmp_path,
        moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"],
        moonshine_setup._MOONSHINE_REQUIRED_FILES,
    )
    assert moonshine_setup.local_moonshine_model_dir(base_dir=tmp_path) == model_dir


def test_moonshine_incomplete_dir_returns_none(tmp_path):
    _make_dir(
        tmp_path,
        moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"],
        moonshine_setup._MOONSHINE_REQUIRED_FILES[:2],
    )
    assert moonshine_setup.local_moonshine_model_dir(base_dir=tmp_path) is None


def test_moonshine_streaming_default_returns_the_model_dir(tmp_path):
    model_dir = _make_dir(
        tmp_path,
        moonshine_streaming_setup._HF_REPO_MAP["streaming-tiny-en"],
        moonshine_streaming_setup._REQUIRED_FILES,
    )
    assert moonshine_streaming_setup.local_moonshine_streaming_model_dir(base_dir=tmp_path) == model_dir


def test_moonshine_streaming_missing_returns_none(tmp_path):
    assert (
        moonshine_streaming_setup.local_moonshine_streaming_model_dir(base_dir=tmp_path)
        is None
    )


# ── pose estimation ───────────────────────────────────────────────────────────


def test_pose_default_returns_the_model_file(tmp_path):
    model_dir = _make_dir(tmp_path, pose_setup._POSE_HF_REPO, (pose_setup._MODEL_FILENAME,))
    assert pose_setup.local_pose_model_path(base_dir=tmp_path) == model_dir / "yolo_pose.vmfb"


def test_pose_missing_returns_none(tmp_path):
    assert pose_setup.local_pose_model_path(base_dir=tmp_path) is None


# ── Face_ID ───────────────────────────────────────────────────────────────────


def test_face_id_default_returns_the_detector(tmp_path):
    model_dir = _make_dir(
        tmp_path, face_setup.FACE_ID_HF_REPO, ("face_detection.vmfb", "face.jpg")
    )
    assert face_setup.local_face_id_model_path(base_dir=tmp_path) == (
        model_dir / "face_detection.vmfb"
    )


def test_face_id_dir_without_the_vmfb_returns_none(tmp_path):
    # A sample image alone is not a usable model.
    _make_dir(tmp_path, face_setup.FACE_ID_HF_REPO, ("face.jpg",))
    assert face_setup.local_face_id_model_path(base_dir=tmp_path) is None


def test_face_id_explicit_repo_id(tmp_path):
    model_dir = _make_dir(tmp_path, "org/custom-face-id", ("face_detection.vmfb",))
    assert (
        face_setup.local_face_id_model_path(model="org/custom-face-id", base_dir=tmp_path)
        == model_dir / "face_detection.vmfb"
    )


# ── helpers respect the MODELS default base dir ───────────────────────────────


def test_local_helpers_default_base_dir_honours_MODELS(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS", str(tmp_path))
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_dir(tmp_path, repo_id, moonshine_setup._MOONSHINE_REQUIRED_FILES)
    assert moonshine_setup.local_moonshine_model_dir() == model_dir
