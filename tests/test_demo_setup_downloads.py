# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from unittest import mock

import pytest

from gemma3 import setup_demo as gemma_setup
from functiongemma import setup_demo as functiongemma_setup
from qwen2_5 import setup_demo as qwen25_setup
from qwen3 import setup_demo as qwen3_setup
from moonshine import setup_demo as moonshine_setup
from moonshine_streaming import setup_demo as moonshine_streaming_setup
from object_detection import setup_demo as object_detection_setup
from pose_estimation import setup_demo as pose_setup
from utils.download import DownloadError, ModelVersionNotFoundError, write_manifest
from utils import model_setup
from utils.version import examples_version

_REVISION = "abc123"
_PINS = "v2.0.0"
_EXAMPLES_VERSION = examples_version()


def _load_liquid_setup():
    # The Liquid demo dir name has a dot + hyphen, so it is not importable as
    # a package; load setup_demo.py by file path (as setup_demos.py does).
    path = Path(__file__).resolve().parents[1] / "LiquidAI" / "LiquidAI-LFM2.5" / "setup_demo.py"
    spec = importlib.util.spec_from_file_location("liquid_setup_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


liquid_setup = _load_liquid_setup()


def _fake_download(default_base_dir: Path):
    def download(repo_id: str, filename: str, *, base_dir: Path | None = None, revision: str | None = None):
        root = Path(base_dir) if base_dir is not None else default_base_dir
        path = root / repo_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{filename}@{revision or 'head'}")
        return path

    return download


def _manifest(model_dir: Path) -> dict:
    return json.loads((model_dir / ".manifest.json").read_text())


def _make_moonshine_copy(base_dir, repo_id, revision, *, version=_EXAMPLES_VERSION):
    model_dir = base_dir / repo_id
    for filename in moonshine_setup._MOONSHINE_REQUIRED_FILES:
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old")
    write_manifest(
        model_dir,
        repo_id,
        list(moonshine_setup._MOONSHINE_REQUIRED_FILES),
        version=version,
        revision=revision,
    )
    return model_dir


# ── setup: version resolution ──────────────────────────────────────────────────


def test_setup_default_tracks_examples_version(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ),
    ):
        moonshine_setup.setup_moonshine(["tiny-en"])

    # No explicit version: the built-in repo must be resolved at the
    # torq-examples version, and the manifest must record it.
    rev.assert_called_once_with(repo_id, revision=_EXAMPLES_VERSION)
    manifest = _manifest(base_dir / repo_id)
    assert manifest["version"] == _EXAMPLES_VERSION
    assert manifest["revision"] == _REVISION


def test_setup_pins_explicit_version(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"], model_version=_PINS)

    rev.assert_called_once_with(repo_id, revision=_PINS)
    # The download itself must use the pinned version, not the examples version.
    for call in download.call_args_list:
        assert call.kwargs["revision"] == _PINS
    assert _manifest(base_dir / repo_id)["version"] == _PINS


def test_setup_per_model_version_spec_wins_over_global(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ),
    ):
        moonshine_setup.setup_moonshine([f"tiny-en:{_PINS}"], model_version="v9.9.9")

    # The 'name:version' suffix takes precedence over --model-version.
    rev.assert_called_once_with(repo_id, revision=_PINS)
    assert _manifest(base_dir / repo_id)["version"] == _PINS


def test_setup_custom_repo_defaults_to_head_untracked(tmp_path):
    base_dir = tmp_path
    repo_id = "custom/moonshine"

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision") as rev,
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
    ):
        moonshine_setup.setup_moonshine(["custom/moonshine"])

    # Non-Synaptics repo without an explicit version: no revision is passed to
    # the HF helpers (HEAD) and nothing is resolved/tracked.
    rev.assert_not_called()
    for call in download.call_args_list:
        assert call.kwargs["revision"] is None
    manifest = _manifest(base_dir / repo_id)
    assert manifest["version"] is None
    assert manifest["revision"] is None


def test_setup_custom_repo_with_explicit_version_is_tracked(tmp_path):
    base_dir = tmp_path
    repo_id = "custom/moonshine"

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ),
    ):
        moonshine_setup.setup_moonshine(["custom/moonshine"], model_version="v1.0.0")

    rev.assert_called_once_with(repo_id, revision="v1.0.0")
    assert _manifest(base_dir / repo_id)["version"] == "v1.0.0"


def test_setup_missing_version_fails_loudly(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(
            model_setup, "get_hf_revision",
            side_effect=ModelVersionNotFoundError(f"Model version '{_PINS}' not found in Hugging Face repo '{repo_id}'."),
        ),
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        try:
            moonshine_setup.setup_moonshine(["tiny-en"], model_version=_PINS)
            raise AssertionError("expected DownloadError")
        except DownloadError:
            pass
    # A missing version must never fall back to another revision.
    download.assert_not_called()


def test_setup_missing_examples_version_fails_loudly(tmp_path):
    base_dir = tmp_path

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch("utils.version.examples_version", return_value=None),
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        try:
            moonshine_setup.setup_moonshine(["tiny-en"])
            raise AssertionError("expected DownloadError")
        except DownloadError as exc:
            assert "VERSION" in str(exc)
    download.assert_not_called()


# ── setup: refresh semantics (pre-existing behaviour, now versioned) ───────────


def test_setup_skips_when_revision_matches(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    _make_moonshine_copy(base_dir, repo_id, _REVISION)

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"])

    download.assert_not_called()


def test_setup_downloads_required_files_and_records_revision(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = base_dir / repo_id

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"])

    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    manifest = _manifest(model_dir)
    assert manifest["files"] == sorted(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    assert manifest["revision"] == _REVISION


def test_setup_refreshes_and_clears_stale_files_on_revision_change(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = base_dir / repo_id
    model_dir.mkdir(parents=True)
    # A complete-but-old copy, including a file dropped from the new set.
    for filename in (*moonshine_setup._MOONSHINE_REQUIRED_FILES, "preprocessor.onnx"):
        (model_dir / filename).write_text("old")
    write_manifest(
        model_dir,
        repo_id,
        [*moonshine_setup._MOONSHINE_REQUIRED_FILES, "preprocessor.onnx"],
        version=_EXAMPLES_VERSION,
        revision="old-revision",
    )

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"])

    # All required files re-downloaded after the stale dir was cleared.
    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    # The dropped file is gone and content was refreshed.
    assert not (model_dir / "preprocessor.onnx").exists()
    assert (model_dir / "encoder.vmfb").read_text() == f"encoder.vmfb@{_EXAMPLES_VERSION}"
    assert _manifest(model_dir)["revision"] == _REVISION
    assert "preprocessor.onnx" not in _manifest(model_dir)["files"]


def test_setup_offline_uses_local_files(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = base_dir / repo_id
    files = list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    for filename in files:
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
    write_manifest(model_dir, repo_id, files, version=_EXAMPLES_VERSION, revision="old-revision")

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=None),
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"])

    download.assert_not_called()


def test_setup_offline_with_missing_files_fails_loudly(tmp_path):
    base_dir = tmp_path

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=None),
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=ConnectionError("offline")
        ),
    ):
        try:
            moonshine_setup.setup_moonshine(["tiny-en"])
            raise AssertionError("expected DownloadError")
        except DownloadError:
            pass


def test_setup_no_update_writes_no_manifest(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
        mock.patch.object(model_setup, "get_hf_revision") as rev,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"], no_update=True)

    # Untracked: files are downloaded but nothing is resolved or recorded.
    rev.assert_not_called()
    assert download.called
    assert not (base_dir / repo_id / ".manifest.json").exists()


def test_setup_no_update_reuses_existing_files(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = base_dir / repo_id
    model_dir.mkdir(parents=True)
    for filename in moonshine_setup._MOONSHINE_REQUIRED_FILES:
        (model_dir / filename).write_text("pre-existing")

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"], no_update=True)

    # The copy is complete, so re-running --no-update must not touch the files.
    download.assert_not_called()
    assert (model_dir / "encoder.vmfb").read_text() == "pre-existing"
    assert not (model_dir / ".manifest.json").exists()


def test_setup_no_update_downloads_head_not_builtin_version(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
        mock.patch.object(model_setup, "get_hf_revision") as rev,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"], no_update=True)

    # Untracked and unversioned: the built-in VERSION tag is not used as the
    # download revision (untagged repos cannot resolve it); HEAD is.
    rev.assert_not_called()
    assert download.called
    for call in download.call_args_list:
        assert call.kwargs["revision"] is None
    assert not (base_dir / repo_id / ".manifest.json").exists()


def test_setup_no_update_keeps_explicit_version(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
        mock.patch.object(model_setup, "get_hf_revision") as rev,
    ):
        moonshine_setup.setup_moonshine(
            ["tiny-en"], no_update=True, model_version=_PINS
        )

    # An explicitly pinned version wins over the untracked HEAD default,
    # but nothing is resolved or recorded.
    rev.assert_not_called()
    assert download.called
    for call in download.call_args_list:
        assert call.kwargs["revision"] == _PINS
    assert not (base_dir / repo_id / ".manifest.json").exists()


def test_setup_no_update_keeps_per_model_version_spec(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
    ):
        moonshine_setup.setup_moonshine([f"tiny-en:{_PINS}"], no_update=True)

    # A per-model 'name:version' spec wins over the untracked HEAD default.
    assert download.called
    for call in download.call_args_list:
        assert call.kwargs["revision"] == _PINS
    assert not (base_dir / repo_id / ".manifest.json").exists()


# ── inference: manifest-driven refresh ─────────────────────────────────────────


def test_inference_refreshes_stale_models(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_moonshine_copy(base_dir, repo_id, "old-revision")

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
    ):
        # model_dir is base/<repo_id>; the helper must recover base_dir.
        moonshine_setup.ensure_moonshine_models(model_dir)

    # The check must resolve the version recorded in the manifest...
    rev.assert_called_once_with(repo_id, revision=_EXAMPLES_VERSION)
    # ...and re-download at it.
    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    assert (model_dir / "encoder.vmfb").read_text() == f"encoder.vmfb@{_EXAMPLES_VERSION}"
    assert _manifest(model_dir)["revision"] == _REVISION


def test_inference_no_refresh_skips_network(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_moonshine_copy(base_dir, repo_id, "old-revision")

    with (
        mock.patch.object(model_setup, "get_hf_revision") as revision,
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir, refresh=False)

    revision.assert_not_called()
    download.assert_not_called()


def test_inference_tracks_manifest_version_not_latest(tmp_path):
    """A pinned model re-checks its own tag on every launch — never 'latest'.

    The ``latest`` mechanism is gone: the version the model was set up with is
    the one its refreshes track, even when it differs from the examples version.
    """
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_moonshine_copy(base_dir, repo_id, _REVISION, version=_PINS)

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir)

    assert rev.call_count == 1
    assert rev.call_args.kwargs["revision"] == _PINS
    download.assert_not_called()


def test_inference_pinned_tag_move_refreshes_at_pinned_version(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_moonshine_copy(base_dir, repo_id, "old-sha", version=_PINS)

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value="new-sha") as rev,
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ) as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir)

    # Re-syncs the pinned tag (which moved upstream) — still the pinned tag.
    assert rev.call_args.kwargs["revision"] == _PINS
    for call in download.call_args_list:
        assert call.kwargs["revision"] == _PINS
    manifest = _manifest(model_dir)
    assert manifest["version"] == _PINS
    assert manifest["revision"] == "new-sha"


def test_inference_unversioned_custom_model_makes_no_network_calls(tmp_path):
    base_dir = tmp_path
    repo_id = "custom/moonshine"
    model_dir = _make_moonshine_copy(base_dir, repo_id, None, version=None)

    with (
        mock.patch.object(model_setup, "get_hf_revision") as rev,
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir)

    rev.assert_not_called()
    download.assert_not_called()


def test_inference_without_manifest_does_not_download(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = base_dir / repo_id
    model_dir.mkdir(parents=True)

    with (
        mock.patch.object(model_setup, "get_hf_revision") as revision,
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir)

    revision.assert_not_called()
    download.assert_not_called()


def test_inference_offline_uses_local_files(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_moonshine_copy(base_dir, repo_id, "old-revision")

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=None),
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir)

    download.assert_not_called()


def test_inference_offline_with_missing_files_keeps_local_state(tmp_path):
    # Offline + incomplete: the repair attempt fails but is logged, not
    # raised, and the demo proceeds with whatever is local (inference then
    # fails loudly at load time if the model is actually unusable).
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_moonshine_copy(base_dir, repo_id, "old-revision")
    (model_dir / "encoder.vmfb").unlink()

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=None),
        mock.patch.object(
            moonshine_setup, "download_from_hf", side_effect=ConnectionError("offline")
        ) as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir)  # must not raise

    assert download.called
    # The failed repair left the (incomplete) copy in place.
    assert _manifest(model_dir)["revision"] == "old-revision"
    assert not (model_dir / "encoder.vmfb").exists()


# ── object detection / pose estimation ─────────────────────────────────────────


def _make_yolo_copy(base_dir, repo_id, *, version=_EXAMPLES_VERSION, revision=_REVISION):
    model_dir = base_dir / repo_id
    filenames = (*object_detection_setup._model_filenames(repo_id), object_detection_setup._LABELS_FILENAME)
    for filename in filenames:
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
    write_manifest(
        model_dir,
        repo_id,
        list(filenames),
        version=version,
        revision=revision,
    )
    return model_dir


def test_yolo_setup_default_tracks_examples_version(tmp_path):
    base_dir = tmp_path
    repo_id = object_detection_setup._OD_HF_REPO_MAP["nano"]

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(
            model_setup, "get_hf_revision", return_value=_REVISION
        ) as rev,
        mock.patch.object(object_detection_setup, "hf_file_exists", return_value=True),
        mock.patch.object(object_detection_setup, "list_hf_files", return_value=[]),
        mock.patch.object(
            object_detection_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ),
    ):
        object_detection_setup.setup_object_detection()

    rev.assert_called_once_with(repo_id, revision=_EXAMPLES_VERSION)
    assert _manifest(base_dir / repo_id)["version"] == _EXAMPLES_VERSION


def test_yolo_inference_tracks_manifest_version(tmp_path):
    repo_id = object_detection_setup._OD_HF_REPO_MAP["nano"]
    model_dir = _make_yolo_copy(tmp_path, repo_id, version=_PINS)

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(object_detection_setup, "download_from_hf") as download,
    ):
        object_detection_setup.ensure_object_detection_models(model_dir)

    assert rev.call_args.kwargs["revision"] == _PINS
    download.assert_not_called()


def test_yolo26_setup_downloads_untracked_at_head(tmp_path):
    base_dir = tmp_path
    nano_repo = object_detection_setup._OD_HF_REPO_MAP["nano"]
    yolo26_repo = object_detection_setup._YOLO26_REPO_ID

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(object_detection_setup, "hf_file_exists", return_value=True),
        mock.patch.object(object_detection_setup, "list_hf_files", return_value=[]),
        mock.patch.object(
            object_detection_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ),
    ):
        object_detection_setup.setup_object_detection()

    # The yolo26 repo has no version tags, so it is downloaded at HEAD and left
    # untracked; the revision check only happens for the tracked nano repo.
    rev.assert_called_once_with(nano_repo, revision=_EXAMPLES_VERSION)
    manifest = _manifest(base_dir / yolo26_repo)
    assert manifest["version"] is None
    assert set(manifest["files"]) == {
        "yolo26n_npu.vmfb", "yolo26s_npu.vmfb", object_detection_setup._LABELS_FILENAME,
    }


def test_yolo26_inference_keeps_untracked_copy(tmp_path):
    yolo26_repo = object_detection_setup._YOLO26_REPO_ID
    model_dir = _make_yolo_copy(tmp_path, yolo26_repo, version=None, revision=None)

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(object_detection_setup, "download_from_hf") as download,
    ):
        object_detection_setup.ensure_object_detection_models(model_dir)

    # Untracked copies (version=None) are never checked against the hub.
    rev.assert_not_called()
    download.assert_not_called()


def test_yolo_sample_files_fall_back_to_repo_root():
    yolo26_repo = object_detection_setup._YOLO26_REPO_ID

    with mock.patch.object(
        object_detection_setup, "list_hf_files",
        return_value=["README.md", "labels.json", "samples/bus.jpg"],
    ):
        assert object_detection_setup._list_sample_files(yolo26_repo, None) == ["samples/bus.jpg"]

    with mock.patch.object(
        object_detection_setup, "list_hf_files",
        return_value=["README.md", "labels.json", "bus.jpg", "yolo26n_npu.vmfb"],
    ):
        assert object_detection_setup._list_sample_files(yolo26_repo, None) == ["bus.jpg"]


def test_pose_inference_tracks_manifest_version(tmp_path):
    repo_id = pose_setup._POSE_HF_REPO
    model_dir = tmp_path / repo_id
    model_path = model_dir / pose_setup._MODEL_FILENAME
    model_path.parent.mkdir(parents=True)
    model_path.write_text(pose_setup._MODEL_FILENAME)
    write_manifest(
        model_dir,
        repo_id,
        [pose_setup._MODEL_FILENAME],
        version=_PINS,
        revision=_REVISION,
    )

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION) as rev,
        mock.patch.object(pose_setup, "download_from_hf") as download,
    ):
        pose_setup.ensure_pose_estimation_models(model_dir)

    assert rev.call_args.kwargs["revision"] == _PINS
    download.assert_not_called()


# ── gemma3 ─────────────────────────────────────────────────────────────────────


def test_gemma_skips_when_revision_matches(tmp_path):
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = base_dir / repo_id
    files = [
        "model.vmfb.trim",
        *gemma_setup._GEMMA3_REQUIRED_FILES,
    ]
    for filename in files:
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
    write_manifest(
        model_dir, repo_id, files, version=_EXAMPLES_VERSION, revision=_REVISION
    )

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(
            model_setup, "get_hf_revision", return_value=_REVISION
        ) as rev,
        mock.patch.object(gemma_setup, "download_from_hf") as download,
        mock.patch.object(model_setup, "download_from_hf", new=gemma_setup.download_from_hf),
    ):
        gemma_setup.setup_gemma3(["instruct"])

    rev.assert_called_once_with(repo_id, revision=_EXAMPLES_VERSION)
    download.assert_not_called()


def test_gemma_repairs_incomplete_download_and_records_lut(tmp_path):
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = base_dir / repo_id
    model_dir.mkdir(parents=True)
    (model_dir / "model.vmfb.trim").write_text("model")
    # Manifest matches upstream revision but required files are missing,
    # so this is an "incomplete" (resumable) state, not a stale one.
    write_manifest(
        model_dir, repo_id, ["model.vmfb.trim"],
        version=_EXAMPLES_VERSION, revision=_REVISION,
    )

    def exists(_repo_id, filename, revision=None):
        assert _repo_id == repo_id
        assert revision == _EXAMPLES_VERSION
        return filename == gemma_setup._GEMMA3_TRIM_LUT_FILENAME

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "hf_file_exists", side_effect=exists),
        mock.patch.object(model_setup, "hf_file_exists", new=gemma_setup.hf_file_exists),
        mock.patch.object(
            gemma_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=gemma_setup.download_from_hf),
    ):
        gemma_setup.setup_gemma3(["instruct"])

    # The existing model.vmfb.trim is preserved (not re-downloaded).
    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == [
        *gemma_setup._GEMMA3_REQUIRED_FILES,
        gemma_setup._GEMMA3_TRIM_LUT_FILENAME,
    ]
    assert (model_dir / "model.vmfb.trim").exists()
    manifest = _manifest(model_dir)
    assert manifest["files"] == [
        "config.json",
        "model.vmfb.trim",
        "token_embeddings.npy",
        "token_id_lut.npy",
        "tokenizer.json",
    ]
    assert manifest["version"] == _EXAMPLES_VERSION
    assert manifest["revision"] == _REVISION


def test_gemma_downloads_split_lm_head_pair(tmp_path):
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = base_dir / repo_id

    def exists(_repo_id, filename, revision=None):
        assert _repo_id == repo_id
        return filename in {"transformer.vmfb", "lm_head.vmfb.trim"}

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "hf_file_exists", side_effect=exists),
        mock.patch.object(model_setup, "hf_file_exists", new=gemma_setup.hf_file_exists),
        mock.patch.object(
            gemma_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=gemma_setup.download_from_hf),
    ):
        gemma_setup.setup_gemma3(["instruct"])

    downloaded = [call.args[1] for call in download.call_args_list]
    expected_files = [
        "transformer.vmfb",
        "lm_head.vmfb.trim",
        *gemma_setup._GEMMA3_REQUIRED_FILES,
    ]
    assert downloaded == expected_files
    manifest = _manifest(model_dir)
    assert manifest["files"] == sorted(expected_files)
    assert manifest["version"] == _EXAMPLES_VERSION
    assert manifest["revision"] == _REVISION


def test_gemma_repairs_existing_split_body_by_fetching_lm_head(tmp_path):
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = base_dir / repo_id
    model_dir.mkdir(parents=True)
    (model_dir / "transformer.vmfb").write_text("model")
    write_manifest(
        model_dir, repo_id, ["transformer.vmfb"],
        version=_EXAMPLES_VERSION, revision=_REVISION,
    )

    def exists(_repo_id, filename, revision=None):
        assert _repo_id == repo_id
        return filename in {"transformer.vmfb", "lm_head.vmfb"}

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "hf_file_exists", side_effect=exists),
        mock.patch.object(model_setup, "hf_file_exists", new=gemma_setup.hf_file_exists),
        mock.patch.object(
            gemma_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=gemma_setup.download_from_hf),
    ):
        gemma_setup.setup_gemma3(["instruct"])

    downloaded = [call.args[1] for call in download.call_args_list]
    expected_files = [
        "transformer.vmfb",
        "lm_head.vmfb",
        *gemma_setup._GEMMA3_REQUIRED_FILES,
    ]
    assert downloaded == [
        "lm_head.vmfb",
        *gemma_setup._GEMMA3_REQUIRED_FILES,
    ]
    manifest = _manifest(model_dir)
    assert manifest["files"] == sorted(expected_files)
    assert manifest["version"] == _EXAMPLES_VERSION
    assert manifest["revision"] == _REVISION


def _make_gemma_split_copy(model_dir: Path, lm_head_name: str, manifest_head: str, *, version=_EXAMPLES_VERSION, revision=_REVISION):
    """A complete split-model copy whose manifest recorded *manifest_head*."""
    model_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "transformer.vmfb",
        lm_head_name,
        *gemma_setup._GEMMA3_REQUIRED_FILES,
        gemma_setup._GEMMA3_TRIM_LUT_FILENAME,
    ]
    for filename in files:
        (model_dir / filename).write_text(filename)
    write_manifest(
        model_dir,
        gemma_setup.GEMMA3_HF_REPO_MAP["instruct"],
        [
            "transformer.vmfb",
            manifest_head,
            *gemma_setup._GEMMA3_REQUIRED_FILES,
            gemma_setup._GEMMA3_TRIM_LUT_FILENAME,
        ],
        version=version,
        revision=revision,
    )
    return model_dir


def test_inference_accepts_alternate_lm_head_name_without_downloading(tmp_path):
    """A supported LM head name the manifest didn't record is still complete.

    ``lm_head.vmfb`` in place of the recorded ``lm_head.vmfb.trim`` used to read
    as an incomplete copy, so every launch re-entered the download path.
    """
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = _make_gemma_split_copy(
        tmp_path / repo_id, "lm_head.vmfb", "lm_head.vmfb.trim"
    )

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "hf_file_exists") as exists,
        mock.patch.object(model_setup, "hf_file_exists", new=gemma_setup.hf_file_exists),
        mock.patch.object(gemma_setup, "download_from_hf") as download,
        mock.patch.object(model_setup, "download_from_hf", new=gemma_setup.download_from_hf),
    ):
        gemma_setup.ensure_gemma3_models(model_dir)

    exists.assert_not_called()
    download.assert_not_called()
    assert (model_dir / "lm_head.vmfb").exists()


def test_inference_still_repairs_a_genuinely_incomplete_copy(tmp_path):
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = _make_gemma_split_copy(
        tmp_path / repo_id, "lm_head.vmfb", "lm_head.vmfb.trim"
    )
    (model_dir / gemma_setup._GEMMA3_TRIM_LUT_FILENAME).unlink()

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "hf_file_exists", return_value=True),
        mock.patch.object(model_setup, "hf_file_exists", new=gemma_setup.hf_file_exists),
        mock.patch.object(
            gemma_setup, "download_from_hf", side_effect=_fake_download(tmp_path)
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=gemma_setup.download_from_hf),
    ):
        gemma_setup.ensure_gemma3_models(model_dir)

    downloaded = [call.args[1] for call in download.call_args_list]
    assert gemma_setup._GEMMA3_TRIM_LUT_FILENAME in downloaded


def test_inference_skips_refresh_when_model_dir_is_not_under_repo_id(tmp_path):
    """A model dir that isn't ``<base>/<repo id>`` must not be refreshed.

    ``base_dir_for`` cannot recover a base dir from such a layout (e.g. a bare
    Hugging Face clone in ``models/gemma-3-270m-it-torq``), and downloading with
    a guessed one fetches a second full copy into an unrelated directory.
    """
    model_dir = _make_gemma_split_copy(
        tmp_path / "models" / "gemma-3-270m-it-torq",
        "lm_head.vmfb",
        "lm_head.vmfb.trim",
    )

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value="new-revision"),
        mock.patch.object(gemma_setup, "hf_file_exists") as exists,
        mock.patch.object(model_setup, "hf_file_exists", new=gemma_setup.hf_file_exists),
        mock.patch.object(gemma_setup, "download_from_hf") as download,
        mock.patch.object(model_setup, "download_from_hf", new=gemma_setup.download_from_hf),
    ):
        gemma_setup.ensure_gemma3_models(model_dir)

    exists.assert_not_called()
    download.assert_not_called()
    # The stale-refresh path must not have cleared the dir it cannot replace.
    assert (model_dir / "transformer.vmfb").exists()
    assert not (tmp_path / "models" / "Synaptics").exists()


# ── prefill opt-in (gemma3) ─────────────────────────────────────────────────


def _gemma_repo_with_prefill(_repo_id, filename, revision=None):
    return filename in {
        "transformer.vmfb",
        "lm_head.vmfb",
        "transformer_prefill.vmfb",
    }


def _run_gemma_setup(base_dir: Path, exists, *, enable_prefill: bool):
    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(model_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "hf_file_exists", side_effect=exists),
        mock.patch.object(model_setup, "hf_file_exists", new=gemma_setup.hf_file_exists),
        mock.patch.object(
            gemma_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=gemma_setup.download_from_hf),
    ):
        gemma_setup.setup_gemma3(["instruct"], enable_prefill=enable_prefill)
    return download


def test_gemma_prefill_not_downloaded_without_flag(tmp_path):
    """The prefill model stays out of the download and the manifest by default."""
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]

    download = _run_gemma_setup(base_dir, _gemma_repo_with_prefill, enable_prefill=False)

    downloaded = [call.args[1] for call in download.call_args_list]
    assert "transformer_prefill.vmfb" not in downloaded
    assert "transformer_prefill.vmfb" not in _manifest(base_dir / repo_id)["files"]


def test_gemma_prefill_downloaded_and_tracked_with_flag(tmp_path):
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]

    download = _run_gemma_setup(base_dir, _gemma_repo_with_prefill, enable_prefill=True)

    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded[-1] == "transformer_prefill.vmfb"
    assert "transformer_prefill.vmfb" in _manifest(base_dir / repo_id)["files"]


def test_gemma_prefill_flag_is_noop_for_native_single_file_repo(tmp_path, caplog):
    """Flag on a repo that publishes no separate prefill model: that layout is
    a native single-file build with the prefill family compiled into
    transformer.vmfb, so nothing extra downloads and no warning is raised."""
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]

    with caplog.at_level(logging.INFO, logger="utils.model_setup"):
        download = _run_gemma_setup(
            base_dir,
            lambda _r, f, revision=None: f in {"transformer.vmfb", "lm_head.vmfb"},
            enable_prefill=True,
        )

    assert "transformer_prefill.vmfb" not in [
        call.args[1] for call in download.call_args_list
    ]
    assert "transformer_prefill.vmfb" not in _manifest(base_dir / repo_id)["files"]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any("nothing extra to download" in r.getMessage() for r in caplog.records)


def test_gemma_prefill_untracking_persists_across_setup_runs(tmp_path):
    """Flag on, then off: prefill leaves the manifest, the local file is kept,
    and later runs neither check nor re-download it."""
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    prefill_path = base_dir / repo_id / "transformer_prefill.vmfb"

    _run_gemma_setup(base_dir, _gemma_repo_with_prefill, enable_prefill=True)
    assert prefill_path.exists()
    assert "transformer_prefill.vmfb" in _manifest(base_dir / repo_id)["files"]

    # Re-run without the flag: the copy is complete, so nothing downloads, but
    # the prefill model is removed from tracking.
    download = _run_gemma_setup(base_dir, _gemma_repo_with_prefill, enable_prefill=False)
    download.assert_not_called()
    assert "transformer_prefill.vmfb" not in _manifest(base_dir / repo_id)["files"]
    assert prefill_path.exists()  # local file kept, just untracked

    # A further run stays excluded.
    download = _run_gemma_setup(base_dir, _gemma_repo_with_prefill, enable_prefill=False)
    download.assert_not_called()
    assert "transformer_prefill.vmfb" not in _manifest(base_dir / repo_id)["files"]


def test_gemma_prefill_retracked_with_flag_when_file_present(tmp_path):
    """Flag off, then on with the file already local: tracked, no download."""
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    prefill_path = base_dir / repo_id / "transformer_prefill.vmfb"

    _run_gemma_setup(base_dir, _gemma_repo_with_prefill, enable_prefill=False)
    assert not prefill_path.exists()
    prefill_path.write_text("prefill")

    download = _run_gemma_setup(base_dir, _gemma_repo_with_prefill, enable_prefill=True)
    download.assert_not_called()
    assert "transformer_prefill.vmfb" in _manifest(base_dir / repo_id)["files"]


def test_gemma_prefill_added_to_complete_copy_with_flag(tmp_path):
    """``--with-batch-prefill`` on an already-complete copy fetches only the prefill build."""
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]

    _run_gemma_setup(base_dir, _gemma_repo_with_prefill, enable_prefill=False)
    download = _run_gemma_setup(base_dir, _gemma_repo_with_prefill, enable_prefill=True)

    downloaded = [call.args[1] for call in download.call_args_list]
    assert "transformer_prefill.vmfb" in downloaded
    # The intact model set is not re-fetched (the hook is incremental).
    assert "transformer.vmfb" not in downloaded
    assert "lm_head.vmfb" not in downloaded
    assert "transformer_prefill.vmfb" in _manifest(base_dir / repo_id)["files"]


def test_gemma_prefill_flag_noop_on_complete_native_copy(tmp_path, caplog):
    """``--with-batch-prefill`` on a complete copy of a native (no separate
    prefill model) repo: the copy is already complete — nothing downloads,
    the manifest is unchanged, and no warning about a missing prefill model."""
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    no_prefill = lambda _r, f, revision=None: f in {"transformer.vmfb", "lm_head.vmfb"}

    _run_gemma_setup(base_dir, no_prefill, enable_prefill=False)
    before = _manifest(base_dir / repo_id)["files"]

    with caplog.at_level(logging.WARNING):
        download = _run_gemma_setup(base_dir, no_prefill, enable_prefill=True)

    download.assert_not_called()
    assert _manifest(base_dir / repo_id)["files"] == before
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_inference_stale_refresh_warns_about_dropped_prefill(tmp_path, caplog):
    """A tag move to a revision without the prefill build wipes the copy and
    re-downloads; the previously-tracked prefill file must not be dropped
    from tracking silently."""
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = _make_gemma_split_copy(
        base_dir / repo_id, "lm_head.vmfb", "lm_head.vmfb",
        revision="old-revision",
    )
    # The old revision also published and tracked a prefill build.
    prefill_path = model_dir / "transformer_prefill.vmfb"
    prefill_path.write_text("prefill")
    write_manifest(
        model_dir,
        repo_id,
        [
            "transformer.vmfb",
            "lm_head.vmfb",
            *gemma_setup._GEMMA3_REQUIRED_FILES,
            "transformer_prefill.vmfb",
        ],
        version=_EXAMPLES_VERSION,
        revision="old-revision",
    )

    def exists(_repo_id, filename, revision=None):
        # The new revision has no prefill build anymore.
        return filename in {"transformer.vmfb", "lm_head.vmfb", *gemma_setup._GEMMA3_REQUIRED_FILES}

    with (
        caplog.at_level(logging.WARNING),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "hf_file_exists", side_effect=exists),
        mock.patch.object(model_setup, "hf_file_exists", new=gemma_setup.hf_file_exists),
        mock.patch.object(
            gemma_setup, "download_from_hf", side_effect=_fake_download(base_dir)
        ),
        mock.patch.object(model_setup, "download_from_hf", new=gemma_setup.download_from_hf),
    ):
        gemma_setup.ensure_gemma3_models(model_dir)

    manifest = _manifest(model_dir)
    assert manifest["revision"] == _REVISION
    assert "transformer_prefill.vmfb" not in manifest["files"]
    assert not prefill_path.exists()  # the stale refresh wiped the dir
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("transformer_prefill.vmfb" in w for w in warnings)


# ── liquid ──────────────────────────────────────────────────────────────────


def test_liquid_w8a8_setup_downloads_prefill(tmp_path):
    """The w8a8 repos carry a batched prefill model at the model revision:
    the built-in names resolve and setup fetches the prefill build."""
    base_dir = tmp_path
    repo_id = liquid_setup._HF_REPO_MAP["230m-w8a8"]
    assert repo_id == "Synaptics/LiquidAI-LFM2.5-230M-w8a8-torq"

    def exists(_repo_id, filename, revision=None):
        assert _repo_id == repo_id
        return filename in {
            "transformer.vmfb",
            "lm_head.vmfb",
            "transformer_prefill.vmfb",
        }

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(liquid_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(liquid_setup, "hf_file_exists", side_effect=exists),
        mock.patch.object(model_setup, "hf_file_exists", new=liquid_setup.hf_file_exists),
        mock.patch.object(
            liquid_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=liquid_setup.download_from_hf),
    ):
        liquid_setup.setup_liquid(["230m-w8a8"], enable_prefill=True)

    downloaded = [call.args[1] for call in download.call_args_list]
    assert "transformer_prefill.vmfb" in downloaded
    assert "transformer_prefill.vmfb" in _manifest(base_dir / repo_id)["files"]


def test_liquid_downloads_new_file_set_with_optional_prefill(tmp_path):
    base_dir = tmp_path
    repo_id = liquid_setup._HF_REPO_MAP["230m"]

    def exists(_repo_id, filename, revision=None):
        assert _repo_id == repo_id
        assert revision == _EXAMPLES_VERSION
        return filename in {
            "transformer.vmfb",
            "lm_head.vmfb",
            "transformer_prefill.vmfb",
        }

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(liquid_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(liquid_setup, "hf_file_exists", side_effect=exists),
        mock.patch.object(model_setup, "hf_file_exists", new=liquid_setup.hf_file_exists),
        mock.patch.object(
            liquid_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=liquid_setup.download_from_hf),
    ):
        liquid_setup.setup_liquid(["230m"], enable_prefill=True)

    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == [
        "transformer.vmfb",
        "lm_head.vmfb",
        *liquid_setup._LIQUID_REQUIRED_FILES,
        "transformer_prefill.vmfb",
    ]
    manifest = _manifest(base_dir / repo_id)
    assert manifest["files"] == sorted(downloaded)
    assert manifest["version"] == _EXAMPLES_VERSION
    assert manifest["revision"] == _REVISION


def test_liquid_falls_back_to_legacy_body_set(tmp_path):
    base_dir = tmp_path
    repo_id = liquid_setup._HF_REPO_MAP["230m"]

    def exists(_repo_id, filename, revision=None):
        assert _repo_id == repo_id
        return filename in {"body.vmfb", "lm_head.vmfb"}

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(liquid_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(liquid_setup, "hf_file_exists", side_effect=exists),
        mock.patch.object(model_setup, "hf_file_exists", new=liquid_setup.hf_file_exists),
        mock.patch.object(
            liquid_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=liquid_setup.download_from_hf),
    ):
        liquid_setup.setup_liquid(["230m"])

    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == [
        "body.vmfb",
        "lm_head.vmfb",
        *liquid_setup._LIQUID_REQUIRED_FILES,
    ]
    # No prefill in this repo: not downloaded, not tracked.
    manifest = _manifest(base_dir / repo_id)
    assert "transformer_prefill.vmfb" not in manifest["files"]


def _make_liquid_copy(model_dir: Path, model_files, revision=_REVISION):
    model_dir.mkdir(parents=True, exist_ok=True)
    files = [*model_files, *liquid_setup._LIQUID_REQUIRED_FILES]
    for filename in files:
        (model_dir / filename).write_text(filename)
    write_manifest(
        model_dir,
        liquid_setup._HF_REPO_MAP["230m"],
        files,
        version=_EXAMPLES_VERSION,
        revision=revision,
    )
    return model_dir


def test_liquid_inference_accepts_alternate_model_set(tmp_path):
    """A manifest recorded the new set, but the local dir holds the legacy set.

    Both are valid decode bodies, so the tracked copy must be complete and no
    download must happen.
    """
    repo_id = liquid_setup._HF_REPO_MAP["230m"]
    model_dir = _make_liquid_copy(
        tmp_path / repo_id,
        ["body.vmfb", "lm_head.vmfb"],
    )
    # Rewrite the manifest as if the new set had been downloaded.
    write_manifest(
        model_dir,
        repo_id,
        ["transformer.vmfb", "lm_head.vmfb", *liquid_setup._LIQUID_REQUIRED_FILES],
        version=_EXAMPLES_VERSION,
        revision=_REVISION,
    )

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(liquid_setup, "hf_file_exists") as exists,
        mock.patch.object(model_setup, "hf_file_exists", new=liquid_setup.hf_file_exists),
        mock.patch.object(liquid_setup, "download_from_hf") as download,
        mock.patch.object(model_setup, "download_from_hf", new=liquid_setup.download_from_hf),
    ):
        liquid_setup.ensure_liquid_models(model_dir)

    exists.assert_not_called()
    download.assert_not_called()


def test_liquid_inference_repairs_missing_recorded_prefill(tmp_path):
    """The manifest tracked a prefill build the local dir lost: re-fetch it."""
    repo_id = liquid_setup._HF_REPO_MAP["230m"]
    model_dir = _make_liquid_copy(
        tmp_path / repo_id,
        ["transformer.vmfb", "lm_head.vmfb"],
    )
    (model_dir / "transformer_prefill.vmfb").write_text("prefill")
    write_manifest(
        model_dir,
        repo_id,
        [
            "transformer.vmfb",
            "lm_head.vmfb",
            *liquid_setup._LIQUID_REQUIRED_FILES,
            "transformer_prefill.vmfb",
        ],
        version=_EXAMPLES_VERSION,
        revision=_REVISION,
    )
    (model_dir / "transformer_prefill.vmfb").unlink()

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(liquid_setup, "hf_file_exists", return_value=True),
        mock.patch.object(model_setup, "hf_file_exists", new=liquid_setup.hf_file_exists),
        mock.patch.object(
            liquid_setup, "download_from_hf", side_effect=_fake_download(tmp_path)
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=liquid_setup.download_from_hf),
    ):
        liquid_setup.ensure_liquid_models(model_dir)

    downloaded = [call.args[1] for call in download.call_args_list]
    # The lost prefill build is re-fetched; the intact model set is not.
    assert "transformer_prefill.vmfb" in downloaded
    assert "transformer.vmfb" not in downloaded
    assert "lm_head.vmfb" not in downloaded


def test_liquid_prefill_not_downloaded_without_flag(tmp_path):
    """The prefill model stays out of the download and the manifest by default."""
    base_dir = tmp_path
    repo_id = liquid_setup._HF_REPO_MAP["230m"]

    def exists(_repo_id, filename, revision=None):
        return filename in {
            "transformer.vmfb",
            "lm_head.vmfb",
            "transformer_prefill.vmfb",
        }

    with (
        mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(liquid_setup, "check_requirements"),
        mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(liquid_setup, "hf_file_exists", side_effect=exists),
        mock.patch.object(model_setup, "hf_file_exists", new=liquid_setup.hf_file_exists),
        mock.patch.object(
            liquid_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
        mock.patch.object(model_setup, "download_from_hf", new=liquid_setup.download_from_hf),
    ):
        liquid_setup.setup_liquid(["230m"])

    downloaded = [call.args[1] for call in download.call_args_list]
    assert "transformer_prefill.vmfb" not in downloaded
    assert "transformer_prefill.vmfb" not in _manifest(base_dir / repo_id)["files"]


def test_liquid_prefill_untracking_persists_across_setup_runs(tmp_path):
    """Flag on, then off: prefill leaves the manifest, the local file is kept,
    and later runs neither check nor re-download it."""
    base_dir = tmp_path
    repo_id = liquid_setup._HF_REPO_MAP["230m"]
    prefill_path = base_dir / repo_id / "transformer_prefill.vmfb"

    def exists(_repo_id, filename, revision=None):
        return filename in {
            "transformer.vmfb",
            "lm_head.vmfb",
            "transformer_prefill.vmfb",
        }

    def run_setup(enable_prefill: bool):
        with (
            mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
            mock.patch.object(liquid_setup, "check_requirements"),
            mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
            mock.patch.object(liquid_setup, "hf_file_exists", side_effect=exists),
            mock.patch.object(model_setup, "hf_file_exists", new=liquid_setup.hf_file_exists),
            mock.patch.object(
                liquid_setup,
                "download_from_hf",
                side_effect=_fake_download(base_dir),
            ) as download,
            mock.patch.object(model_setup, "download_from_hf", new=liquid_setup.download_from_hf),
        ):
            liquid_setup.setup_liquid(["230m"], enable_prefill=enable_prefill)
        return download

    run_setup(True)
    assert prefill_path.exists()
    assert "transformer_prefill.vmfb" in _manifest(base_dir / repo_id)["files"]

    download = run_setup(False)
    download.assert_not_called()
    assert "transformer_prefill.vmfb" not in _manifest(base_dir / repo_id)["files"]
    assert prefill_path.exists()  # local file kept, just untracked

    download = run_setup(False)
    download.assert_not_called()
    assert "transformer_prefill.vmfb" not in _manifest(base_dir / repo_id)["files"]


def test_liquid_prefill_added_to_complete_copy_with_flag(tmp_path):
    """``--with-batch-prefill`` on an already-complete copy fetches only the prefill build."""
    base_dir = tmp_path
    repo_id = liquid_setup._HF_REPO_MAP["230m"]

    def exists(_repo_id, filename, revision=None):
        return filename in {
            "transformer.vmfb",
            "lm_head.vmfb",
            "transformer_prefill.vmfb",
        }

    def run_setup(enable_prefill: bool):
        with (
            mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
            mock.patch.object(liquid_setup, "check_requirements"),
            mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
            mock.patch.object(liquid_setup, "hf_file_exists", side_effect=exists),
            mock.patch.object(model_setup, "hf_file_exists", new=liquid_setup.hf_file_exists),
            mock.patch.object(
                liquid_setup,
                "download_from_hf",
                side_effect=_fake_download(base_dir),
            ) as download,
            mock.patch.object(model_setup, "download_from_hf", new=liquid_setup.download_from_hf),
        ):
            liquid_setup.setup_liquid(["230m"], enable_prefill=enable_prefill)
        return download

    run_setup(False)
    download = run_setup(True)

    downloaded = [call.args[1] for call in download.call_args_list]
    assert "transformer_prefill.vmfb" in downloaded
    # The intact model set is not re-fetched (the hook is incremental).
    assert "transformer.vmfb" not in downloaded
    assert "lm_head.vmfb" not in downloaded
    assert "transformer_prefill.vmfb" in _manifest(base_dir / repo_id)["files"]


def test_liquid_prefill_flag_is_noop_for_native_single_file_repo(tmp_path, caplog):
    """Flag on a repo that publishes no transformer_prefill.vmfb: that layout is
    a native single-file build with the prefill family compiled into
    transformer.vmfb, so nothing extra downloads, no warning is raised, and
    a later run treats the copy as complete (no re-download loop)."""
    base_dir = tmp_path
    repo_id = liquid_setup._HF_REPO_MAP["230m"]

    def exists(_repo_id, filename, revision=None):
        return filename in {"transformer.vmfb", "lm_head.vmfb"}

    def run_setup(enable_prefill: bool):
        with (
            mock.patch.object(model_setup, "default_models_dir", return_value=base_dir),
            mock.patch.object(liquid_setup, "check_requirements"),
            mock.patch.object(model_setup, "get_hf_revision", return_value=_REVISION),
            mock.patch.object(liquid_setup, "hf_file_exists", side_effect=exists),
            mock.patch.object(model_setup, "hf_file_exists", new=liquid_setup.hf_file_exists),
            mock.patch.object(
                liquid_setup,
                "download_from_hf",
                side_effect=_fake_download(base_dir),
            ) as download,
            mock.patch.object(model_setup, "download_from_hf", new=liquid_setup.download_from_hf),
        ):
            liquid_setup.setup_liquid(["230m"], enable_prefill=enable_prefill)
        return download

    with caplog.at_level(logging.INFO, logger="utils.model_setup"):
        download = run_setup(True)

    downloaded = [call.args[1] for call in download.call_args_list]
    assert "transformer_prefill.vmfb" not in downloaded
    assert "transformer_prefill.vmfb" not in _manifest(base_dir / repo_id)["files"]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any("nothing extra to download" in r.getMessage() for r in caplog.records)

    # Second run: the copy is complete — nothing downloads.
    download = run_setup(True)
    download.assert_not_called()


# ── prefill opt-in shape ─────────────────────────────────────────────────────


def _load_setup_module(*path_parts):
    """Load a demo's setup_demo.py by path (for non-importable demo dirs)."""
    path = Path(__file__).resolve().parents[1] / Path(*path_parts) / "setup_demo.py"
    modname = "_test_" + "_".join(path_parts).replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefill_is_opt_in_per_demo():
    """Every text LLM demo exposes the shared prefill option."""
    liquidvl_int8_setup = _load_setup_module("LiquidAI", "LiquidAI-LFM2-VL-450M")
    checks = [
        (gemma_setup.setup_gemma3, True),
        (functiongemma_setup.setup_functiongemma, True),
        (qwen25_setup.setup_qwen25, True),
        (qwen3_setup.setup_qwen3, True),
        (liquid_setup.setup_liquid, True),
        (moonshine_setup.setup_moonshine, False),
        (moonshine_streaming_setup.setup_moonshine_streaming, False),
        (liquidvl_int8_setup.setup_liquidvl_int8, False),
        (object_detection_setup.setup_object_detection, False),
        (pose_setup.setup_pose_estimation, False),
    ]
    for setup_fn, supports_prefill in checks:
        params = inspect.signature(setup_fn).parameters
        assert ("enable_prefill" in params) is supports_prefill, setup_fn.__name__


def test_demo_main_prefill_flag_only_when_opted_in():
    """``demo_main`` exposes ``--with-batch-prefill`` (and forwards ``enable_prefill``)
    only for demos that pass ``supports_prefill=True``."""
    calls: list[dict] = []

    def fake_setup(models, **kwargs):
        calls.append(kwargs)

    # Opted in: the flag is accepted and forwarded.
    with mock.patch.object(sys, "argv", ["setup_demo.py", "--with-batch-prefill"]):
        model_setup.demo_main(
            fake_setup,
            description="d",
            default_models=[],
            repo_map={},
            supports_prefill=True,
        )
    assert calls == [{"model_version": None, "no_update": False, "enable_prefill": True}]

    # Not opted in: the flag is unknown and setup_fn never runs.
    calls.clear()
    with mock.patch.object(sys, "argv", ["setup_demo.py", "--with-batch-prefill"]):
        with pytest.raises(SystemExit):
            model_setup.demo_main(
                fake_setup, description="d", default_models=[], repo_map={}
            )
    assert not calls

    # Not opted in, no flag: the plain call is unchanged.
    with mock.patch.object(sys, "argv", ["setup_demo.py"]):
        model_setup.demo_main(
            fake_setup, description="d", default_models=[], repo_map={}
        )
    assert calls == [{"model_version": None, "no_update": False}]


def test_setup_demos_prefill_forwarded_only_to_capable_demos(tmp_path, caplog):
    """``setup_demos.py`` forwards ``enable_prefill`` only to the prefill-capable
    demos; for every other demo the call is unchanged and just a warning is
    logged."""
    import setup_demos

    with mock.patch.object(gemma_setup, "setup_gemma3") as gemma_mock:
        setup_demos.setup_demo("gemma3", enable_prefill=True)
    gemma_mock.assert_called_once_with(
        [gemma_setup.DEFAULT_MODEL], model_version=None, no_update=False, enable_prefill=True
    )

    with (
        caplog.at_level(logging.WARNING),
        mock.patch.object(moonshine_setup, "setup_moonshine") as moonshine_mock,
    ):
        setup_demos.setup_demo("moonshine", enable_prefill=True)
    moonshine_mock.assert_called_once_with(
        ["tiny-en"], model_version=None, no_update=False
    )
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "--with-batch-prefill" in r.getMessage() and "moonshine" in r.getMessage()
        for r in warnings
    )


def test_setup_demos_pose_forwards_version_flags():
    """The root dispatch forwards ``--model-version``/``--no-update`` to
    ``pose_estimation`` like every other tracked demo (the pose branch used to
    call ``setup_pose_estimation()`` bare, silently dropping both flags)."""
    import setup_demos

    with mock.patch.object(pose_setup, "setup_pose_estimation") as pose_mock:
        setup_demos.setup_demo("pose_estimation", model_version="v2.2.1", no_update=True)
    pose_mock.assert_called_once_with(model_version="v2.2.1", no_update=True)

    with mock.patch.object(pose_setup, "setup_pose_estimation") as pose_mock:
        setup_demos.setup_demo("pose_estimation")
    pose_mock.assert_called_once_with(model_version=None, no_update=False)


@pytest.mark.parametrize("setup_fn,ensure_fn,assets", [
    (gemma_setup.setup_gemma3, gemma_setup.ensure_gemma3_models, gemma_setup),
    (functiongemma_setup.setup_functiongemma, functiongemma_setup.ensure_functiongemma_models, gemma_setup),
    (qwen25_setup.setup_qwen25, qwen25_setup.ensure_qwen25_models, qwen25_setup),
    (qwen3_setup.setup_qwen3, qwen3_setup.ensure_qwen3_models, qwen3_setup),
    (liquid_setup.setup_liquid, liquid_setup.ensure_liquid_models, liquid_setup),
], ids=["gemma3", "functiongemma", "qwen2_5", "qwen3", "lfm2.5"])
@pytest.mark.parametrize("sibling", [False, True])
def test_llm_prefill_setup_and_refresh(tmp_path, monkeypatch, setup_fn, ensure_fn, assets, sibling):
    repo_id = "org/test-llm"
    prefill = "transformer_prefill.vmfb"
    model_dir = tmp_path / repo_id
    exists = mock.Mock(side_effect=lambda repo, filename, revision=None: filename != prefill or sibling)
    download = mock.Mock(side_effect=_fake_download(tmp_path))
    for module in (model_setup, assets):
        monkeypatch.setattr(module, "hf_file_exists", exists)
        monkeypatch.setattr(module, "download_from_hf", download)
    monkeypatch.setattr(model_setup, "default_models_dir", lambda: tmp_path)
    monkeypatch.setattr(model_setup, "get_hf_revision", lambda *args, **kwargs: _REVISION)
    monkeypatch.setattr(model_setup, "check_requirements", lambda path: None)
    if assets is liquid_setup:
        monkeypatch.setattr(assets, "check_requirements", lambda path: None)

    setup_fn([repo_id], model_version=_PINS)
    assert not (model_dir / prefill).exists()
    setup_fn([repo_id], model_version=_PINS, enable_prefill=True)
    assert (model_dir / prefill).exists() is sibling
    assert (prefill in _manifest(model_dir)["files"]) is sibling

    download.reset_mock()
    setup_fn([repo_id], model_version=_PINS, enable_prefill=True)
    download.assert_not_called()
    if not sibling:
        assert exists.call_args_list[-1].kwargs["revision"] == _REVISION

    if sibling:
        (model_dir / prefill).unlink()
        ensure_fn(model_dir)
        assert (model_dir / prefill).exists()
        assert prefill in _manifest(model_dir)["files"]

    setup_fn([repo_id], model_version=_PINS, enable_prefill=False)
    assert (model_dir / prefill).exists() is sibling
    assert prefill not in _manifest(model_dir)["files"]
    ensure_fn(model_dir)
    assert prefill not in _manifest(model_dir)["files"]


@pytest.mark.parametrize("module,fn", [
    (qwen25_setup, "setup_qwen25"), (qwen3_setup, "setup_qwen3"),
    (functiongemma_setup, "setup_functiongemma"),
])
def test_root_setup_forwards_prefill_to_all_text_llms(module, fn, caplog):
    import setup_demos

    name = module.__package__
    with mock.patch.object(module, fn) as setup:
        setup_demos.setup_demo(name, enable_prefill=True)
    assert setup.call_args.kwargs["enable_prefill"] is True
    assert not caplog.records


@pytest.mark.parametrize("offline", [False, True])
def test_prefill_completeness_at_tracked_revision(tmp_path, monkeypatch, offline):
    repo_id = "org/test-llm"
    model_dir = tmp_path / repo_id
    model_dir.mkdir(parents=True)
    (model_dir / "transformer.vmfb").touch()
    write_manifest(model_dir, repo_id, ["transformer.vmfb"], version=_PINS, revision=_REVISION)
    exists = mock.Mock(side_effect=ConnectionError("offline")) if offline else mock.Mock(return_value=False)
    monkeypatch.setattr(model_setup, "hf_file_exists", exists)
    monkeypatch.setattr(model_setup, "get_hf_revision", lambda *args, **kwargs: _REVISION)
    download = mock.Mock()
    model_setup.refresh_model(
        repo_id, model_dir, tmp_path, version=_PINS,
        files_present=lambda path: True, download=download, enable_prefill=True,
    )
    exists.assert_called_once_with(repo_id, "transformer_prefill.vmfb", revision=_REVISION)
    download.assert_not_called()


def test_no_update_downloads_prefill_without_manifest(tmp_path, monkeypatch):
    repo_id = "org/test-llm"
    download = _fake_download(tmp_path)
    monkeypatch.setattr(model_setup, "hf_file_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(model_setup, "download_from_hf", download)
    def core(repo, base_dir, *, revision):
        download(repo, "transformer.vmfb", base_dir=base_dir, revision=revision)
        return ["transformer.vmfb"]
    model_setup.download_models(
        {}, [repo_id], label="test", base_dir=tmp_path, no_update=True,
        files_present=lambda path: False, download=core, enable_prefill=True,
    )
    assert (tmp_path / repo_id / "transformer_prefill.vmfb").exists()
    assert not (tmp_path / repo_id / ".manifest.json").exists()
