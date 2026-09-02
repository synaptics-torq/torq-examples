# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import json
from pathlib import Path
from unittest import mock

from gemma3 import setup_demo as gemma_setup
from moonshine import setup_demo as moonshine_setup
from object_detection import setup_demo as object_detection_setup
from pose_estimation import setup_demo as pose_setup
from utils.download import write_manifest

_REVISION = "abc123"


def _fake_download(default_base_dir: Path):
    def download(repo_id: str, filename: str, *, base_dir: Path | None = None, revision: str | None = None):
        root = Path(base_dir) if base_dir is not None else default_base_dir
        path = root / repo_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
        return path

    return download


def _make_moonshine_copy(base_dir, repo_id, revision):
    model_dir = base_dir / repo_id
    for filename in moonshine_setup._MOONSHINE_REQUIRED_FILES:
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old")
    write_manifest(
        model_dir,
        repo_id,
        list(moonshine_setup._MOONSHINE_REQUIRED_FILES),
        revision=revision,
    )
    return model_dir


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
    write_manifest(model_dir, repo_id, files, revision=_REVISION)

    with (
        mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(gemma_setup, "check_requirements"),
        mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "download_from_hf") as download,
    ):
        gemma_setup.setup_gemma3(["instruct"])

    download.assert_not_called()


def test_gemma_repairs_incomplete_download_and_records_lut(tmp_path):
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = base_dir / repo_id
    model_dir.mkdir(parents=True)
    (model_dir / "model.vmfb.trim").write_text("model")
    # Manifest matches upstream revision but required files are missing,
    # so this is an "incomplete" (resumable) state, not a stale one.
    write_manifest(model_dir, repo_id, ["model.vmfb.trim"], revision=_REVISION)

    def exists(_repo_id, filename, revision=None):
        assert _repo_id == repo_id
        return filename == gemma_setup._GEMMA3_TRIM_LUT_FILENAME

    with (
        mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(gemma_setup, "check_requirements"),
        mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "_hf_file_exists", side_effect=exists),
        mock.patch.object(
            gemma_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
    ):
        gemma_setup.setup_gemma3(["instruct"])

    # The existing model.vmfb.trim is preserved (not re-downloaded).
    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == [
        *gemma_setup._GEMMA3_REQUIRED_FILES,
        gemma_setup._GEMMA3_TRIM_LUT_FILENAME,
    ]
    assert (model_dir / "model.vmfb.trim").exists()
    manifest = json.loads((model_dir / ".manifest.json").read_text())
    assert manifest["files"] == [
        "config.json",
        "model.vmfb.trim",
        "token_embeddings.npy",
        "token_id_lut.npy",
        "tokenizer.json",
    ]
    assert manifest["revision"] == _REVISION


def test_gemma_downloads_split_lm_head_pair(tmp_path):
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = base_dir / repo_id

    def exists(_repo_id, filename, revision=None):
        assert _repo_id == repo_id
        return filename in {"transformer.vmfb", "lm_head.vmfb.trim"}

    with (
        mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(gemma_setup, "check_requirements"),
        mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "_hf_file_exists", side_effect=exists),
        mock.patch.object(
            gemma_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
    ):
        gemma_setup.setup_gemma3(["instruct"])

    downloaded = [call.args[1] for call in download.call_args_list]
    expected_files = [
        "transformer.vmfb",
        "lm_head.vmfb.trim",
        *gemma_setup._GEMMA3_REQUIRED_FILES,
    ]
    assert downloaded == expected_files
    manifest = json.loads((model_dir / ".manifest.json").read_text())
    assert manifest["files"] == sorted(expected_files)
    assert manifest["revision"] == _REVISION


def test_gemma_repairs_existing_split_body_by_fetching_lm_head(tmp_path):
    base_dir = tmp_path
    repo_id = gemma_setup.GEMMA3_HF_REPO_MAP["instruct"]
    model_dir = base_dir / repo_id
    model_dir.mkdir(parents=True)
    (model_dir / "transformer.vmfb").write_text("model")
    write_manifest(model_dir, repo_id, ["transformer.vmfb"], revision=_REVISION)

    def exists(_repo_id, filename, revision=None):
        assert _repo_id == repo_id
        return filename in {"transformer.vmfb", "lm_head.vmfb"}

    with (
        mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(gemma_setup, "check_requirements"),
        mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "_hf_file_exists", side_effect=exists),
        mock.patch.object(
            gemma_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
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
    manifest = json.loads((model_dir / ".manifest.json").read_text())
    assert manifest["files"] == sorted(expected_files)
    assert manifest["revision"] == _REVISION


def test_moonshine_skips_when_revision_matches(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = base_dir / repo_id
    files = list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    for filename in files:
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
    write_manifest(model_dir, repo_id, files, revision=_REVISION)

    with (
        mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(moonshine_setup, "check_requirements"),
        mock.patch.object(moonshine_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"])

    download.assert_not_called()


def test_moonshine_downloads_required_files_and_records_revision(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = base_dir / repo_id

    with (
        mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(moonshine_setup, "check_requirements"),
        mock.patch.object(moonshine_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(
            moonshine_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"])

    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    manifest = json.loads((model_dir / ".manifest.json").read_text())
    assert manifest["files"] == sorted(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    assert manifest["revision"] == _REVISION


def test_moonshine_refreshes_and_clears_stale_files_on_revision_change(tmp_path):
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
        revision="old-revision",
    )

    with (
        mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(moonshine_setup, "check_requirements"),
        mock.patch.object(moonshine_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(
            moonshine_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"])

    # All required files re-downloaded after the stale dir was cleared.
    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    # The dropped file is gone and content was refreshed.
    assert not (model_dir / "preprocessor.onnx").exists()
    assert (model_dir / "encoder.vmfb").read_text() == "encoder.vmfb"
    manifest = json.loads((model_dir / ".manifest.json").read_text())
    assert manifest["revision"] == _REVISION
    assert "preprocessor.onnx" not in manifest["files"]


def test_inference_refreshes_stale_models(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_moonshine_copy(base_dir, repo_id, "old-revision")

    with (
        mock.patch.object(moonshine_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(
            moonshine_setup,
            "download_from_hf",
            side_effect=_fake_download(base_dir),
        ) as download,
    ):
        # model_dir is base/<repo_id>; the helper must recover base_dir.
        moonshine_setup.ensure_moonshine_models(model_dir)

    downloaded = [call.args[1] for call in download.call_args_list]
    assert downloaded == list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    assert (model_dir / "encoder.vmfb").read_text() == "encoder.vmfb"
    manifest = json.loads((model_dir / ".manifest.json").read_text())
    assert manifest["revision"] == _REVISION


def test_inference_no_refresh_skips_network(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_moonshine_copy(base_dir, repo_id, "old-revision")

    with (
        mock.patch.object(moonshine_setup, "get_hf_revision") as revision,
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir, refresh=False)

    revision.assert_not_called()
    download.assert_not_called()


def test_inference_does_not_refresh_a_pinned_model(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = _make_moonshine_copy(base_dir, repo_id, "pinned-revision")
    write_manifest(
        model_dir,
        repo_id,
        list(moonshine_setup._MOONSHINE_REQUIRED_FILES),
        revision="pinned-revision",
        auto_update=False,
    )

    with (
        mock.patch.object(moonshine_setup, "get_hf_revision") as revision,
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir)

    revision.assert_not_called()
    download.assert_not_called()


def test_yolo_inference_does_not_refresh_a_pinned_model(tmp_path):
    repo_id = object_detection_setup._OD_HF_REPO_MAP["nano"]
    model_dir = tmp_path / repo_id
    for filename in (
        object_detection_setup._MODEL_FILENAME,
        object_detection_setup._LABELS_FILENAME,
    ):
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
    write_manifest(
        model_dir,
        repo_id,
        [
            object_detection_setup._MODEL_FILENAME,
            object_detection_setup._LABELS_FILENAME,
        ],
        revision="pinned-revision",
        auto_update=False,
    )

    with (
        mock.patch.object(object_detection_setup, "get_hf_revision") as revision,
        mock.patch.object(object_detection_setup, "download_from_hf") as download,
    ):
        object_detection_setup.ensure_object_detection_models(model_dir)

    revision.assert_not_called()
    download.assert_not_called()


def test_pose_inference_does_not_refresh_a_pinned_model(tmp_path):
    repo_id = pose_setup._POSE_HF_REPO
    model_dir = tmp_path / repo_id
    model_path = model_dir / pose_setup._MODEL_FILENAME
    model_path.parent.mkdir(parents=True)
    model_path.write_text(pose_setup._MODEL_FILENAME)
    write_manifest(
        model_dir,
        repo_id,
        [pose_setup._MODEL_FILENAME],
        revision="pinned-revision",
        auto_update=False,
    )

    with (
        mock.patch.object(pose_setup, "get_hf_revision") as revision,
        mock.patch.object(pose_setup, "download_from_hf") as download,
    ):
        pose_setup.ensure_pose_estimation_models(model_dir)

    revision.assert_not_called()
    download.assert_not_called()


def test_inference_without_manifest_does_not_download(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = base_dir / repo_id
    model_dir.mkdir(parents=True)

    with (
        mock.patch.object(moonshine_setup, "get_hf_revision") as revision,
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.ensure_moonshine_models(model_dir)

    revision.assert_not_called()
    download.assert_not_called()


def test_moonshine_offline_uses_local_files(tmp_path):
    base_dir = tmp_path
    repo_id = moonshine_setup.MOONSHINE_HF_REPO_MAP["tiny-en"]
    model_dir = base_dir / repo_id
    files = list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
    for filename in files:
        path = model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
    write_manifest(model_dir, repo_id, files, revision="old-revision")

    with (
        mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
        mock.patch.object(moonshine_setup, "check_requirements"),
        mock.patch.object(moonshine_setup, "get_hf_revision", return_value=None),
        mock.patch.object(moonshine_setup, "download_from_hf") as download,
    ):
        moonshine_setup.setup_moonshine(["tiny-en"])

    download.assert_not_called()


def _make_gemma_split_copy(model_dir: Path, lm_head_name: str, manifest_head: str):
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
        revision=_REVISION,
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
        mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "_hf_file_exists") as exists,
        mock.patch.object(gemma_setup, "download_from_hf") as download,
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
        mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
        mock.patch.object(gemma_setup, "_hf_file_exists", return_value=True),
        mock.patch.object(
            gemma_setup, "download_from_hf", side_effect=_fake_download(tmp_path)
        ) as download,
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
        mock.patch.object(gemma_setup, "get_hf_revision", return_value="new-revision"),
        mock.patch.object(gemma_setup, "_hf_file_exists") as exists,
        mock.patch.object(gemma_setup, "download_from_hf") as download,
    ):
        gemma_setup.ensure_gemma3_models(model_dir)

    exists.assert_not_called()
    download.assert_not_called()
    # The stale-refresh path must not have cleared the dir it cannot replace.
    assert (model_dir / "transformer.vmfb").exists()
    assert not (tmp_path / "models" / "Synaptics").exists()
