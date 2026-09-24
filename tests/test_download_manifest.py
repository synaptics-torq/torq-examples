# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import json
from unittest import mock

import pytest

from pathlib import Path

from utils.download import (
    DownloadError,
    ModelStatus,
    ModelVersionNotFoundError,
    base_dir_for,
    check_model_status,
    clear_model_dir,
    ensure_model,
    get_hf_revision,
    read_manifest,
    verify_manifest,
    write_manifest,
)


def _make_copy(model_dir, repo_id, files, *, version=None, revision=None):
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        (model_dir / filename).write_text(filename)
    write_manifest(model_dir, repo_id, list(files), version=version, revision=revision)
    return model_dir


def test_write_read_and_verify_manifest(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "nested").mkdir(parents=True)
    (model_dir / "a.vmfb").write_text("a")
    (model_dir / "nested" / "b.json").write_text("b")

    manifest_path = write_manifest(
        model_dir,
        "org/repo",
        ["nested/b.json", "a.vmfb"],
        version="v2.1.0",
        revision="deadbeef",
    )

    assert manifest_path.name == ".manifest.json"
    manifest = read_manifest(model_dir)
    assert manifest is not None
    assert manifest["repo_id"] == "org/repo"
    assert manifest["version"] == "v2.1.0"
    assert manifest["revision"] == "deadbeef"
    assert manifest["files"] == ["a.vmfb", "nested/b.json"]
    assert "auto_update" not in manifest
    assert verify_manifest(model_dir)


def test_write_manifest_defaults_to_unversioned(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "a.vmfb").write_text("a")

    write_manifest(model_dir, "org/repo", ["a.vmfb"])

    manifest = read_manifest(model_dir)
    assert manifest["version"] is None
    assert manifest["revision"] is None
    assert verify_manifest(model_dir)


def test_verify_manifest_rejects_missing_manifest(tmp_path):
    assert read_manifest(tmp_path) is None
    assert not verify_manifest(tmp_path)


def test_verify_manifest_rejects_corrupt_manifest(tmp_path):
    (tmp_path / ".manifest.json").write_text("{not json")

    assert read_manifest(tmp_path) is None
    assert not verify_manifest(tmp_path)


def test_verify_manifest_rejects_empty_or_missing_files(tmp_path):
    (tmp_path / ".manifest.json").write_text(
        json.dumps({"repo_id": "org/repo", "files": []})
    )
    assert not verify_manifest(tmp_path)

    (tmp_path / ".manifest.json").write_text(
        json.dumps({"repo_id": "org/repo", "files": ["missing.vmfb"]})
    )
    assert not verify_manifest(tmp_path)


def test_base_dir_for_strips_the_repo_id():
    assert base_dir_for(Path("/models/org/repo"), "org/repo") == Path("/models")
    assert base_dir_for(Path("/models/repo"), "repo") == Path("/models")


def test_base_dir_for_rejects_a_layout_that_does_not_end_in_the_repo_id():
    # A bare clone keeps only the repo name, so the org component is missing.
    assert base_dir_for(Path("/models/repo"), "org/repo") is None
    # A renamed directory cannot be mapped back to the repo either.
    assert base_dir_for(Path("/models/org/other"), "org/repo") is None
    # Shorter than the repo id: nothing to strip.
    assert base_dir_for(Path("/repo"), "org/repo") is None


def test_get_hf_revision_raises_loudly_on_missing_version(tmp_path):
    pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import RevisionNotFoundError

    with mock.patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.side_effect = RevisionNotFoundError(
            "nope", response=mock.Mock()
        )
        try:
            assert get_hf_revision("org/repo", revision="v9.9.9") is None
            raise AssertionError("expected ModelVersionNotFoundError")
        except ModelVersionNotFoundError as exc:
            assert "v9.9.9" in str(exc)
            assert "org/repo" in str(exc)
            assert isinstance(exc, DownloadError)


def test_get_hf_revision_returns_none_when_offline(tmp_path):
    pytest.importorskip("huggingface_hub")

    with mock.patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.side_effect = ConnectionError("offline")
        assert get_hf_revision("org/repo", revision="v1.0.0") is None
        # The metadata check uses the short check timeout, not the download one.
        _, kwargs = api.return_value.model_info.call_args
        assert kwargs["revision"] == "v1.0.0"
        assert "timeout" in kwargs


def test_check_model_status_stale_when_tracked_version_moved(tmp_path):
    model_dir = _make_copy(tmp_path / "m", "org/repo", ["a.vmfb"], version="v2.1.0", revision="old-sha")

    status = check_model_status(
        model_dir, "org/repo", files_present=True, version="v2.1.0", revision="new-sha"
    )
    assert status is ModelStatus.STALE


def test_check_model_status_stale_for_legacy_manifest_without_version(tmp_path):
    # A pre-versioning copy (manifest with only a SHA) is re-fetched once at
    # the tracked version.
    model_dir = _make_copy(tmp_path / "m", "org/repo", ["a.vmfb"], revision="old-sha")

    status = check_model_status(
        model_dir, "org/repo", files_present=True, version="v2.1.0", revision="new-sha"
    )
    assert status is ModelStatus.STALE


def test_check_model_status_up_to_date_when_version_matches(tmp_path):
    model_dir = _make_copy(tmp_path / "m", "org/repo", ["a.vmfb"], version="v2.1.0", revision="sha")

    status = check_model_status(
        model_dir, "org/repo", files_present=True, version="v2.1.0", revision="sha"
    )
    assert status is ModelStatus.UP_TO_DATE


def test_check_model_status_up_to_date_offline_with_files(tmp_path):
    model_dir = _make_copy(tmp_path / "m", "org/repo", ["a.vmfb"], version="v2.1.0", revision="sha")

    status = check_model_status(
        model_dir, "org/repo", files_present=True, version="v2.1.0", revision=None
    )
    assert status is ModelStatus.UP_TO_DATE


def test_check_model_status_incomplete_when_files_missing(tmp_path):
    # Same commit as upstream, but a required file is gone: a repair, not a
    # refresh (the dir must not be cleared).
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.vmfb").write_text("a")
    write_manifest(
        model_dir, "org/repo", ["a.vmfb", "b.vmfb"], version="v2.1.0", revision="sha"
    )

    status = check_model_status(
        model_dir, "org/repo", files_present=False, version="v2.1.0", revision="sha"
    )
    assert status is ModelStatus.INCOMPLETE


def test_check_model_status_no_manifest_with_known_version_is_stale(tmp_path):
    # An empty dir has never been set up; the full download path applies.
    model_dir = tmp_path / "m"
    model_dir.mkdir()

    status = check_model_status(
        model_dir, "org/repo", files_present=False, version="v2.1.0", revision="sha"
    )
    assert status is ModelStatus.STALE


def test_check_model_status_unversioned_never_stale(tmp_path):
    # An unversioned copy (custom repo at HEAD) has no tag to re-sync, so a
    # different HEAD SHA must not mark it stale.
    model_dir = _make_copy(tmp_path / "m", "org/repo", ["a.vmfb"], version=None, revision="sha")

    status = check_model_status(
        model_dir, "org/repo", files_present=True, version=None, revision=None
    )
    assert status is ModelStatus.UP_TO_DATE


def test_ensure_model_re_downloads_when_stale(tmp_path):
    model_dir = _make_copy(tmp_path / "m", "org/repo", ["a.vmfb"], version="v2.1.0", revision="old-sha")

    with mock.patch("utils.download.write_manifest") as write:
        status = ensure_model(
            model_dir,
            "org/repo",
            files_present=True,
            version="v2.1.0",
            revision="new-sha",
            download=lambda: ["a.vmfb", "b.vmfb"],
        )

    assert status is ModelStatus.STALE
    # The stale dir was cleared before re-downloading.
    assert not (model_dir / "a.vmfb").exists()
    assert write.called
    written = write.call_args.args
    assert written[0] == model_dir and written[1] == "org/repo" and written[2] == ["a.vmfb", "b.vmfb"]
    assert write.call_args.kwargs["version"] == "v2.1.0"
    assert write.call_args.kwargs["revision"] == "new-sha"


def test_ensure_model_repairs_incomplete_without_clearing(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.vmfb").write_text("a")
    write_manifest(model_dir, "org/repo", ["a.vmfb", "b.vmfb"], version="v2.1.0", revision="sha")

    status = ensure_model(
        model_dir,
        "org/repo",
        files_present=False,
        version="v2.1.0",
        revision="sha",
        download=lambda: ["a.vmfb", "b.vmfb"],
    )

    assert status is ModelStatus.INCOMPLETE
    # Incomplete is a repair, not a refresh: existing files are kept.
    assert (model_dir / "a.vmfb").read_text() == "a"


def test_ensure_model_adopts_new_version_name_without_downloading(tmp_path):
    # Same commit, different version name: the user explicitly switched
    # versions, so the manifest must track the new name without re-downloading.
    model_dir = _make_copy(tmp_path / "m", "org/repo", ["a.vmfb"], version="v2.0.0", revision="sha")

    with mock.patch("utils.download.clear_model_dir") as clear:
        status = ensure_model(
            model_dir,
            "org/repo",
            files_present=True,
            version="v2.1.0",
            revision="sha",
            download=lambda: ["a.vmfb"],
        )

    assert status is ModelStatus.UP_TO_DATE
    clear.assert_not_called()
    manifest = read_manifest(model_dir)
    assert manifest["version"] == "v2.1.0"
    assert manifest["revision"] == "sha"
    assert manifest["files"] == ["a.vmfb"]


def test_ensure_model_untracked_writes_no_manifest(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()

    status = ensure_model(
        model_dir,
        "org/repo",
        files_present=False,
        version=None,
        revision=None,
        download=lambda: ["a.vmfb"],
        record=False,
    )

    assert status is ModelStatus.INCOMPLETE
    assert not (model_dir / ".manifest.json").exists()


def test_ensure_model_untracked_complete_skips_download(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.vmfb").write_text("a")

    with mock.patch("utils.download.clear_model_dir") as clear:
        status = ensure_model(
            model_dir,
            "org/repo",
            files_present=True,
            version="v2.1.0",  # requested version is irrelevant when untracked
            revision="sha",
            download=lambda: ["a.vmfb"],
            record=False,
        )

    assert status is ModelStatus.UP_TO_DATE
    clear.assert_not_called()
    assert not (model_dir / ".manifest.json").exists()
