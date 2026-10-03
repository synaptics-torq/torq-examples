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
    get_hf_file_info as real_get_hf_file_info,
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


# ── per-file tracking ───────────────────────────────────────────────────
#
# tests/conftest.py stubs utils.download.get_hf_file_info to return None for
# every test by default (network-free); the tests below patch it explicitly.


def _make_tracked(model_dir, files, file_info, *, version="v2.1.0", revision="old-sha"):
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        (model_dir / filename).write_text(filename)
    write_manifest(
        model_dir, "org/repo", list(files),
        version=version, revision=revision, file_info=file_info,
    )
    return model_dir


def _fake_download(model_dir, files):
    """Stand-in for the demos' download hooks: fetch what's missing only."""
    def download():
        for filename in files:
            path = model_dir / filename
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(filename + "@new")
        return list(files)
    return download


def test_write_manifest_records_file_info_for_tracked_files(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.vmfb").write_text("a")
    (model_dir / "b.json").write_text("b")

    write_manifest(
        model_dir, "org/repo", ["a.vmfb", "b.json"],
        version="v2.1.0", revision="sha",
        file_info={
            "a.vmfb": {"size": 1, "sha256": "aaa"},
            "b.json": {"size": 2, "sha256": None},
            "untracked.bin": {"size": 9, "sha256": "zzz"},  # not tracked -> dropped
        },
    )

    assert read_manifest(model_dir)["file_info"] == {
        "a.vmfb": {"size": 1, "sha256": "aaa"},
        "b.json": {"size": 2, "sha256": None},
    }


def test_write_manifest_omits_file_info_when_not_given(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.vmfb").write_text("a")

    write_manifest(model_dir, "org/repo", ["a.vmfb"])

    assert "file_info" not in read_manifest(model_dir)


def test_get_hf_file_info_parses_tree_items():
    pytest.importorskip("huggingface_hub")

    lfs_file = mock.Mock(path="a.vmfb", type="file", size=10)
    lfs_file.lfs = mock.Mock(sha256="aaa")
    plain_file = mock.Mock(path="b.json", type="file", size=2)
    plain_file.lfs = None
    subdir = mock.Mock(path="samples", type="directory", size=None)
    api = mock.Mock()
    api.list_repo_tree.return_value = [lfs_file, plain_file, subdir]

    with mock.patch("huggingface_hub.HfApi", return_value=api):
        info = real_get_hf_file_info("org/repo", revision="sha")
    assert info == {
        "a.vmfb": {"size": 10, "sha256": "aaa"},
        "b.json": {"size": 2, "sha256": None},
    }

    with mock.patch("huggingface_hub.HfApi", return_value=api):
        info = real_get_hf_file_info("org/repo", revision="sha", filenames={"a.vmfb"})
    assert set(info) == {"a.vmfb"}


def test_get_hf_file_info_returns_none_when_offline():
    pytest.importorskip("huggingface_hub")

    with mock.patch("huggingface_hub.HfApi") as api:
        api.return_value.list_repo_tree.side_effect = ConnectionError("offline")
        assert real_get_hf_file_info("org/repo", revision="sha") is None


def test_ensure_model_stale_refreshes_only_changed_files(tmp_path):
    model_dir = _make_tracked(
        tmp_path / "m", ["a.vmfb", "b.vmfb"],
        {"a.vmfb": {"size": 1, "sha256": "aaa"}, "b.vmfb": {"size": 2, "sha256": "bbb"}},
    )
    new_info = {
        "a.vmfb": {"size": 1, "sha256": "aaa"},   # unchanged upstream
        "b.vmfb": {"size": 9, "sha256": "ccc"},   # changed upstream
    }

    with mock.patch("utils.download.get_hf_file_info", return_value=dict(new_info)):
        status = ensure_model(
            model_dir, "org/repo", files_present=True,
            version="v2.1.0", revision="new-sha",
            download=_fake_download(model_dir, ["a.vmfb", "b.vmfb"]),
        )

    assert status is ModelStatus.STALE
    assert (model_dir / "a.vmfb").read_text() == "a.vmfb"      # unchanged: kept
    assert (model_dir / "b.vmfb").read_text() == "b.vmfb@new"  # changed: re-fetched
    manifest = read_manifest(model_dir)
    assert manifest["revision"] == "new-sha"
    assert manifest["file_info"] == new_info


def test_ensure_model_stale_refetches_non_lfs_files_without_hashes(tmp_path):
    # Non-LFS files (no sha256) cannot be proven identical on a tag move, so
    # they are re-fetched even when the size is unchanged.
    model_dir = _make_tracked(
        tmp_path / "m", ["a.vmfb", "b.json"],
        {"a.vmfb": {"size": 1, "sha256": "aaa"}, "b.json": {"size": 2, "sha256": None}},
    )
    new_info = {
        "a.vmfb": {"size": 1, "sha256": "aaa"},
        "b.json": {"size": 2, "sha256": None},
    }

    with mock.patch("utils.download.get_hf_file_info", return_value=dict(new_info)):
        status = ensure_model(
            model_dir, "org/repo", files_present=True,
            version="v2.1.0", revision="new-sha",
            download=_fake_download(model_dir, ["a.vmfb", "b.json"]),
        )

    assert status is ModelStatus.STALE
    assert (model_dir / "a.vmfb").read_text() == "a.vmfb"
    assert (model_dir / "b.json").read_text() == "b.json@new"


def test_ensure_model_stale_keeps_files_dropped_upstream(tmp_path):
    model_dir = _make_tracked(
        tmp_path / "m", ["a.vmfb", "gone.vmfb"],
        {"a.vmfb": {"size": 1, "sha256": "aaa"}, "gone.vmfb": {"size": 3, "sha256": "ggg"}},
    )

    with mock.patch("utils.download.get_hf_file_info", return_value={
        "a.vmfb": {"size": 1, "sha256": "aaa"},  # gone.vmfb no longer published
    }):
        status = ensure_model(
            model_dir, "org/repo", files_present=True,
            version="v2.1.0", revision="new-sha",
            download=_fake_download(model_dir, ["a.vmfb"]),
        )

    assert status is ModelStatus.STALE
    assert (model_dir / "a.vmfb").read_text() == "a.vmfb"
    assert (model_dir / "gone.vmfb").read_text() == "gone.vmfb"  # kept on disk
    assert read_manifest(model_dir)["files"] == ["a.vmfb"]       # but untracked


def test_ensure_model_stale_legacy_manifest_clears_everything(tmp_path):
    # A manifest without file_info (pre-T7 copy) falls back to the old
    # behaviour: clear the directory and re-download everything.
    model_dir = _make_copy(tmp_path / "m", "org/repo", ["a.vmfb"], version="v2.1.0", revision="old-sha")

    with mock.patch("utils.download.get_hf_file_info", return_value=None) as gi:
        status = ensure_model(
            model_dir, "org/repo", files_present=True,
            version="v2.1.0", revision="new-sha",
            download=_fake_download(model_dir, ["a.vmfb"]),
        )

    assert status is ModelStatus.STALE
    assert (model_dir / "a.vmfb").read_text() == "a.vmfb@new"  # full re-download
    # No API call for the stale check (legacy manifest short-circuits it); the
    # single call is the post-download manifest write, which records per-file
    # hashes when the Hub is reachable (offline here -> manifest stays legacy).
    assert gi.call_count == 1
    assert gi.call_args.kwargs == {"revision": "new-sha", "filenames": {"a.vmfb"}}
    assert "file_info" not in read_manifest(model_dir)


def test_ensure_model_stale_offline_clears_everything(tmp_path):
    # Tracked hashes but the Hub unreachable: cannot verify per-file, so
    # fall back to the full re-download.
    model_dir = _make_tracked(
        tmp_path / "m", ["a.vmfb"],
        {"a.vmfb": {"size": 1, "sha256": "aaa"}},
    )

    with mock.patch("utils.download.get_hf_file_info", return_value=None):
        status = ensure_model(
            model_dir, "org/repo", files_present=True,
            version="v2.1.0", revision="new-sha",
            download=_fake_download(model_dir, ["a.vmfb"]),
        )

    assert status is ModelStatus.STALE
    assert (model_dir / "a.vmfb").read_text() == "a.vmfb@new"
    assert "file_info" not in read_manifest(model_dir)


def test_ensure_model_incomplete_records_file_info(tmp_path):
    # A repair (same revision, missing file) also upgrades the manifest with
    # per-file hashes.
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.vmfb").write_text("a")
    write_manifest(model_dir, "org/repo", ["a.vmfb", "b.vmfb"], version="v2.1.0", revision="sha")
    new_info = {
        "a.vmfb": {"size": 1, "sha256": "aaa"},
        "b.vmfb": {"size": 2, "sha256": "bbb"},
    }

    with mock.patch("utils.download.get_hf_file_info", return_value=dict(new_info)):
        status = ensure_model(
            model_dir, "org/repo", files_present=False,
            version="v2.1.0", revision="sha",
            download=_fake_download(model_dir, ["a.vmfb", "b.vmfb"]),
        )

    assert status is ModelStatus.INCOMPLETE
    assert (model_dir / "a.vmfb").read_text() == "a"           # untouched
    assert (model_dir / "b.vmfb").read_text() == "b.vmfb@new"  # repaired
    assert read_manifest(model_dir)["file_info"] == new_info


def test_ensure_model_version_adopt_keeps_file_info(tmp_path):
    model_dir = _make_tracked(
        tmp_path / "m", ["a.vmfb"],
        {"a.vmfb": {"size": 1, "sha256": "aaa"}},
        version="v2.0.0", revision="sha",
    )

    with mock.patch("utils.download.get_hf_file_info") as gi:
        status = ensure_model(
            model_dir, "org/repo", files_present=True,
            version="v2.1.0", revision="sha",
            download=_fake_download(model_dir, ["a.vmfb"]),
        )

    assert status is ModelStatus.UP_TO_DATE
    gi.assert_not_called()
    assert read_manifest(model_dir)["file_info"] == {"a.vmfb": {"size": 1, "sha256": "aaa"}}
