# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import json

from utils.download import read_manifest, verify_manifest, write_manifest


def test_write_read_and_verify_manifest(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "nested").mkdir(parents=True)
    (model_dir / "a.vmfb").write_text("a")
    (model_dir / "nested" / "b.json").write_text("b")

    manifest_path = write_manifest(
        model_dir,
        "org/repo",
        ["nested/b.json", "a.vmfb"],
    )

    assert manifest_path.name == ".manifest.json"
    manifest = read_manifest(model_dir)
    assert manifest is not None
    assert manifest["repo_id"] == "org/repo"
    assert manifest["files"] == ["a.vmfb", "nested/b.json"]
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
