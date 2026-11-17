# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from unittest import mock

import utils.version
from utils.download import DownloadError
from utils.version import (
    compatible_version,
    examples_version,
    follow_examples_series,
    parse_model_specs,
    release_series,
    resolve_model_version,
    warn_if_model_incompatible,
)


def test_examples_version_reads_repo_version_file(tmp_path, monkeypatch):
    version_file = tmp_path / "VERSION"
    version_file.write_text("2.1.0\n")
    monkeypatch.setattr(utils.version, "_VERSION_FILE", version_file)

    assert examples_version() == "v2.1.0"


def test_examples_version_accepts_v_prefixed_file(tmp_path, monkeypatch):
    version_file = tmp_path / "VERSION"
    version_file.write_text("v3.0.1\n")
    monkeypatch.setattr(utils.version, "_VERSION_FILE", version_file)

    assert examples_version() == "v3.0.1"


def test_examples_version_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.version, "_VERSION_FILE", tmp_path / "VERSION")

    assert examples_version() is None


def test_examples_version_empty_file_returns_none(tmp_path, monkeypatch):
    version_file = tmp_path / "VERSION"
    version_file.write_text("   \n")
    monkeypatch.setattr(utils.version, "_VERSION_FILE", version_file)

    assert examples_version() is None


def test_repo_ships_a_version_file():
    # The demos rely on this at setup time; a release without it would fail
    # loudly for every built-in repo, so the file must exist in the tree.
    assert utils.version._VERSION_FILE.is_file()
    assert examples_version() is not None


def test_parse_model_specs_plain_names():
    assert parse_model_specs(["tiny-en", "instruct"]) == [
        ("tiny-en", None),
        ("instruct", None),
    ]


def test_parse_model_specs_with_versions():
    assert parse_model_specs(["default:v2.0.0", "custom/gemma3"]) == [
        ("default", "v2.0.0"),
        ("custom/gemma3", None),
    ]


def test_parse_model_specs_custom_repo_with_version():
    assert parse_model_specs(["custom/gemma3:v1.2.3"]) == [("custom/gemma3", "v1.2.3")]


def test_resolve_model_version_explicit_always_wins():
    builtin = frozenset({"Synaptics/repo"})
    assert resolve_model_version("Synaptics/repo", "v2.0.0", builtin_repos=builtin) == "v2.0.0"
    assert resolve_model_version("custom/repo", "v1.0.0", builtin_repos=builtin) == "v1.0.0"


def test_resolve_model_version_custom_repo_defaults_to_head():
    assert resolve_model_version("custom/repo", None, builtin_repos=frozenset({"Synaptics/repo"})) is None


def test_resolve_model_version_builtin_defaults_to_examples_version():
    builtin = frozenset({"Synaptics/repo"})
    assert resolve_model_version("Synaptics/repo", None, builtin_repos=builtin) == examples_version()


def test_resolve_model_version_builtin_without_version_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.version, "_VERSION_FILE", tmp_path / "VERSION")
    builtin = frozenset({"Synaptics/repo"})

    try:
        resolve_model_version("Synaptics/repo", None, builtin_repos=builtin)
        raise AssertionError("expected DownloadError")
    except DownloadError as exc:
        assert "VERSION" in str(exc)


def test_resolve_model_version_explicit_silences_missing_version_file(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.version, "_VERSION_FILE", tmp_path / "VERSION")
    builtin = frozenset({"Synaptics/repo"})

    assert resolve_model_version("Synaptics/repo", "v2.0.0", builtin_repos=builtin) == "v2.0.0"


def test_compatible_version_prefers_exact_tag_then_newest_of_series():
    tags = ["v2.2.9", "v2.3.0", "v2.3.4", "v2.3.10", "v2.4.0", "v2.3.x", "main-2.3.11"]
    with mock.patch("utils.version.list_hf_tags", return_value=tags):
        assert compatible_version("Synaptics/repo", "v2.3.4") == "v2.3.4"
        # Patch level is ignored both ways: newer and older patches are compatible.
        assert compatible_version("Synaptics/repo", "v2.3.5") == "v2.3.10"
        assert compatible_version("Synaptics/repo", "v2.2.0") == "v2.2.9"


def test_compatible_version_none_without_a_match():
    with mock.patch("utils.version.list_hf_tags", return_value=["v2.2.0"]):
        assert compatible_version("Synaptics/repo", "v2.3.0") is None  # no tag of the series
        assert compatible_version("Synaptics/repo", "q4") is None  # not X.Y.Z
    with mock.patch("utils.version.list_hf_tags", return_value=None):
        assert compatible_version("Synaptics/repo", "v2.3.0") is None  # Hub unreachable


def test_release_series():
    assert release_series("v2.3.4") == ("2", "3")
    assert release_series("2.3.0rc1") == ("2", "3")
    assert release_series("q4") is None
    assert release_series(None) is None


def test_follow_examples_series(tmp_path, monkeypatch, caplog):
    version_file = tmp_path / "VERSION"
    version_file.write_text("3.0.1\n")
    monkeypatch.setattr(utils.version, "_VERSION_FILE", version_file)
    with mock.patch("utils.version.list_hf_tags", return_value=["v2.3.0", "v3.0.0", "v3.0.1"]) as tags:
        assert follow_examples_series("Synaptics/repo", "v2.3.0") == "v3.0.1"
        tags.reset_mock()
        # Same series (any patch) and versions without a series are left alone.
        assert follow_examples_series("Synaptics/repo", "v3.0.0") == "v3.0.0"
        assert follow_examples_series("Synaptics/repo", "q4") == "q4"
        assert follow_examples_series("Synaptics/repo", None) is None
        tags.assert_not_called()
    with mock.patch("utils.version.list_hf_tags", return_value=["v2.3.0"]):
        assert follow_examples_series("Synaptics/repo", "v2.3.0") == "v2.3.0"
    assert "has no compatible model version yet" in caplog.text


def _set_version(tmp_path, monkeypatch, version):
    version_file = tmp_path / "VERSION"
    version_file.write_text(f"{version}\n")
    monkeypatch.setattr(utils.version, "_VERSION_FILE", version_file)


def test_warn_if_model_incompatible_ignores_patch(tmp_path, monkeypatch, caplog):
    _set_version(tmp_path, monkeypatch, "2.3.1")
    warn_if_model_incompatible("Synaptics/repo", "v2.3.0")
    warn_if_model_incompatible("Synaptics/repo", "v2.3.9")
    warn_if_model_incompatible("Synaptics/repo", "q4")
    assert not caplog.records
    warn_if_model_incompatible("Synaptics/repo", "v2.2.0")
    assert "model version v2.2.0, which is not compatible" in caplog.text


def test_resolve_model_version_builtin_uses_compatible_tag(tmp_path, monkeypatch):
    version_file = tmp_path / "VERSION"
    version_file.write_text("2.3.4\n")
    monkeypatch.setattr(utils.version, "_VERSION_FILE", version_file)
    builtin = frozenset({"Synaptics/repo"})
    with mock.patch("utils.version.list_hf_tags", return_value=["v2.3.0", "v2.3.1"]):
        assert resolve_model_version("Synaptics/repo", None, builtin_repos=builtin) == "v2.3.1"
        # An explicit version is used verbatim.
        assert resolve_model_version("Synaptics/repo", "v2.3.0", builtin_repos=builtin) == "v2.3.0"
