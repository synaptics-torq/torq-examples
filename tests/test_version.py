# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import utils.version
from utils.download import DownloadError
from utils.version import examples_version, parse_model_specs, resolve_model_version


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
