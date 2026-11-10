# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Every demo folder that has a setup_demo.py must carry a valid info.json,
and ``setup_demos.DEMOS`` must resolve each demo to a module with a callable
uniform ``setup`` entry point."""

import json
import logging
from pathlib import Path

import setup_demos

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_FIELDS = ("name", "description", "task_category")


def _setup_demo_folders() -> list[str]:
    """Top-level folders (relative to the repo root) that contain a setup_demo.py."""
    return sorted(
        path.parent.relative_to(_REPO_ROOT).as_posix()
        for path in _REPO_ROOT.glob("*/setup_demo.py")
    )


def test_every_setup_demo_folder_has_valid_info_json():
    folders = _setup_demo_folders()
    assert folders, "expected at least one demo folder with a setup_demo.py"
    for folder in folders:
        info_path = _REPO_ROOT / folder / "info.json"
        assert info_path.is_file(), f"{folder} has a setup_demo.py but no info.json"
        info = json.loads(info_path.read_text())
        for field in _REQUIRED_FIELDS:
            assert field in info, f"{folder}/info.json is missing '{field}'"
        assert info["name"], f"{folder}/info.json has an empty 'name'"
        assert info["description"], f"{folder}/info.json has an empty 'description'"
        assert info["validation"], f"{folder}/info.json has no 'validation' datasets"
        for entry in info["validation"]:
            assert {"dataset", "samples"} <= set(entry), f"{folder}/info.json: {entry}"


def test_every_default_dataset_has_a_loader():
    """Each info.json dataset is supported by its category's LOADERS or by the
    demo's own ``src/validate.py`` (which passes demo-specific ``loaders=``)."""
    import importlib

    for folder in _setup_demo_folders():
        info = json.loads((_REPO_ROOT / folder / "info.json").read_text())
        loaders = importlib.import_module(f"utils.tasks.{info['task_category']}").LOADERS
        validate_src = (_REPO_ROOT / folder / "src" / "validate.py").read_text()
        for entry in info["validation"]:
            assert entry["dataset"] in loaders or f'"{entry["dataset"]}"' in validate_src, (
                f"{folder}: no loader for {entry['dataset']}"
            )


def test_discovery_finds_exactly_the_demo_folders():
    assert _setup_demo_folders() == sorted(setup_demos._DEMO_FOLDERS)


def test_demos_resolve_to_callable_setup():
    names = set(setup_demos.DEMOS)
    assert names == {info["name"] for info in setup_demos._DEMO_FOLDERS.values()}
    for name in setup_demos.DEMOS:
        resolved = setup_demos.resolve_demo(name)
        assert resolved is not None, f"DEMOS entry '{name}' does not resolve"
        folder, info = resolved
        assert info["name"] == name
        module = setup_demos._load_demo_setup(folder)
        assert callable(module.setup), f"{folder}/setup_demo.py has no callable 'setup'"


def test_renamed_demo_folders_resolve_with_deprecation_warning(caplog):
    for old_folder, new_folder in setup_demos.DEPRECATED_DEMOS.items():
        with caplog.at_level(logging.WARNING):
            resolved = setup_demos.resolve_demo(old_folder)
        assert resolved is not None, f"renamed demo '{old_folder}' does not resolve"
        assert resolved[0] == new_folder
        assert old_folder in caplog.text and new_folder in caplog.text


def test_run_demo_resolves_and_rejects():
    import subprocess
    import sys

    run = [sys.executable, str(_REPO_ROOT / "run_demo.py")]
    ok = subprocess.run([*run, "object_detection:infer_video", "--help"], capture_output=True, text=True)
    assert ok.returncode == 0 and "usage: infer_video.py" in ok.stdout
    bad = subprocess.run([*run, "no_such_demo"], capture_output=True, text=True)
    assert bad.returncode != 0 and "Unknown demo" in bad.stderr
    escape = subprocess.run([*run, "object_detection:../../run_demo"], capture_output=True, text=True)
    assert escape.returncode != 0 and "Invalid script name" in escape.stderr


def test_old_embeddinggemma_folder_name_is_the_info_name(caplog):
    # The old folder name is also the demo's name: it resolves without a warning.
    with caplog.at_level(logging.WARNING):
        assert setup_demos.resolve_demo("EmbeddingGemma-2")[0] == "embeddinggemma2"
    assert not caplog.text
