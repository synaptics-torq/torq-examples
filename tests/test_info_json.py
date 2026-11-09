# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Every demo folder that has a setup_demo.py must carry a valid info.json,
and ``setup_demos.DEMOS`` must resolve each demo to a module with a callable
uniform ``setup`` entry point."""

import json
from pathlib import Path

import setup_demos

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_FIELDS = ("name", "description", "task_category")


def _setup_demo_folders() -> list[str]:
    """Folders (relative to the repo root) that contain a setup_demo.py."""
    return sorted(
        path.parent.relative_to(_REPO_ROOT).as_posix()
        for pattern in ("*/setup_demo.py", "*/*/setup_demo.py")
        for path in _REPO_ROOT.glob(pattern)
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
