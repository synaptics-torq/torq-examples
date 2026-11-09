# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""One-time setup for torq-examples demos.

Usage:
    python setup_demos.py gemma3 moonshine   # set up specific demos
    python setup_demos.py --all                # set up all demos
"""

import argparse
import glob
import json
import logging
import os
import site
import sys
from importlib import import_module
from typing import Final

from utils.deps import MissingRequirementsError, check_requirements
from utils.download import DownloadError
from utils.log import add_logging_args, configure_logging

PTH_NAME = "torq-examples.pth"
logger = logging.getLogger("setup")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _demo_folders() -> dict[str, dict]:
    """Map demo folder (relative to the repo root) to its ``info.json``."""
    folders: dict[str, dict] = {}
    for info_path in sorted(
        path
        for pattern in ("*/info.json", "*/*/info.json")
        for path in glob.glob(pattern, root_dir=REPO_ROOT)
    ):
        with open(os.path.join(REPO_ROOT, info_path), encoding="utf-8") as f:
            folders[os.path.dirname(info_path)] = json.load(f)
    return folders


_DEMO_FOLDERS: Final[dict[str, dict]] = _demo_folders()

#: Demo names (the ``name`` field of each info.json), sorted by folder.
DEMOS: Final[list[str]] = [
    _DEMO_FOLDERS[folder]["name"] for folder in sorted(_DEMO_FOLDERS)
]

#: Names of the demos whose info.json marks them as batched-prefill capable
#: (shown in the --with-batch-prefill help text).
_PREFILL_DEMOS: Final[list[str]] = sorted(
    info["name"] for info in _DEMO_FOLDERS.values() if info.get("batch_prefill")
)


def _site_packages_dir() -> str:
    """Return the appropriate site-packages directory."""
    if sys.prefix != sys.base_prefix:
        # Inside a virtual environment
        return site.getsitepackages()[0]
    return site.getusersitepackages()


def install():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    site_dir = _site_packages_dir()
    os.makedirs(site_dir, exist_ok=True)
    pth_file = os.path.join(site_dir, PTH_NAME)
    with open(pth_file, "w") as f:
        f.write(repo_root + "\n")
    logger.debug("Created %s", pth_file)
    logger.info("Added '%s' to Python's import path. To undo, delete '%s'", repo_root, pth_file)


def _load_demo_module(*path_parts):
    """Load a demo's setup_demo.py by file path.

    The Liquid demos live under LiquidAI/ in directories whose names contain
    dots and hyphens, so they are not importable as packages; load them by file
    path instead.
    """
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *path_parts)
    modname = "_demo_" + "_".join(path_parts).replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_demo_setup(folder: str):
    """Load the setup_demo module of a demo folder (relative to the repo root)."""
    if "/" not in folder and folder.isidentifier():
        return import_module(f"{folder}.setup_demo")
    return _load_demo_module(*folder.split("/"), "setup_demo.py")


def resolve_demo(name: str) -> tuple[str, dict] | None:
    """Resolve a demo by info name or folder path; return ``(folder, info)``."""
    for folder, info in _DEMO_FOLDERS.items():
        if name in (folder, info["name"]):
            return folder, info
    return None


def setup_demo(
    name: str,
    *,
    model_version: str | None = None,
    no_update: bool = False,
    enable_prefill: bool = False,
):
    resolved = resolve_demo(name)
    if resolved is None:
        logger.error("Unknown demo: '%s'. Valid demos: %s", name, ", ".join(DEMOS))
        sys.exit(1)
    folder, info = resolved
    if enable_prefill and not info.get("batch_prefill"):
        logger.warning(
            "--with-batch-prefill has no effect for the '%s' demo (no batched prefill "
            "model); ignoring.", name,
        )
    module = _load_demo_setup(folder)
    try:
        module.setup(
            model_version=model_version,
            no_update=no_update,
            enable_prefill=enable_prefill,
        )
    except (DownloadError, MissingRequirementsError) as e:
        logger.error("Setup failed for '%s': %s", name, e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-time setup for torq-examples demos.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "demos", nargs="*", default=[], metavar="DEMO",
        help=(
            f"Demo(s) to set up. Valid names: {', '.join(DEMOS)}, or a demo folder "
            "path relative to the repo root (e.g. LiquidAI/LiquidAI-LFM2.5)"
        ),
    )
    group.add_argument(
        "--all", action="store_true", dest="all_demos",
        help="Set up all demos",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help=(
            "Model version tag to download for every demo (default: the "
            "torq-examples version for built-in repos, the repo's latest "
            "revision for custom repos). A pinned version is kept in sync with "
            "its own tag but never upgraded."
        ),
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help=(
            "Download without tracking: no .manifest.json is written, so the "
            "models are never checked for updates or refreshed (at your own "
            "risk). Unless a version is pinned explicitly, the repo's latest "
            "(HEAD) revision is downloaded."
        ),
    )
    parser.add_argument(
        "--with-batch-prefill",
        action="store_true",
        help=(
            "Also download and track the optional batched prefill model "
            "(transformer_prefill.vmfb) for the demos that have one "
            f"({', '.join(_PREFILL_DEMOS)}). It is an extra model, so "
            "it uses more memory, and its prompt-chunk size is fixed. "
            "Warns and is ignored for the other demos."
        ),
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging)

    if sys.prefix == sys.base_prefix:
        logger.warning(
            "Running outside a virtual environment. "
            "A venv is highly recommended: python3 -m venv .venv && source .venv/bin/activate"
        )

    try:
        check_requirements("requirements.txt")
    except MissingRequirementsError as e:
        logger.error("%s", e)
        sys.exit(1)

    # Always ensure .pth is installed first
    install()

    demos_to_run = list(DEMOS) if args.all_demos else args.demos
    if not demos_to_run:
        parser.print_help()
        sys.exit(0)

    for name in demos_to_run:
        setup_demo(
            name,
            model_version=args.model_version,
            no_update=args.no_update,
            enable_prefill=args.with_batch_prefill,
        )
