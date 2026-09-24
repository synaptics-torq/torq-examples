# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""One-time setup for torq-examples demos.

Usage:
    python setup_demos.py gemma3 moonshine   # set up specific demos
    python setup_demos.py --all                # set up all demos
"""

import argparse
import logging
import os
import site
import sys

from utils.deps import MissingRequirementsError, check_requirements
from utils.download import DownloadError
from utils.log import add_logging_args, configure_logging

PTH_NAME = "torq-examples.pth"
logger = logging.getLogger("setup")

DEMOS = [
    "gemma3",
    "LiquidAI-LFM2.5-230M",
    "moonshine",
    "moonshine_streaming",
    "LiquidAI-LFM2-VL-450M",
    "object_detection",
    "pose_estimation",
    "face_id",
]


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


def setup_demo(name: str):
    try:
        if name == "gemma3":
            from gemma3.setup_demo import setup_gemma3
            setup_gemma3(["instruct"])
        elif name == "LiquidAI-LFM2.5-230M":
            # 230M is the LFM2.5 size with a published runtime HF repo
            # (Synaptics/LiquidAI-LFM2.5-230M); the 350M runtime repo is
            # not published yet.
            _mod = _load_demo_module("LiquidAI", "LiquidAI-LFM2.5-230M", "setup_demo.py")
            _mod.setup_liquid(["230m"])
        elif name == "moonshine":
            from moonshine.setup_demo import setup_moonshine
            setup_moonshine(["tiny-en"])
        elif name == "moonshine_streaming":
            from moonshine_streaming.setup_demo import setup_moonshine_streaming
            setup_moonshine_streaming(["streaming-tiny-en"])
        elif name == "LiquidAI-LFM2-VL-450M":
            _mod = _load_demo_module("LiquidAI", "LiquidAI-LFM2-VL-450M", "setup_demo.py")
            _mod.setup_liquidvl(["default"])
        elif name == "object_detection":
            from object_detection.setup_demo import setup_object_detection
            setup_object_detection()
        elif name == "pose_estimation":
            from pose_estimation.setup_demo import setup_pose_estimation
            setup_pose_estimation()
        elif name == "face_id":
            from Face_ID.setup_demo import setup_face_id
            setup_face_id()
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
        help=f"Demo(s) to set up. Valid names: {', '.join(DEMOS)}",
    )
    group.add_argument(
        "--all", action="store_true", dest="all_demos",
        help="Set up all demos",
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
        if name not in DEMOS:
            logger.error("Unknown demo: '%s'. Valid demos: %s", name, ", ".join(DEMOS))
            sys.exit(1)
        setup_demo(name)
