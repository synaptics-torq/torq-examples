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

from utils.deps import check_requirements
from utils.errors import SetupError
from utils.log import add_logging_args, configure_logging

PTH_NAME = "torq-examples.pth"
logger = logging.getLogger("setup")

DEMOS = ["gemma3"]


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


def setup_demo(name: str):
    try:
        if name == "gemma3":
            from gemma3.setup import setup_gemma3
            setup_gemma3(["instruct"])
    except SetupError as e:
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
    except SetupError as e:
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
