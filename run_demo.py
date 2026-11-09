# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Run a demo by name.

Usage:
    python run_demo.py gemma3 [ARGS...]                       # runs gemma3/src/infer.py
    python run_demo.py object_detection:infer_video [ARGS...]  # runs object_detection/src/infer_video.py

DEMO is the demo's info.json name or its folder; ARGS are passed through unchanged.
The demo replaces this process, so signals, stdin and the exit code pass straight through.
"""

import os
import sys

from setup_demos import DEMOS, REPO_ROOT, resolve_demo

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(f"{__doc__}\nDemos: {', '.join(DEMOS)}")
    sys.exit(len(sys.argv) < 2)

name, _, script = sys.argv[1].partition(":")
resolved = resolve_demo(name)
if resolved is None:
    sys.exit(f"Unknown demo: '{name}'. Valid demos: {', '.join(DEMOS)}")
if os.path.basename(script) != script:
    sys.exit(f"Invalid script name: '{script}'")
path = os.path.join(REPO_ROOT, resolved[0], "src", f"{script or 'infer'}.py")
if not os.path.isfile(path):
    sys.exit(f"No such script: {path}")
os.execv(sys.executable, [sys.executable, path, *sys.argv[2:]])
