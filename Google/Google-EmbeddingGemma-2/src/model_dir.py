# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Default ``--model-dir``: the files ``setup_demos.py Google-EmbeddingGemma-2`` downloaded."""

import argparse
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent.parent


def resolve_model_dir(parser: argparse.ArgumentParser, model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    # setup_demo.py sits in the demo directory (not an importable package) and imports the
    # repo's `utils` package, so both directories go on the path.
    for p in (_DEMO_DIR, _DEMO_DIR.parent.parent):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from setup_demo import local_embeddinggemma2_model_dir
        local = local_embeddinggemma2_model_dir()
    except Exception:
        local = None
    if local is None:
        parser.error("no local EmbeddingGemma-2 model files found; pass -m/--model-dir or run "
                     "`python setup_demos.py Google-EmbeddingGemma-2` from the torq-examples root")
    return str(local)
