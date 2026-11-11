# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Default ``--model-dir``: the files ``setup_demos.py EmbeddingGemma-2`` downloaded."""

import argparse

from embeddinggemma2.setup_demo import local_embeddinggemma2_model_dir


def resolve_model_dir(parser: argparse.ArgumentParser, model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    local = local_embeddinggemma2_model_dir()
    if local is None:
        parser.error("no local EmbeddingGemma-2 model files found; pass -m/--model-dir or run "
                     "`python setup_demos.py EmbeddingGemma-2` from the torq-examples root")
    return str(local)
