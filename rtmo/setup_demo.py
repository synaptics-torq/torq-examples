# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download the RTMO pose demo assets from Hugging Face (Synaptics/RTMO_pose):
the three NSS-only hybrid vmfbs, the matching TFLite parts (source of the int8
quant params — a compiled vmfb does not expose them), and two sample images.
The DCC pose-decode weights ship with the demo (rtmo_core/postprocess_weights.npz).
"""

import logging
from pathlib import Path
from typing import Final

from utils.deps import MissingRequirementsError, check_requirements
from utils.download import DownloadError, default_models_dir, download_from_hf

logger = logging.getLogger("rtmo.setup")

RTMO_REPO_ID: Final[str] = "Synaptics/RTMO_pose"

VMFB_FILES: Final[tuple[str, ...]] = (
    "vmfb/rtmo_hyb_backbone_int8.vmfb",
    "vmfb/rtmo_hyb_transformer_bf16.vmfb",
    "vmfb/rtmo_hyb_head_int8.vmfb",
)
TFLITE_FILES: Final[tuple[str, ...]] = (
    "tflite/rtmo_hybrid_backbone_int8.tflite",
    "tflite/rtmo_hybrid_transformer_bf16.tflite",
    "tflite/rtmo_hybrid_head_int8.tflite",
)
SAMPLE_FILES: Final[tuple[str, ...]] = ("calib/person.jpg", "calib/people.jpg")


def download_rtmo(base_dir=None) -> Path:
    """Download the RTMO assets (already-present files are skipped); return the model dir."""
    base_dir = Path(base_dir) if base_dir is not None else default_models_dir()
    for filename in (*VMFB_FILES, *TFLITE_FILES, *SAMPLE_FILES):
        download_from_hf(RTMO_REPO_ID, filename, base_dir=base_dir)
    return base_dir / RTMO_REPO_ID


def ensure_rtmo_models(model_dir=None, *, refresh=True) -> Path:
    """Ensure the assets are present. ``model_dir`` may be the .../Synaptics/RTMO_pose
    dir (as passed by infer.py) or None for the shared models/ dir."""
    if model_dir is None:
        base_dir = default_models_dir()
    else:
        base_dir = Path(model_dir)
        for _ in Path(RTMO_REPO_ID).parts:  # strip Synaptics/RTMO_pose -> base
            base_dir = base_dir.parent
    return download_rtmo(base_dir)


def setup_rtmo() -> None:
    """setup_demos.py entry point: verify deps + download assets."""
    check_requirements(Path(__file__).parent / "requirements.txt")
    logger.info("Setting up RTMO pose demo from %s", RTMO_REPO_ID)
    try:
        model_dir = download_rtmo()
    except Exception as e:
        raise DownloadError(f"Unable to download RTMO assets from {RTMO_REPO_ID}") from e
    logger.info("RTMO assets ready at %s", model_dir)


if __name__ == "__main__":
    import argparse
    import sys

    from utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(description="Download the RTMO pose demo assets.")
    add_logging_args(parser)
    configure_logging(parser.parse_args().logging)
    try:
        setup_rtmo()
    except (DownloadError, MissingRequirementsError, ValueError) as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
