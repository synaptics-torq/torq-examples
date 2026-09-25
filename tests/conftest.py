# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared pytest configuration for torq-examples tests.

Two things are set up here, both at conftest import time so they take effect
before pytest collects and imports the test modules:

1. The repository root is added to ``sys.path`` so ``utils.*``, ``gemma3.*``
   and ``moonshine.*`` resolve regardless of the directory pytest runs from.
2. Lightweight stand-ins are installed for the heavy optional dependencies
   (numpy, torq, iree, tokenizers, ml_dtypes) that ``utils.llm``,
   ``utils.inference`` and ``gemma3.src.runner`` import at module load. The
   board-specific runtime (``torq``/``iree``) is unavailable on the host CI
   runner and the pure-Python logic under test does not exercise it. Each
   stub is installed only when the real package cannot be imported, so it
   never shadows a genuine install (numpy/tokenizers/ml_dtypes run for real
   where present).
"""

import importlib
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        # Broken native extensions can raise anything at import time;
        # for stub purposes that is equivalent to "not installed".
        return False
    return True


def _install_runner_import_stubs() -> None:
    if not _importable("numpy"):
        numpy = types.ModuleType("numpy")

        class ndarray:
            pass

        numpy.ndarray = ndarray
        numpy.typing = types.ModuleType("numpy.typing")
        sys.modules["numpy"] = numpy
        sys.modules["numpy.typing"] = numpy.typing
    elif "numpy.typing" not in sys.modules:
        import numpy.typing  # noqa: F401

    if "numpy.typing" not in sys.modules:
        numpy_typing = types.ModuleType("numpy.typing")

        class NDArray:
            pass

        numpy_typing.NDArray = NDArray
        sys.modules["numpy.typing"] = numpy_typing
    elif not hasattr(sys.modules["numpy.typing"], "NDArray"):
        class NDArray:
            pass

        sys.modules["numpy.typing"].NDArray = NDArray

    if not _importable("torq.runtime"):
        torq = types.ModuleType("torq")
        runtime = types.ModuleType("torq.runtime")
        runtime_utils = types.ModuleType("torq.runtime.utils")

        class InferenceRunner:
            pass

        class VMFBInferenceRunner:
            pass

        class TensorInfo:
            pass

        runtime.InferenceRunner = InferenceRunner
        runtime.VMFBInferenceRunner = VMFBInferenceRunner
        runtime_utils.TensorInfo = TensorInfo
        torq.runtime = runtime
        sys.modules["torq"] = torq
        sys.modules["torq.runtime"] = runtime
        sys.modules["torq.runtime.utils"] = runtime_utils

    if not _importable("iree.runtime"):
        iree = types.ModuleType("iree")
        runtime = types.ModuleType("iree.runtime")

        class DeviceArray:
            pass

        runtime.DeviceArray = DeviceArray
        iree.runtime = runtime
        sys.modules["iree"] = iree
        sys.modules["iree.runtime"] = runtime

    if not _importable("tokenizers"):
        tokenizers = types.ModuleType("tokenizers")

        class Tokenizer:
            pass

        tokenizers.Tokenizer = Tokenizer
        sys.modules["tokenizers"] = tokenizers

    if not _importable("ml_dtypes"):
        ml_dtypes = types.ModuleType("ml_dtypes")
        ml_dtypes.bfloat16 = object()
        sys.modules["ml_dtypes"] = ml_dtypes


_install_runner_import_stubs()
