import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_runner_import_stubs():
    if "numpy" not in sys.modules:
        numpy = types.ModuleType("numpy")

        class ndarray:
            pass

        numpy.ndarray = ndarray
        numpy.typing = types.ModuleType("numpy.typing")
        sys.modules["numpy"] = numpy
        sys.modules["numpy.typing"] = numpy.typing

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

    if "torq.runtime" not in sys.modules:
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

    if "iree.runtime" not in sys.modules:
        iree = types.ModuleType("iree")
        runtime = types.ModuleType("iree.runtime")

        class DeviceArray:
            pass

        runtime.DeviceArray = DeviceArray
        iree.runtime = runtime
        sys.modules["iree"] = iree
        sys.modules["iree.runtime"] = runtime

    if "tokenizers" not in sys.modules:
        tokenizers = types.ModuleType("tokenizers")

        class Tokenizer:
            pass

        tokenizers.Tokenizer = Tokenizer
        sys.modules["tokenizers"] = tokenizers

    if "ml_dtypes" not in sys.modules:
        ml_dtypes = types.ModuleType("ml_dtypes")
        ml_dtypes.bfloat16 = object()
        sys.modules["ml_dtypes"] = ml_dtypes


_install_runner_import_stubs()
from gemma3.src.runner import (
    resolve_token_id_lut,
)
from utils.llm import (
    discover_lm_head_path,
    resolve_lm_head_path,
    resolve_token_id_lut as utils_resolve_token_id_lut,
)


class GemmaLutValidationTest(unittest.TestCase):
    def test_gemma_reexports_utils_lut_resolver(self):
        self.assertIs(resolve_token_id_lut, utils_resolve_token_id_lut)

    def test_full_vocab_logits_do_not_need_lut(self):
        self.assertIsNone(resolve_token_id_lut(262144, 262144, None))

    def test_full_vocab_logits_ignore_unneeded_lut(self):
        logger = logging.getLogger("test_gemma_lut_validation")
        with self.assertLogs(logger, level="WARNING") as logs:
            selected = resolve_token_id_lut(262144, 262144, [1, 2, 3], logger)

        self.assertIsNone(selected)
        self.assertIn("ignoring the LUT", "\n".join(logs.output))

    def test_compact_logits_require_valid_lut(self):
        lut = [10, 20, 30]
        self.assertIs(resolve_token_id_lut(3, 262144, lut), lut)

    def test_compact_logits_reject_missing_lut(self):
        with self.assertRaisesRegex(ValueError, "required"):
            resolve_token_id_lut(3, 262144, None)

    def test_compact_logits_reject_invalid_lut_length(self):
        with self.assertRaisesRegex(ValueError, "does not match logits size"):
            resolve_token_id_lut(3, 262144, [10, 20])

    def test_unknown_vocab_still_checks_lut_length_when_possible(self):
        with self.assertRaisesRegex(ValueError, "does not match logits size"):
            resolve_token_id_lut(3, None, [10, 20])


class GemmaLMHeadDiscoveryTest(unittest.TestCase):
    def test_discovers_single_sibling_lm_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.vmfb"
            lm_head = Path(tmp) / "model-lm-head.vmfb"
            model.touch()
            lm_head.touch()

            self.assertEqual(discover_lm_head_path(model), lm_head)

    def test_discovers_sibling_lm_head_with_vmfb_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.vmfb.trim"
            lm_head = Path(tmp) / "lm_head.vmfb.w4a16"
            model.touch()
            lm_head.touch()

            self.assertEqual(discover_lm_head_path(model), lm_head)

    def test_discovery_ignores_model_path_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "lm_head_model.vmfb"
            model.touch()

            self.assertIsNone(discover_lm_head_path(model))

    def test_discovery_rejects_multiple_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.vmfb"
            model.touch()
            (Path(tmp) / "lm_head.vmfb").touch()
            (Path(tmp) / "other-lm-head.vmfb").touch()

            with self.assertRaisesRegex(ValueError, "multiple LM head candidates"):
                discover_lm_head_path(model)

    def test_explicit_lm_head_overrides_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.vmfb"
            explicit = Path(tmp) / "elsewhere.vmfb"
            model.touch()
            explicit.touch()
            (Path(tmp) / "lm_head.vmfb").touch()

            self.assertEqual(resolve_lm_head_path(model, explicit), explicit)

    def test_disable_lm_head_skips_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.vmfb"
            model.touch()
            (Path(tmp) / "lm_head.vmfb").touch()

            self.assertIsNone(resolve_lm_head_path(model, disable_lm_head=True))

    def test_rejects_explicit_and_disabled_lm_head(self):
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            resolve_lm_head_path("model.vmfb", "lm_head.vmfb", disable_lm_head=True)


if __name__ == "__main__":
    unittest.main()
