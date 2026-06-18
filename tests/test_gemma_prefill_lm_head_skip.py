import logging
import unittest
from unittest.mock import patch

from tests.test_gemma_lut_validation import _install_runner_import_stubs


_install_runner_import_stubs()

from gemma3.src.runner import Gemma3Static
from utils.inference import SplitLMHeadRunner


class RecordingGemma(Gemma3Static):
    __slots__ = ("calls",)

    def __init__(self):
        self.calls = []

    def llm_step(
        self,
        token: int,
        seq_pos: int,
        *,
        compute_logits: bool = True,
        sample_next: bool = True,
    ) -> int:
        self.calls.append((token, seq_pos, compute_logits, sample_next))
        return token + 100 if sample_next else 0


class FakeBody:
    def __init__(self):
        self.calls = []

    def infer(self, inputs):
        self.calls.append(inputs)
        return ["hidden"]


class TensorInfo:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class MetadataBody:
    model_path = "body.vmfb"

    def __init__(self, outputs_info):
        self.outputs_info = outputs_info


class ExplodingLMHead:
    def infer(self, inputs):
        raise AssertionError("LM head should not run")


def _fake_vmfb_runner(inputs_info):
    class FakeVMFBRunner:
        def __init__(self, *args, **kwargs):
            self.inputs_info = inputs_info
            self.outputs_info = [TensorInfo((1, 262144), "float32")]

    return FakeVMFBRunner


class PosBuffer:
    def __init__(self):
        self.values = []

    def __setitem__(self, key, value):
        self.values.append((key, value))

    def copy(self):
        return list(self.values)


class GemmaPrefillLMHeadSkipTest(unittest.TestCase):
    def test_prefill_samples_only_last_token_by_default(self):
        runner = RecordingGemma()

        next_token, pos = runner._prefill([10, 11, 12], start=7)

        self.assertEqual(next_token, 112)
        self.assertEqual(pos, 10)
        self.assertEqual(
            runner.calls,
            [(10, 7, False, False), (11, 8, False, False), (12, 9, True, True)],
        )

    def test_prefill_can_skip_producing_next_token(self):
        runner = RecordingGemma()

        next_token, pos = runner._prefill(
            [10, 11, 12],
            start=7,
            produce_next_token=False,
        )

        self.assertEqual(next_token, 0)
        self.assertEqual(pos, 10)
        self.assertEqual(
            runner.calls,
            [(10, 7, False, False), (11, 8, False, False), (12, 9, False, False)],
        )

    def test_llm_step_uses_skip_lm_head_when_logits_are_disabled(self):
        runner = Gemma3Static.__new__(Gemma3Static)
        model = SplitLMHeadRunner.__new__(SplitLMHeadRunner)
        body = FakeBody()
        model._body = body
        model._lm_head = ExplodingLMHead()
        model._infer_time_ms = 0.0
        runner._model = model
        runner._emb_buf = None
        runner._token_embeddings = None
        runner._pos_buf = PosBuffer()
        runner._logger = logging.getLogger("test_llm_step_body_only")

        token = runner.llm_step(123, 5, compute_logits=False, sample_next=False)

        self.assertEqual(token, 0)
        self.assertEqual(len(body.calls), 1)

    def test_llm_step_can_compute_logits_without_sampling(self):
        runner = Gemma3Static.__new__(Gemma3Static)
        model = SplitLMHeadRunner.__new__(SplitLMHeadRunner)
        body = FakeBody()
        model._body = body
        model._lm_head = FakeBody()
        model._infer_time_ms = 0.0
        runner._model = model
        runner._emb_buf = None
        runner._token_embeddings = None
        runner._pos_buf = PosBuffer()
        runner._logger = logging.getLogger("test_llm_step_logits_no_sample")

        token = runner.llm_step(123, 5, compute_logits=True, sample_next=False)

        self.assertEqual(token, 0)
        self.assertEqual(len(body.calls), 1)
        self.assertEqual(len(model._lm_head.calls), 1)

    def test_llm_step_rejects_sampling_without_logits(self):
        runner = Gemma3Static.__new__(Gemma3Static)

        with self.assertRaisesRegex(ValueError, "requires compute_logits"):
            runner.llm_step(123, 5, compute_logits=False, sample_next=True)

    def test_split_runner_skip_lm_head_infer_does_not_call_lm_head(self):
        runner = SplitLMHeadRunner.__new__(SplitLMHeadRunner)
        body = FakeBody()
        runner._body = body
        runner._lm_head = ExplodingLMHead()
        runner._infer_time_ms = 0.0

        self.assertEqual(runner.infer(["input"], skip_lm_head=True), ["hidden"])
        self.assertEqual(body.calls, [["input"]])

    def test_split_runner_accepts_matching_body_and_lm_head_metadata(self):
        body = MetadataBody([TensorInfo((1, 1, 256), "float32")])

        with patch(
            "utils.inference.VMFBInferenceRunner",
            _fake_vmfb_runner([TensorInfo((1, 1, 256), "float32")]),
        ):
            runner = SplitLMHeadRunner(body, "lm_head.vmfb")

        self.assertIs(runner._body, body)

    def test_split_runner_rejects_lm_head_shape_mismatch(self):
        body = MetadataBody([TensorInfo((1, 1, 256), "float32")])

        with patch(
            "utils.inference.VMFBInferenceRunner",
            _fake_vmfb_runner([TensorInfo((1, 1, 128), "float32")]),
        ):
            with self.assertRaisesRegex(ValueError, "input shape"):
                SplitLMHeadRunner(body, "lm_head.vmfb")

    def test_split_runner_rejects_lm_head_dtype_mismatch(self):
        body = MetadataBody([TensorInfo((1, 1, 256), "float32")])

        with patch(
            "utils.inference.VMFBInferenceRunner",
            _fake_vmfb_runner([TensorInfo((1, 1, 256), "float16")]),
        ):
            with self.assertRaisesRegex(ValueError, "input dtype"):
                SplitLMHeadRunner(body, "lm_head.vmfb")


if __name__ == "__main__":
    unittest.main()
