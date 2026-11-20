# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("tokenizers")

_SRC = Path(__file__).resolve().parents[1] / "moonshine" / "src"


@pytest.fixture
def infer(monkeypatch):
    monkeypatch.syspath_prepend(str(_SRC))
    monkeypatch.delitem(sys.modules, "runner", raising=False)
    spec = importlib.util.spec_from_file_location("moonshine_infer", _SRC / "infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("runner", None)


def test_transcribe_downmixes_and_resamples_to_the_model_rate(infer, tmp_path):
    # A 1 s, 48 kHz stereo file: the model must still get 16 kHz mono audio.
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.zeros((48000, 2), dtype=np.float32), 48000)
    runner = mock.Mock(input_freq=16000)
    runner.run.return_value = np.array([[1]])
    tokenizer = mock.Mock()
    tokenizer.decode_batch.return_value = ["hello"]

    result = infer._transcribe(wav, runner, tokenizer)

    assert runner.run.call_args.args[0].shape == (1, 16000)
    assert runner.input_freq == 16000
    assert result.text == "hello"
    assert result.audio_duration_ms == pytest.approx(1000)
