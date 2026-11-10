# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for utils.runtime: RuntimeConfig, Model, to_host, and npu_clock."""

import argparse
import logging

import numpy as np
import pytest

import utils.runtime as runtime
from utils.runtime import Model, RuntimeConfig, npu_clock, to_host


class _DeviceArray:
    """Stand-in for an iree DeviceArray: a value plus a to_host()."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def to_host(self):
        return self.value


class _Info:
    shape = (1, 4)
    dtype = "float32"


class _FakeRunner:
    """Records its constructor kwargs; infer() returns a device array per input."""

    instances = []

    def __init__(self, model_path, *, function="main", device_uri="torq",
                 n_threads=None, runtime_flags=None, device_outputs=False):
        self.model_path = model_path
        self.kwargs = dict(function=function, device_uri=device_uri,
                           n_threads=n_threads, runtime_flags=runtime_flags,
                           device_outputs=device_outputs)
        self.inputs_info = [_Info()]
        self.outputs_info = [_Info()]
        self.device = "fake-device"
        self.infer_time_ms = 1.23
        self.last_inputs = None
        _FakeRunner.instances.append(self)

    def infer(self, inputs):
        self.last_inputs = list(inputs)
        return [_DeviceArray(np.zeros((1, 4))) for _ in inputs]

    def allocate_device_array(self, array):
        return _DeviceArray(array)


@pytest.fixture
def fake_runner(monkeypatch):
    _FakeRunner.instances.clear()
    monkeypatch.setattr(runtime, "VMFBInferenceRunner", _FakeRunner)
    yield


# ── RuntimeConfig ─────────────────────────────────────────────────────────────


def test_dmabuf_forces_device_io_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="utils.runtime"):
        cfg = RuntimeConfig(tda="dmabuf", device_io=False)
    assert cfg.device_io is True
    assert "dmabuf" in caplog.text


def test_cpu_keeps_host_io():
    assert RuntimeConfig(tda="cpu", device_io=False).device_io is False


def test_from_args_defaults_on_bare_namespace():
    assert RuntimeConfig.from_args(argparse.Namespace()) == RuntimeConfig()


# ── Model ─────────────────────────────────────────────────────────────────────


def test_model_passes_options_through(fake_runner):
    cfg = RuntimeConfig(device="cpu", tda="cpu", device_io=False, threads=4,
                        extra_flags=("--a",))
    Model("m.vmfb", cfg)
    assert _FakeRunner.instances[-1].kwargs == dict(
        function="main", device_uri="cpu", n_threads=4,
        runtime_flags=["--torq_device_allocator=cpu", "--a"], device_outputs=False,
    )


def test_model_device_outputs_or_device_io(fake_runner):
    Model("m.vmfb", RuntimeConfig(tda="cpu", device_io=True))
    assert _FakeRunner.instances[-1].kwargs["device_outputs"] is True
    Model("m.vmfb", RuntimeConfig(tda="cpu", device_io=False), device_outputs=True)
    assert _FakeRunner.instances[-1].kwargs["device_outputs"] is True


def test_to_device_only_when_device_io(fake_runner):
    host = Model("m.vmfb", RuntimeConfig(tda="cpu", device_io=False))
    arr = np.zeros(4)
    assert host.to_device(arr) is arr

    device = Model("m.vmfb", RuntimeConfig(tda="cpu", device_io=True))
    staged = device.to_device(arr)
    assert isinstance(staged, _DeviceArray)
    # A pre-staged device array is never uploaded twice.
    assert device.to_device(staged) is staged


def test_infer_stages_and_returns_raw_outputs(fake_runner):
    m = Model("m.vmfb", RuntimeConfig(tda="cpu", device_io=True))
    outs = m.infer([np.zeros((1, 4))])
    assert all(isinstance(o, _DeviceArray) for o in outs)
    assert isinstance(_FakeRunner.instances[-1].last_inputs[0], _DeviceArray)


def test_run_returns_host_arrays(fake_runner):
    m = Model("m.vmfb", RuntimeConfig(tda="cpu", device_io=True))
    outs = m.run(np.zeros((1, 4)))
    assert all(isinstance(o, np.ndarray) for o in outs)


def test_to_host_passthrough():
    arr = np.zeros(3)
    assert to_host(arr) is arr
    assert to_host(_DeviceArray(arr)).shape == (3,)


# ── npu_clock ─────────────────────────────────────────────────────────────────


def _freq_calls(monkeypatch):
    calls = []

    def fake(level="max"):
        calls.append(level)
        return True, f"set to {level}"

    monkeypatch.setattr("utils.npu.configure_npu_userspace_frequency", fake)
    return calls


def test_npu_clock_sets_max_and_restores_min(monkeypatch, capsys):
    calls = _freq_calls(monkeypatch)
    with npu_clock():
        pass
    assert calls == ["max", "min"]
    out = capsys.readouterr().out
    assert "[NPU] set to max" in out
    assert "[NPU] set to min" in out


def test_npu_clock_restores_min_on_exception(monkeypatch):
    calls = _freq_calls(monkeypatch)
    with pytest.raises(RuntimeError):
        with npu_clock():
            raise RuntimeError("boom")
    assert calls == ["max", "min"]
