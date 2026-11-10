# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for utils.metrics: SystemSampler math and the MetricsRun JSON output."""

import argparse
import json
from pathlib import Path

import pytest

from utils import metrics as metrics_module
from utils.metrics import MetricsRun, SystemSampler
from utils.runtime import add_common_inference_args


def _write_proc_stat(path: Path, user: int, nice: int, system: int, idle: int, iowait: int) -> None:
    # A per-core line must not be picked up: only the aggregate "cpu " line counts.
    path.write_text(
        f"cpu  {user} {nice} {system} {idle} {iowait} 0 0 0 0 0\n"
        f"cpu0 {user} {nice} {system} {idle} {iowait} 0 0 0 0 0\n"
    )


def _patch_paths(tmp_path, monkeypatch, *, npu: str | None = None) -> tuple[Path, Path]:
    proc_stat = tmp_path / "stat"
    _write_proc_stat(proc_stat, 0, 0, 0, 100, 0)
    npu_file = tmp_path / "npu"
    if npu is not None:
        npu_file.write_text(f"{npu}\n")
    monkeypatch.setattr(metrics_module, "PROC_STAT_PATH", str(proc_stat))
    monkeypatch.setattr(metrics_module, "NPU_INFERENCE_TIME_PATH", str(npu_file))
    return proc_stat, npu_file


def _patch_clock(monkeypatch, wall_ns: int) -> None:
    clock = iter([0, wall_ns])
    monkeypatch.setattr(metrics_module, "perf_counter_ns", lambda: next(clock))


# ── SystemSampler ────────────────────────────────────────────────────────────


def test_sampler_cpu_load_math(tmp_path, monkeypatch):
    proc_stat, _ = _patch_paths(tmp_path, monkeypatch, npu=None)

    sampler = SystemSampler()
    _write_proc_stat(proc_stat, 100, 0, 100, 500, 50)
    sampler.start()
    _write_proc_stat(proc_stat, 200, 0, 100, 550, 50)
    values = sampler.stop()

    # d_idle = 50 (idle only; iowait unchanged), d_total = 150.
    assert values["cpu_load"] == pytest.approx(100.0 * (1.0 - 50.0 / 150.0))
    assert "npu_load" not in values
    assert values["peak_memory"] > 0


def test_sampler_npu_load_math(tmp_path, monkeypatch):
    _, npu_file = _patch_paths(tmp_path, monkeypatch, npu=1000)
    _patch_clock(monkeypatch, wall_ns=6_000_000)

    sampler = SystemSampler()
    sampler.start()
    npu_file.write_text("4000\n")
    values = sampler.stop()

    # 3000 us = 3 ms of active NPU in a 6 ms wall run -> 50%.
    assert values["npu_load"] == pytest.approx(50.0)


def test_sampler_npu_load_clamped_to_100(tmp_path, monkeypatch):
    _, npu_file = _patch_paths(tmp_path, monkeypatch, npu=0)
    _patch_clock(monkeypatch, wall_ns=6_000_000)

    sampler = SystemSampler()
    sampler.start()
    npu_file.write_text("20000\n")
    values = sampler.stop()

    # 20 ms of "active" time in a 6 ms wall run is impossible -> clamped.
    assert values["npu_load"] == 100.0


def test_sampler_omits_npu_load_when_file_missing(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch, npu=None)

    sampler = SystemSampler()
    sampler.start()
    values = sampler.stop()

    # Never reported as 0: the key is absent instead.
    assert "npu_load" not in values
    assert values["cpu_load"] == 0.0  # the fake /proc/stat did not advance


# ── MetricsRun ───────────────────────────────────────────────────────────────


def _metrics_run_env(tmp_path, monkeypatch) -> Path:
    _patch_paths(tmp_path, monkeypatch, npu=0)
    return tmp_path / "metrics.json"


def test_metrics_run_json_shape_and_means(tmp_path, monkeypatch):
    out = _metrics_run_env(tmp_path, monkeypatch)

    with MetricsRun(str(out)) as run:
        run.add("time_to_first_token", 100)
        run.add("latency_text", 5)
        run.add("time_to_first_token", 300)
        run.add("latency_audio", 10)

    doc = json.loads(out.read_text())
    assert set(doc) == {"metrics", "measurements"}
    assert [m["name"] for m in doc["metrics"]] == [
        "time_to_first_token", "latency_text", "latency_audio",
        "cpu_load", "npu_load", "peak_memory",
    ]
    for m in doc["metrics"]:
        assert set(m) == {"name", "description", "unit"}
    by_name = {m["name"]: m for m in doc["metrics"]}
    assert by_name["time_to_first_token"]["unit"] == "ns"
    assert by_name["latency_text"] == {
        "name": "latency_text", "description": "Latency (text)", "unit": "ns",
    }
    assert by_name["latency_audio"]["description"] == "Latency (audio)"
    assert by_name["cpu_load"]["unit"] == "percent"
    assert by_name["npu_load"]["unit"] == "percent"
    assert by_name["peak_memory"]["unit"] == "bytes"
    # The metrics list holds exactly the names present in measurements.
    assert [m["name"] for m in doc["metrics"]] == list(doc["measurements"])

    assert doc["measurements"]["time_to_first_token"] == 200.0  # mean of 100, 300
    assert doc["measurements"]["latency_text"] == 5.0
    assert doc["measurements"]["cpu_load"] == 0.0
    assert doc["measurements"]["npu_load"] == 0.0
    assert doc["measurements"]["peak_memory"] > 0
    # Atomic write leaves no temp files behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.json", "npu", "stat"]


def test_metrics_run_rejects_unknown_name(tmp_path, monkeypatch):
    _metrics_run_env(tmp_path, monkeypatch)

    with MetricsRun(None) as run:
        with pytest.raises(ValueError, match="Unknown metric name"):
            run.add("bogus", 1)


def test_metrics_run_writes_on_keyboard_interrupt(tmp_path, monkeypatch):
    out = _metrics_run_env(tmp_path, monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        with MetricsRun(str(out)) as run:
            run.add("latency", 42)
            raise KeyboardInterrupt

    doc = json.loads(out.read_text())
    assert doc["measurements"]["latency"] == 42.0


def test_metrics_run_no_file_when_path_none(tmp_path, monkeypatch):
    _metrics_run_env(tmp_path, monkeypatch)

    with MetricsRun(None) as run:
        run.add("latency", 1)

    assert not any(p.suffix == ".json" for p in tmp_path.iterdir())


# ── CLI flag ─────────────────────────────────────────────────────────────────


def test_add_common_inference_args_includes_metrics_out():
    parser = argparse.ArgumentParser()
    add_common_inference_args(parser)
    assert parser.parse_args([]).metrics_out is None
    assert parser.parse_args(["--metrics-out", "m.json"]).metrics_out == "m.json"
