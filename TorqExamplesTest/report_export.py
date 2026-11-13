"""Merge per-demo JSON artifacts into one machine-readable report.

The suite scatters structured results across the logs directory
(`<demo>_profile.json`, `setup_model_versions.json`, `prepare_metadata.json`,
`dut_pip_list.log`, `*_run.log`). Marketing / devzone ingestion wants a single
document. `generate_json_report` joins them into one JSON file:

    {
      "report_version": 1,
      "generated_at": "...",
      "torq_examples_commit": "...",
      "dut_sdk_version": "...",
      "torq_runtime_version": "...",
      "target_serial": "...",
      "summary": { per-demo + per-model-file inference/profile status counts },
      "models": [ per profile case: version anchors + inference/profile status
                  + parsed metrics ]
    }

Metric strings ("27.934 ms", "80.6%", "99.2 MB (includes ...)") are parsed
into {"value": float, "unit": str, "note": str} so downstream consumers do
not re-parse free text.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from TorqExamplesTest.generate_test_summary import (
    _build_infer_index,
    _case_id,
    _infer_status_for,
    _profile_status,
    load_infer_json,
    load_model_versions,
    load_prepare_metadata,
    load_profile_json,
    load_run_metrics,
    load_runtime_version,
    merge_profile_records,
)

# Worst-status ordering for aggregating per-model-file statuses up to a demo.
_STATUS_SEVERITY = {
    "fail": 3,
    "missing-metrics": 3,
    "not_run": 2,
    "skipped": 1,
    "pass": 0,
}

METRIC_KEYS = {
    "Avg CPU usage": "avg_cpu_usage",
    "Avg DRAM footprint": "avg_dram_footprint",
    "Avg NPU usage": "avg_npu_usage",
    "Avg inference time": "avg_inference_time",
    "Avg memory usage": "avg_memory_usage",
    "Peak DRAM footprint": "peak_dram_footprint",
    "Peak memory usage": "peak_memory_usage",
}

_VALUE_RE = re.compile(r"^\s*([\d.]+)\s*([A-Za-z%]+)?\s*(\(.*\))?\s*$")


def _parse_value(raw: str) -> dict[str, Any] | None:
    m = _VALUE_RE.match(raw or "")
    if not m:
        return None
    value: dict[str, Any] = {"value": float(m.group(1))}
    if m.group(2):
        value["unit"] = m.group(2)
    if m.group(3):
        value["note"] = m.group(3).strip("()")
    return value


def _version_key(case_id: str) -> str:
    """Normalize a profile case id ('demo:repo[:file]') to the suite's
    underscore case id ('demo_repo'), dropping the per-file segment so
    per-file profile cases join the repo-level version/inference records.
    (File names may contain underscores, so split on ':' not '_'.)"""
    return "_".join(case_id.split(":")[:2])


def _repo_id_from_path(model_path: str) -> str:
    if "/models/" not in model_path:
        return ""
    mp = model_path.split("/models/", 1)[1]
    if "/" in mp and mp.rsplit("/", 1)[1].endswith((".vmfb", ".trim")):
        mp = mp.rsplit("/", 1)[0]
    return mp


def _junit_inference_status(junit_path: Path, case_name: str) -> str:
    """pass/fail/skipped/not_run for the inference case matching case_name
    (new test_dut_infer[...] or legacy test_run_demo[...]).

    New-flow JUnit cases are demo-level (test_dut_infer[gemma3]) while
    profile case names are demo:repo[:file], so both the full normalized
    key and the bare demo name are accepted as matches."""
    key = _version_key(case_name)
    demo = case_name.split(":", 1)[0]
    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError):
        return "not_run"
    for tc in root.findall(".//testcase"):
        name = tc.get("name") or ""
        if not name.startswith(("test_run_demo[", "test_dut_infer[")):
            continue
        if _case_id(name) not in (key, demo):
            continue
        if tc.find("failure") is not None or tc.find("error") is not None:
            return "fail"
        if tc.find("skipped") is not None:
            return "skipped"
        return "pass"
    return "not_run"


def generate_json_report(
    junit_path: Path,
    profile_dir: Path,
    out_path: Path,
) -> None:
    profile_records = merge_profile_records(load_profile_json(profile_dir))
    model_versions = load_model_versions(profile_dir)
    metadata = load_prepare_metadata(profile_dir)
    run_metrics = load_run_metrics(profile_dir)
    runtime_version = load_runtime_version(profile_dir)

    # Per-model-file inference records are authoritative when present (new
    # flow). Without them (older runs) fall back to the JUnit demo-level case.
    infer_records = load_infer_json(profile_dir)
    by_drv, by_dr = _build_infer_index(infer_records)
    have_infer = bool(infer_records)

    models: list[dict[str, Any]] = []
    target_serial = ""
    for record in profile_records:
        if not isinstance(record, dict):
            continue
        case_name = str(record.get("case_name") or "")
        version = model_versions.get(_version_key(case_name), {})
        metrics: dict[str, Any] = {}
        for label, key in METRIC_KEYS.items():
            parsed = _parse_value(str(record.get("metrics", {}).get(label, "")))
            if parsed is not None:
                metrics[key] = parsed
        model_path = str(record.get("model_path", ""))
        if have_infer:
            inference_status = _infer_status_for(record, by_drv, by_dr)
        else:
            inference_status = _junit_inference_status(junit_path, case_name)
        parts = case_name.split(":")
        vmfb = record.get("vmfb") if "vmfb" in record else (parts[2] if len(parts) >= 3 else "")
        entry: dict[str, Any] = {
            "demo": record.get("demo_name", ""),
            "case": case_name,
            "repo_id": _repo_id_from_path(model_path),
            "vmfb": str(vmfb or ""),
            "model_version": version.get("version", ""),
            "revision": version.get("revision", ""),
            "model_path": model_path,
            "target_serial": record.get("target", ""),
            "inference_status": inference_status,
            "profile_status": _profile_status(record),
            "profile": metrics,
        }
        rm = run_metrics.get(_version_key(case_name)) or run_metrics.get(
            str(record.get("demo_name", ""))
        )
        if rm:
            run_block: dict[str, Any] = {}
            if rm.get("TTFT"):
                run_block["ttft"] = _parse_value(rm["TTFT"])
            if rm.get("tok/s"):
                run_block["tokens_per_s"] = _parse_value(rm["tok/s"])
            if rm.get("RTF"):
                run_block["rtf"] = _parse_value(rm["RTF"])
            if run_block:
                entry["run_metrics"] = run_block
        if entry["target_serial"]:
            target_serial = entry["target_serial"]
        models.append(entry)

    # Per-demo inference = the worst per-model-file status for that demo.
    demos: dict[str, str] = {}
    for entry in models:
        demo = str(entry["demo"])
        status = str(entry["inference_status"])
        if demo not in demos or (
            _STATUS_SEVERITY.get(status, 0) > _STATUS_SEVERITY.get(demos[demo], 0)
        ):
            demos[demo] = status
    demo_status_counts: dict[str, int] = {}
    for status in demos.values():
        demo_status_counts[status] = demo_status_counts.get(status, 0) + 1

    # Per-model-file counts (zero-omission: one row per profiled vmfb).
    infer_file_counts: dict[str, int] = {}
    profile_file_counts: dict[str, int] = {}
    for entry in models:
        infer_file_counts[entry["inference_status"]] = (
            infer_file_counts.get(entry["inference_status"], 0) + 1
        )
        profile_file_counts[entry["profile_status"]] = (
            profile_file_counts.get(entry["profile_status"], 0) + 1
        )

    report = {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torq_examples_commit": metadata.get("torq_examples_commit", ""),
        "dut_sdk_version": metadata.get("dut_sdk_version", ""),
        "torq_compiler_version": metadata.get("torq_compiler_version", ""),
        "torq_compiler_role": metadata.get("torq_compiler_role", ""),
        "torq_runtime_version": runtime_version,
        "target_serial": target_serial,
        "summary": {
            "demos": len(demos),
            "inference": demo_status_counts,
            "model_files": len(models),
            "inference_models": infer_file_counts,
            "profile_models": profile_file_counts,
            "profiled_models": len(models),
        },
        "models": models,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote JSON report: {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, default=Path("TEST-TorqExamplesTest.xml"))
    parser.add_argument("--profile-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, default=Path("torq_examples_report.json"))
    args = parser.parse_args()
    generate_json_report(args.junit, args.profile_dir, args.out)


if __name__ == "__main__":
    main()
