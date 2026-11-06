#!/usr/bin/env python3
"""Generate one TorqExamples-style HTML report from JUnit XML and profile data."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROFILE_METRICS = (
    "Avg CPU usage",
    "Avg DRAM footprint",
    "Avg NPU usage",
    "Avg inference time",
    "Avg memory usage",
    "Peak DRAM footprint",
    "Peak memory usage",
)


def _testcase_status(testcase: ET.Element) -> str:
    if testcase.find("failure") is not None or testcase.find("error") is not None:
        return "FAIL"
    if testcase.find("skipped") is not None:
        return "SKIP"
    return "PASS"


def _profile_records_from_property(testcase: ET.Element) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    properties = testcase.find("properties")
    if properties is None:
        return records

    for prop in properties.findall("property"):
        if prop.get("name") != "profile_metrics":
            continue
        raw_value = prop.get("value") or prop.text or ""
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            records.extend(record for record in parsed if isinstance(record, dict))
    return records


def _case_id(test_name: str) -> str:
    """Return the pytest parameter ID enclosed in the testcase name."""
    if "[" not in test_name or not test_name.endswith("]"):
        return ""
    return test_name.rsplit("[", 1)[1][:-1]


def _model_versions_from_property(
    testcase: ET.Element,
) -> dict[str, dict[str, str]]:
    """Map pytest model case IDs to version records embedded in JUnit."""
    properties = testcase.find("properties")
    if properties is None:
        return {}

    test_case_id = _case_id(testcase.get("name") or "")
    for prop in properties.findall("property"):
        if prop.get("name") != "model_versions":
            continue
        raw_value = prop.get("value") or prop.text or ""
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}

        demo_name = ""
        for record in parsed.values():
            if not isinstance(record, dict):
                continue
            repo_name = Path(str(record.get("repo_id") or "")).name
            suffix = f"_{repo_name}"
            if repo_name and test_case_id.endswith(suffix):
                demo_name = test_case_id[:-len(suffix)]
                break
        if not demo_name:
            return {}
        return _normalize_model_versions({demo_name: parsed})
    return {}


def _normalize_model_versions(parsed: object) -> dict[str, dict[str, str]]:
    """Normalize nested setup version records into a case-ID lookup."""
    if not isinstance(parsed, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for demo_name, model_records in parsed.items():
        if not isinstance(model_records, dict):
            continue
        for record in model_records.values():
            if not isinstance(record, dict):
                continue
            repo_name = Path(str(record.get("repo_id") or "")).name
            if not repo_name:
                continue
            version = record.get("resolved_version") or record.get("version") or ""
            revision = record.get("revision") or ""
            normalized[f"{demo_name}_{repo_name}"] = {
                "version": str(version),
                "revision": str(revision),
            }
    return normalized


def parse_junit(
    junit_path: Path,
) -> tuple[
    list[tuple[str, str, float]],
    list[dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, str]],
]:
    """Read testcase results, profiling records, and prepare metadata."""
    if not junit_path.exists():
        raise FileNotFoundError(f"JUnit XML report not found: {junit_path}")

    root = ET.parse(junit_path).getroot()
    tests: list[tuple[str, str, float]] = []
    profile_records: list[dict[str, Any]] = []
    metadata: dict[str, str] = {}
    model_versions: dict[str, dict[str, str]] = {}
    for testcase in root.findall(".//testcase"):
        name = testcase.get("name") or ""
        try:
            duration = float(testcase.get("time") or 0.0)
        except ValueError:
            duration = 0.0
        tests.append((name, _testcase_status(testcase), duration))
        profile_records.extend(_profile_records_from_property(testcase))
        model_versions.update(_model_versions_from_property(testcase))
        properties = testcase.find("properties")
        if properties is not None:
            for prop in properties.findall("property"):
                prop_name = prop.get("name") or ""
                if prop_name not in {"torq_examples_commit", "dut_sdk_version"}:
                    continue
                prop_value = prop.get("value") or prop.text or ""
                if prop_value:
                    metadata[prop_name] = prop_value
    return tests, profile_records, metadata, model_versions


def load_prepare_metadata(profile_dir: Path) -> dict[str, str]:
    """Load prepare metadata when JUnit properties are unavailable."""
    metadata_path = profile_dir / "prepare_metadata.json"
    try:
        parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        key: str(value)
        for key, value in parsed.items()
        if key in {"torq_examples_commit", "dut_sdk_version"} and value
    }


def load_model_versions(profile_dir: Path) -> dict[str, dict[str, str]]:
    """Load per-model version records when JUnit properties are unavailable."""
    return _normalize_model_versions(_raw_model_versions(profile_dir))


def load_profile_json(profile_dir: Path) -> list[dict[str, Any]]:
    """Load structured profiling records as a fallback for older JUnit reports."""
    if not profile_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for profile_path in sorted(profile_dir.glob("*_profile.json")):
        try:
            parsed = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            records.extend(record for record in parsed if isinstance(record, dict))
    return records


_RUN_METRIC_PATTERNS = (
    # "(2587 ms, TTFT: 541 ms, 18.6 tok/s)" and the ASR variant with ", RTF: 0.20"
    re.compile(
        r"\((\d+(?:\.\d+)?) ms, TTFT: (\d+(?:\.\d+)?) ms, (\d+(?:\.\d+)?) tok/s"
        r"(?:, RTF: (\d+(?:\.\d+)?))?\)"
    ),
    # "(TTFT 1311 ms | prompt 78 tok, gen: 104 tok @ 5.8 tok/s)"
    re.compile(
        r"\(TTFT (\d+(?:\.\d+)?) ms \| prompt (\d+) tok, gen: (\d+) tok @ (\d+(?:\.\d+)?) tok/s\)"
    ),
)


def _raw_model_versions(profile_dir: Path) -> dict[str, Any]:
    versions_path = profile_dir / "setup_model_versions.json"
    try:
        parsed = json.loads(versions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _avg(values: list[float]) -> float:
    return sum(values) / len(values)


def load_run_metrics(profile_dir: Path) -> dict[str, dict[str, str]]:
    """Extract TTFT / tokens-per-second / RTF from *_run.log fallback files.

    Run logs are named {demo}_{org}_{repo}_run.log while report case IDs are
    {demo}_{repo}; match on both ends so the org segment is irrelevant.
    Values are averaged when a demo emits several metric lines (multi-turn).
    """
    if not profile_dir.exists():
        return {}
    demo_repo_pairs: list[tuple[str, str]] = []
    for demo, models in _raw_model_versions(profile_dir).items():
        if not isinstance(models, dict):
            continue
        for record in models.values():
            if isinstance(record, dict) and record.get("repo_id"):
                demo_repo_pairs.append((str(demo), Path(str(record["repo_id"])).name))

    out: dict[str, dict[str, str]] = {}
    for log_path in sorted(profile_dir.glob("*_run.log")):
        stem = log_path.name[: -len("_run.log")]
        case_id = ""
        for demo, repo_name in demo_repo_pairs:
            if stem.startswith(f"{demo}_") and stem.endswith(f"_{repo_name}"):
                case_id = f"{demo}_{repo_name}"
                break
        if not case_id:
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ttfts: list[float] = []
        speeds: list[float] = []
        rtfs: list[float] = []
        for line in text.splitlines():
            m = _RUN_METRIC_PATTERNS[0].search(line)
            if m:
                ttfts.append(float(m.group(2)))
                speeds.append(float(m.group(3)))
                if m.group(4):
                    rtfs.append(float(m.group(4)))
                continue
            m = _RUN_METRIC_PATTERNS[1].search(line)
            if m:
                ttfts.append(float(m.group(1)))
                speeds.append(float(m.group(4)))
        if not ttfts:
            # Run log exists but the demo CLI did not emit any metric line
            # (single-shot demos never do; streaming demos should).
            out[case_id] = {"note": "n/a"}
            continue
        n = len(ttfts)
        suffix = f" (avg of {n})" if n > 1 else ""
        metrics = {
            "TTFT": f"{_avg(ttfts):.0f} ms{suffix}",
            "tok/s": f"{_avg(speeds):.1f}{suffix}",
        }
        if rtfs:
            metrics["RTF"] = f"{_avg(rtfs):.2f}"
        out[case_id] = metrics
    return out


def load_run_metric_gaps(profile_dir: Path) -> list[str]:
    """Case IDs whose run log exists but emits no TTFT/tok/s metric lines."""
    return sorted(cid for cid, m in load_run_metrics(profile_dir).items() if "note" in m)


def load_runtime_version(profile_dir: Path) -> str:
    """Read the torq-runtime version recorded in the DUT pip list log."""
    pip_list_path = profile_dir / "dut_pip_list.log"
    try:
        lines = pip_list_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "torq-runtime":
            return parts[1]
    return ""


def merge_profile_records(*record_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge records while preferring JUnit data and removing duplicate profile cases."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for records in record_groups:
        for record in records:
            key = (
                str(record.get("demo_name", "")),
                str(record.get("case_name", "")),
                str(record.get("model_path", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)
    return merged


def build_html(
    tests: list[tuple[str, str, float]],
    profile_records: list[dict[str, Any]],
    metadata: dict[str, str],
    model_versions: dict[str, dict[str, str]],
    run_metrics: dict[str, dict[str, str]],
    runtime_version: str,
    metric_gaps: list[str],
    out_path: Path,
) -> None:
    """Write the single-page report: run info + test results (7 cols) + profile table."""
    th_style = "border:1px solid #ccc;padding:8px 10px;text-align:left;background:#f2f2f2"
    td_style = "border:1px solid #ccc;padding:8px 10px;text-align:left"
    table_style = 'border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;margin-bottom:24px"'

    info_values = (
        (
            "Torq Examples Commit",
            metadata.get("torq_examples_commit")
            or os.environ.get("TORQ_EXAMPLES_REF", ""),
        ),
        (
            "DUT SDK Version",
            metadata.get("dut_sdk_version")
            or os.environ.get("SYNA_SDK_REVISION", ""),
        ),
        (
            "Torq Runtime Version",
            runtime_version,
        ),
        (
            "Compiler",
            "n/a in this run - models are pre-compiled (see VERSION / revision columns)",
        ),
    )
    info_rows = "".join(
        f'<tr><td style="{td_style}"><strong>{html.escape(label)}</strong></td>'
        f'<td style="{td_style}">{html.escape(value)}</td></tr>'
        for label, value in info_values
    )

    def _display_name(name: str) -> str:
        # The "convert" step currently only downloads pre-compiled models
        # from HuggingFace and validates them (no compilation happens in CI),
        # so report it as "download" to avoid misleading readers.
        return name.replace("test_convert_demo[", "test_download_demo[")

    test_rows: list[str] = []
    for test_name, status, _duration in tests:
        color = {
            "PASS": "#28a745",
            "FAIL": "#d73a49",
            "SKIP": "#6c757d",
        }[status]
        version_record = model_versions.get(_case_id(test_name), {})
        version = version_record.get("version", "")
        revision = version_record.get("revision", "")
        run_record = (
            run_metrics.get(_case_id(test_name), {})
            if test_name.startswith("test_run_demo[")
            else {}
        )
        if run_record.get("note"):
            run_record = {"TTFT": run_record["note"]}
        test_rows.append(
            f'<tr><td style="{td_style}">{html.escape(_display_name(test_name))}</td>'
            f'<td style="{td_style};color:{color};font-weight:bold">{status}</td>'
            f'<td style="{td_style}">{html.escape(version)}</td>'
            f'<td style="{td_style}">{html.escape(revision)}</td>'
            f'<td style="{td_style}">{html.escape(run_record.get("TTFT", ""))}</td>'
            f'<td style="{td_style}">{html.escape(run_record.get("tok/s", ""))}</td>'
            f'<td style="{td_style}">{html.escape(run_record.get("RTF", ""))}</td></tr>'
        )
    if not test_rows:
        test_rows.append(
            f'<tr><td style="{td_style}" colspan="7">No test cases found</td></tr>'
        )
    if metric_gaps:
        gap_list = ", ".join(html.escape(g) for g in metric_gaps)
        test_rows.append(
            f'<tr><td style="{td_style}" colspan="7">'
            f'<em>Note: "n/a" in the TTFT/Throughput columns means the demo CLI does not '
            f'emit these metrics: {gap_list}. For moonshine_streaming, metric printing is '
            f'to be added on the demo side.</em></td></tr>'
        )
    test_header = "".join(
        f'<th style="{th_style}">{html.escape(h)}</th>'
        for h in (
            "TestCase Name",
            "TestCase Result",
            "VERSION",
            "Hugging Face commit (revision)",
            "TTFT",
            "Throughput (tok/s)",
            "RTF",
        )
    )

    profile_rows: list[str] = []
    for record in profile_records:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        demo_label = str(
            record.get("case_name")
            or record.get("demo_name")
            or record.get("model_path")
            or ""
        )
        cells = [f'<td style="{td_style}">{html.escape(demo_label)}</td>']
        for metric in PROFILE_METRICS:
            cells.append(
                f'<td style="{td_style}">{html.escape(str(metrics.get(metric, "")))}</td>'
            )
        profile_rows.append(f"<tr>{''.join(cells)}</tr>")
    if not profile_rows:
        profile_rows.append(
            f'<tr><td style="{td_style}" colspan="8">No profile data found</td></tr>'
        )
    profile_headers = "".join(
        f'<th style="{th_style}">{html.escape(header)}</th>'
        for header in ("demo", *PROFILE_METRICS)
    )

    html_text = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'><title>Test + Profile Summary</title></head>"
        '<body style="font-family:Arial,sans-serif;margin:24px">'
        "<h2>Test + Profile Summary</h2>"
        f'<table {table_style}><tbody>{info_rows}</tbody></table>'
        "<h3>Test Demo Results</h3>"
        f'<table {table_style}><tbody><tr>{test_header}</tr>{"".join(test_rows)}</tbody></table>'
        "<h3>Profile Summary</h3>"
        f'<table {table_style}><tbody><tr>{profile_headers}</tr>{"".join(profile_rows)}</tbody></table>'
        "</body></html>\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote combined summary: {out_path}")


def generate_summary(junit_path: Path, profile_dir: Path, out_path: Path) -> None:
    tests, junit_profile_records, junit_metadata, junit_model_versions = parse_junit(
        junit_path
    )
    file_profile_records = load_profile_json(profile_dir)
    profile_records = merge_profile_records(junit_profile_records, file_profile_records)
    file_metadata = load_prepare_metadata(profile_dir)
    metadata = {**file_metadata, **junit_metadata}
    file_model_versions = load_model_versions(profile_dir)
    model_versions = {**file_model_versions, **junit_model_versions}
    run_metrics = load_run_metrics(profile_dir)
    runtime_version = load_runtime_version(profile_dir)
    metric_gaps = load_run_metric_gaps(profile_dir)
    build_html(
        tests, profile_records, metadata, model_versions, run_metrics,
        runtime_version, metric_gaps, out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--junit",
        type=Path,
        default=Path("TEST-TorqExamplesTest.xml"),
        help="Final pytest JUnit XML report",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing *_profile.json fallback files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("test_and_profile_summary.html"),
        help="Combined HTML output path",
    )
    args = parser.parse_args()
    generate_summary(args.junit, args.profile_dir, args.out)


if __name__ == "__main__":
    main()
