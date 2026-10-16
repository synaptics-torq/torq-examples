#!/usr/bin/env python3
"""Generate one TorqExamples-style HTML report from JUnit XML and profile data."""

from __future__ import annotations

import argparse
import html
import json
import os
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
    versions_path = profile_dir / "setup_model_versions.json"
    try:
        parsed = json.loads(versions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _normalize_model_versions(parsed)


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
    out_path: Path,
) -> None:
    """Write the same single-table, inline-style report used by TorqExamplesTest."""
    th_style = "border:1px solid #ccc;padding:8px 10px;text-align:left;background:#f2f2f2"
    td_style = "border:1px solid #ccc;padding:8px 10px;text-align:left"
    total_cols = 8

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
    )
    info_rows: list[str] = []
    info_padding = "".join(
        f'<td style="{td_style}"></td>' for _ in range(total_cols - 2)
    )
    for label, value in info_values:
        info_rows.append(
            f'<tr><td style="{td_style}"><strong>{html.escape(label)}</strong></td>'
            f'<td style="{td_style}">{html.escape(value)}</td>{info_padding}</tr>'
        )

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
        padding = "".join(
            f'<td style="{td_style}"></td>' for _ in range(total_cols - 4)
        )
        test_rows.append(
            f'<tr><td style="{td_style}">{html.escape(test_name)}</td>'
            f'<td style="{td_style};color:{color};font-weight:bold">{status}</td>'
            f'<td style="{td_style}">{html.escape(version)}</td>'
            f'<td style="{td_style}">{html.escape(revision)}</td>'
            f"{padding}</tr>"
        )
    if not test_rows:
        test_rows.append(
            f'<tr><td style="{td_style}" colspan="{total_cols}">No test cases found</td></tr>'
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
            f'<tr><td style="{td_style}" colspan="{total_cols}">No profile data found</td></tr>'
        )

    test_header_padding = "".join(
        f'<th style="{th_style}"></th>' for _ in range(total_cols - 4)
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
        '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse">'
        f"<tbody>{''.join(info_rows)}"
        f'<tr><td style="{td_style}" colspan="{total_cols}"><strong>Test Demo Results</strong></td></tr>'
        f'<tr><th style="{th_style}">TestCase Name</th>'
        f'<th style="{th_style}">TestCase Result</th>'
        f'<th style="{th_style}">VERSION</th>'
        f'<th style="{th_style}">Hugging Face commit (revision)</th>'
        f"{test_header_padding}</tr>"
        f"{''.join(test_rows)}"
        f'<tr><td style="{td_style}" colspan="{total_cols}"><strong>Profile Summary</strong></td></tr>'
        f"<tr>{profile_headers}</tr>"
        f"{''.join(profile_rows)}"
        "</tbody></table></body></html>\n"
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
    build_html(tests, profile_records, metadata, model_versions, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--junit",
        type=Path,
        default=Path("TEST-TorqExamplesRefactorTest.xml"),
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
