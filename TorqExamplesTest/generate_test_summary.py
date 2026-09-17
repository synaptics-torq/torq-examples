#!/usr/bin/env python3
"""
Generate a combined HTML summary from JUnit XML and profile HTML.

Usage:
  python3 generate_test_summary.py --junit TEST-TorqExamplesTest.xml --profile profile_summary.html --out combined_summary.html

This script uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET


METRIC_KEYS = [
    "Avg inference time",
    "Avg DRAM footprint",
    "Peak DRAM footprint",
    "Avg memory usage",
    "Peak memory usage",
    "Avg CPU usage",
    "Avg NPU usage",
]


def parse_profile_html(path: Path) -> Dict[str, Dict[str, str]]:
    """Parse profile_summary.html table into mapping demo -> metrics dict."""
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="ignore")
    # Extract the first <table>...</table>
    m = re.search(r"<table[^>]*>(.*?)</table>", txt, flags=re.S | re.I)
    if not m:
        return {}
    table_html = m.group(1)
    # Extract headers
    headers = re.findall(r"<th[^>]*>(.*?)</th>", table_html, flags=re.S | re.I)
    headers = [re.sub(r"<[^>]+>", "", h).strip() for h in headers]

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I)
    data: Dict[str, Dict[str, str]] = {}
    for r in rows[1:]:  # skip header row
        cols = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S | re.I)
        cols = [re.sub(r"<[^>]+>", "", c).strip() for c in cols]
        if not cols:
            continue
        # Map headers to cols
        rowmap = {headers[i]: cols[i] if i < len(cols) else "" for i in range(len(headers))}
        demo_name = rowmap.get("demo") or rowmap.get("Demo") or rowmap.get(headers[0])
        if demo_name:
            data[demo_name] = rowmap
    return data


def parse_junit(junit_path: Path) -> List[Tuple[str, str, float]]:
    """Return list of (test_label, status, time_seconds)"""
    if not junit_path.exists():
        return []
    tree = ET.parse(str(junit_path))
    root = tree.getroot()
    tests: List[Tuple[str, str, float]] = []
    # testsuite may contain testcase elements
    for tc in root.findall('.//testcase'):
        name = tc.get('name') or ''
        classname = tc.get('classname') or ''
        label = f"{classname}.{name}" if classname else name
        time = 0.0
        try:
            time = float(tc.get('time') or 0.0)
        except Exception:
            time = 0.0
        status = 'PASS'
        if tc.find('failure') is not None or tc.find('error') is not None:
            status = 'FAIL'
        elif tc.find('skipped') is not None:
            status = 'SKIP'
        tests.append((label, status, time))
    return tests


def find_best_demo_match(test_label: str, profile_keys: List[str]) -> str | None:
    low = test_label.lower()
    # exact substring match
    for k in profile_keys:
        if k.lower() in low or low in k.lower():
            return k
    # fallback: token match
    for k in profile_keys:
        for token in re.split(r'[^A-Za-z0-9]+', k.lower()):
            if token and token in low:
                return k
    return None


def build_html(tests: List[Tuple[str, str, float]], profile_map: Dict[str, Dict[str, str]], out_path: Path) -> None:
    th_style = "border:1px solid #ccc;padding:8px 10px;text-align:left;background:#f2f2f2"
    td_style = "border:1px solid #ccc;padding:8px 10px;text-align:left"

    # We'll make a single table with sections. Use 8 columns total (profile section requires 8).
    total_cols = 8

    # Build rows for testcases. Prefer one aggregated row per demo found in profile_map
    def _matches_demo(label: str, demo: str) -> bool:
        low = (label or "").lower()
        dlow = (demo or "").lower()
        if not low or not dlow:
            return False
        if dlow in low or low in dlow:
            return True
        for token in re.split(r'[^A-Za-z0-9]+', dlow):
            if token and token in low:
                return True
        return False

    test_rows_html = []
    profile_keys = list(profile_map.keys())
    # Build Test Demo Results rows directly from the JUnit XML `name` values.
    # Use the last segment after '.' as the TestCase Name and show the status (PASS/FAIL/SKIP).
    test_rows_html = []
    for label, status, _t in tests:
        # label is typically 'ClassName.test_name' or similar; take last segment
        tc_name = (label or '').split('.')[-1]
        color = "#28a745" if status == 'PASS' else ("#d73a49" if status == 'FAIL' else "#6c757d")
        test_cell = f"<td style=\"{td_style}\">{html.escape(tc_name)}</td>"
        result_cell = f"<td style=\"{td_style};color:{color};font-weight:bold\">{html.escape(status)}</td>"
        pad = ''.join(f"<td style=\"{td_style}\"></td>" for _ in range(total_cols - 2))
        test_rows_html.append(f"<tr>{test_cell}{result_cell}{pad}</tr>")

    if not test_rows_html:
        test_rows_html = [f"<tr><td style=\"{td_style}\" colspan=\"{total_cols}\">No test cases found</td></tr>"]

    # Build rows for profile summary
    profile_header_cells = ''.join(f"<th style=\"{th_style}\">{html.escape(h)}</th>" for h in [
        "demo",
        "Avg CPU usage",
        "Avg DRAM footprint",
        "Avg NPU usage",
        "Avg inference time",
        "Avg memory usage",
        "Peak DRAM footprint",
        "Peak memory usage",
    ])

    profile_rows = []
    for demo, metrics in profile_map.items():
        cells = [f"<td style=\"{td_style}\">{html.escape(demo)}</td>"]
        # Order metrics as requested
        for key in ["Avg CPU usage", "Avg DRAM footprint", "Avg NPU usage", "Avg inference time", "Avg memory usage", "Peak DRAM footprint", "Peak memory usage"]:
            cells.append(f"<td style=\"{td_style}\">{html.escape(metrics.get(key, ''))}</td>")
        profile_rows.append(f"<tr>{''.join(cells)}</tr>")

    if not profile_rows:
        profile_rows = [f"<tr><td style=\"{td_style}\" colspan=\"{total_cols}\">No profile data found</td></tr>"]

    # Build DUT and GitHub info
    torq_commit = os.environ.get('TORQ_EXAMPLES_REF', '')
    sdk_rev = os.environ.get('SYNA_SDK_REVISION', '')

    info_rows = []
    pad_cells = ''.join(f'<td style="{td_style}"></td>' for _ in range(total_cols - 2))
    info_rows.append(f"<tr><td style=\"{td_style}\"><strong>Torq Examples Commit</strong></td><td style=\"{td_style}\">{html.escape(torq_commit)}</td>{pad_cells}</tr>")
    info_rows.append(f"<tr><td style=\"{td_style}\"><strong>DUT SDK Version</strong></td><td style=\"{td_style}\">{html.escape(sdk_rev)}</td>{pad_cells}</tr>")

    # Testcases header padding and header HTML (precompute to avoid statements inside expression)
    header_pad = ''.join(f'<th style="{th_style}"></th>' for _ in range(total_cols - 2))
    testcases_header_html = f"<tr><th style=\"{th_style}\">TestCase Name</th><th style=\"{th_style}\">TestCase Result</th>{header_pad}</tr>"

    # Assemble full HTML
    html_text = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'><title>Test + Profile Summary</title></head>"
        "<body style=\"font-family:Arial,sans-serif;margin:24px\">"
        f"<h2>Test + Profile Summary</h2>"
        f"<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\" style=\"border-collapse:collapse\">"
        # DUT/GitHub info rows
        f"<tbody>{''.join(info_rows)}"
        # Test Demo Results title row
        f"<tr><td style=\"{td_style}\" colspan=\"{total_cols}\"><strong>Test Demo Results</strong></td></tr>"
        # Testcases header
        f"{testcases_header_html}"
        # Test rows
        f"{''.join(test_rows_html)}"
        # Profile Summary title
        f"<tr><td style=\"{td_style}\" colspan=\"{total_cols}\"><strong>Profile Summary</strong></td></tr>"
        # Profile header
        f"<tr>{profile_header_cells}</tr>"
        # Profile rows
        f"{''.join(profile_rows)}"
        f"</tbody></table></body></html>\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding='utf-8')
    print(f"Wrote combined summary: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--junit', type=Path, default=Path('TEST-TorqExamplesTest.xml'))
    parser.add_argument('--profile', type=Path, default=Path('profile_summary.html'))
    parser.add_argument('--out', type=Path, default=Path('test_and_profile_summary.html'))
    args = parser.parse_args()

    profile_map = parse_profile_html(args.profile)
    tests = parse_junit(args.junit)
    build_html(tests, profile_map, args.out)


if __name__ == '__main__':
    main()
