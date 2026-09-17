#!/usr/bin/env python3
"""
Collect profiling metrics from pulled DUT logs and write an HTML summary.

Usage:
    python3 collect_profile_metrics.py --logs-dir ./log --out ./log/profile_summary.html

This script intentionally uses only the Python standard library so it does not
depend on pandas or any NumPy-linked binary packages.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Dict, List


METRIC_PATTERNS = [
    "Avg inference time",
    "Avg DRAM footprint",
    "Peak DRAM footprint",
    "Avg memory usage",
    "Peak memory usage",
    "Avg CPU usage",
    "Avg NPU usage",
]


def shorten_demo_name(name: str) -> str:
    """Return the substring after 'Synaptics' in `name`.

    If 'Synaptics' is not present, exit the script with an error.
    """
    if 'synaptics' not in name.lower():
        raise SystemExit(f"Demo name does not contain 'Synaptics': {name}")
    m = re.search(r'(?i)synaptics[_/\\-]?(.*)', name)
    if not m:
        raise SystemExit(f"Unable to shorten demo name after 'Synaptics': {name}")
    short = m.group(1).strip()
    if not short:
        raise SystemExit(f"Empty demo name after 'Synaptics' in: {name}")
    return short


def find_log_files(logs_dir: Path) -> List[Path]:
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs dir not found: {logs_dir}")

    files = []
    for path in sorted(logs_dir.iterdir()):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if 'profile' in lower_name and path.suffix in {'.log'}:
            files.append(path)
    print(f"Found files: {files}")
    return files


def parse_metrics_from_text(lines: List[str]) -> Dict[str, str]:
    joined = "\n".join(lines).strip()
    metrics: Dict[str, str] = {}
    for line in lines:
        normalized = line.strip()
        if not normalized:
            continue
        for pattern in METRIC_PATTERNS:
            if pattern.lower() in normalized.lower():
                # Keep the original value text after the metric label, which may look like:
                # "Avg inference time:  16.467 ms" or "Avg CPU usage:       22.2%"
                value_part = normalized.split(pattern, 1)[1].strip()
                if value_part.startswith(":"):
                    value_part = value_part[1:].strip()
                metrics[pattern] = value_part
                break
    metrics['raw_summary'] = joined
    return metrics


def collect(logs_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for log in find_log_files(logs_dir):
        try:
            txt = log.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue

        # shorten demo name by splitting on 'Synaptics'
        demo_short = shorten_demo_name(log.stem)
        metrics = parse_metrics_from_text(txt.splitlines())
        row = {'demo': demo_short, 'logfile': log.name}
        row.update(metrics)
        rows.append(row)
    return rows


def write_html(rows: List[Dict[str, str]], out_path: Path) -> None:
    if not rows:
        print("No profiling metrics found in logs.")
        return

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    # Exclude 'logfile' and 'raw_summary' from the HTML output columns
    ordered = ['demo'] + sorted(k for k in all_keys if k not in {'demo', 'logfile', 'raw_summary'})

    th_style = "border:1px solid #ccc;padding:8px 10px;text-align:left;background:#f2f2f2"
    td_style = "border:1px solid #ccc;padding:8px 10px;text-align:left"
    header = "".join(f"<th style=\"{th_style}\">{html.escape(str(key))}</th>" for key in ordered)
    body_rows = []
    for row in rows:
        cells = []
        for key in ordered:
            value = row.get(key, '')
            cells.append(f"<td style=\"{td_style}\">{html.escape(str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    # Use table attributes and inline styles so Jenkins' HTML sanitizer
    # (which may strip <style> blocks) does not remove the visible borders.
    html_text = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'><title>Profile Summary</title></head>"
        "<body style=\"font-family:Arial,sans-serif;margin:24px\">"
        f"<h2>Profile Summary</h2>"
        f"<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\" style=\"border-collapse:collapse\">"
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        "</body></html>\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding='utf-8')
    print(f"Wrote profile summary (HTML): {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs-dir', type=Path, default=Path('./log'), help='Directory containing pulled DUT logs')
    parser.add_argument('--out', type=Path, default=Path('./log/profile_summary.html'), help='Output HTML file path')
    args = parser.parse_args()

    rows = collect(args.logs_dir)
    print(f"Collected {len(rows)} profiling metric blocks from logs in {args.logs_dir}")
    write_html(rows, args.out)


if __name__ == '__main__':
    main()
