#!/usr/bin/env python3
"""Run the DUT pytest suite and always generate the combined HTML report."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TextIO

try:
    from TorqExamplesRefactorTest.generate_test_summary import generate_summary
except ModuleNotFoundError:
    from generate_test_summary import generate_summary


SUITE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SUITE_DIR.parent
EXAMPLES_REPO_ROOT = (
    WORKSPACE_ROOT
    if (WORKSPACE_ROOT / "setup_demos.py").exists()
    else WORKSPACE_ROOT / "torq-examples"
)
LOG_ARTIFACT_SUFFIXES = {".log", ".json", ".xml", ".html"}


class TeeStream:
    """Write text to the original console and the suite's main log."""

    def __init__(self, console: TextIO, log_file: TextIO) -> None:
        self.console = console
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.console.isatty()


def clean_previous_artifacts(
    junit_path: Path, profile_dir: Path, out_path: Path
) -> list[str]:
    """Delete artifacts from the previous run before pytest starts."""
    resolved_profile_dir = profile_dir.resolve()
    protected_directories = {
        Path(resolved_profile_dir.anchor),
        WORKSPACE_ROOT.resolve(),
        SUITE_DIR.resolve(),
        EXAMPLES_REPO_ROOT.resolve(),
    }
    if resolved_profile_dir in protected_directories:
        raise ValueError(
            f"Refusing to clean unsafe profile directory: {profile_dir}"
        )

    paths_to_remove = [junit_path, out_path]
    if profile_dir.exists():
        paths_to_remove.extend(
            path
            for path in profile_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in LOG_ARTIFACT_SUFFIXES
        )

    removed: set[Path] = set()
    messages: list[str] = []
    for path in paths_to_remove:
        normalized = path.resolve()
        if normalized in removed or not path.exists():
            continue
        path.unlink()
        removed.add(normalized)
        messages.append(f"Removed previous artifact: {path}")
    return messages


def run_and_stream(command: list[str], env: dict[str, str]) -> int:
    """Run pytest and stream its merged output to both console and main.log."""
    with subprocess.Popen(
        command,
        cwd=WORKSPACE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("Failed to capture pytest output")
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--junit",
        type=Path,
        default=WORKSPACE_ROOT / "TEST-TorqExamplesRefactorTest.xml",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=EXAMPLES_REPO_ROOT / "logs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=WORKSPACE_ROOT / "test_and_profile_summary.html",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional pytest arguments; place them after --",
    )
    args = parser.parse_args()

    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    cleanup_messages = clean_previous_artifacts(
        args.junit, args.profile_dir, args.out
    )

    args.junit.parent.mkdir(parents=True, exist_ok=True)
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    main_log_path = args.profile_dir / "main.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(SUITE_DIR / "test_demo_execution.py"),
        "-q",
        f"--junitxml={args.junit}",
        *pytest_args,
    ]
    subprocess_env = os.environ.copy()
    existing_pythonpath = subprocess_env.get("PYTHONPATH", "")
    subprocess_env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (str(SUITE_DIR), str(WORKSPACE_ROOT), existing_pythonpath)
        if path
    )
    subprocess_env["PYTHONUNBUFFERED"] = "1"

    with main_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee_stdout = TeeStream(sys.stdout, log_file)
        tee_stderr = TeeStream(sys.stderr, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print("Cleaning artifacts from the previous automation run...")
            for message in cleanup_messages:
                print(message)
            print(f"Main test log: {main_log_path}")
            print(f"Running: {' '.join(command)}")
            pytest_returncode = run_and_stream(command, subprocess_env)

            try:
                generate_summary(args.junit, args.profile_dir, args.out)
            except Exception as exc:
                print(
                    f"Failed to generate combined HTML report: {exc}",
                    file=sys.stderr,
                )
                return pytest_returncode or 1

            print(f"Pytest exit code: {pytest_returncode}")
            return pytest_returncode


if __name__ == "__main__":
    raise SystemExit(main())
