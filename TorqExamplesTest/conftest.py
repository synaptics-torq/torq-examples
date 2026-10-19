"""Repository-local pytest plugin for Torq Examples automation artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

import pytest


SUITE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SUITE_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from TorqExamplesTest.generate_test_summary import generate_summary


EXAMPLES_REPO_ROOT = (
    WORKSPACE_ROOT
    if (WORKSPACE_ROOT / "setup_demos.py").exists()
    else WORKSPACE_ROOT / "torq-examples"
)
LOG_ARTIFACT_SUFFIXES = {".log", ".json", ".xml", ".html"}
_STATE_ATTRIBUTE = "_torq_examples_plugin_state"


class TeeStream:
    """Write text to both the original pytest console and the main log."""

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

    def __getattr__(self, name: str):
        return getattr(self.console, name)


class TorqExamplesPluginState:
    """Resources and artifact paths retained for one pytest session."""

    def __init__(
        self,
        junit_path: Path,
        profile_dir: Path,
        summary_path: Path,
    ) -> None:
        self.junit_path = junit_path
        self.profile_dir = profile_dir
        self.summary_path = summary_path
        self.main_log_path = profile_dir / "main.log"
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file: TextIO | None = None

    def start_tee(self) -> None:
        self.log_file = self.main_log_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        )
        sys.stdout = TeeStream(self.original_stdout, self.log_file)
        sys.stderr = TeeStream(self.original_stderr, self.log_file)

    def stop_tee(self) -> None:
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose report locations while keeping bare pytest fully functional."""
    group = parser.getgroup("torq-examples", "Torq Examples report generation")
    group.addoption(
        "--torq-profile-dir",
        type=Path,
        default=EXAMPLES_REPO_ROOT / "logs",
        help="Directory for Torq Examples logs and profile JSON artifacts",
    )
    group.addoption(
        "--torq-summary-html",
        type=Path,
        default=WORKSPACE_ROOT / "test_and_profile_summary.html",
        help="Combined test and profiling HTML output path",
    )


def _configured_junit_path(config: pytest.Config) -> Path:
    xml_path = getattr(config.option, "xmlpath", None)
    if not xml_path:
        raise pytest.UsageError(
            "Torq Examples reporting requires --junitxml; keep it enabled in pytest.ini"
        )
    path = Path(xml_path)
    if not path.is_absolute():
        path = Path(config.invocation_params.dir) / path
    return path.resolve()


def _resolve_option_path(config: pytest.Config, option_name: str) -> Path:
    path = Path(config.getoption(option_name))
    if not path.is_absolute():
        path = Path(config.invocation_params.dir) / path
    return path.resolve()


def _clean_previous_artifacts(
    junit_path: Path,
    profile_dir: Path,
    summary_path: Path,
) -> list[str]:
    """Delete only known artifacts from the previous pytest session."""
    protected_directories = {
        Path(profile_dir.anchor),
        WORKSPACE_ROOT.resolve(),
        SUITE_DIR.resolve(),
        EXAMPLES_REPO_ROOT.resolve(),
    }
    if profile_dir in protected_directories:
        raise pytest.UsageError(
            f"Refusing to clean unsafe profile directory: {profile_dir}"
        )

    paths_to_remove = [junit_path, summary_path]
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


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Prepare artifact paths and install the console tee before pytest runs."""
    junit_path = _configured_junit_path(config)
    profile_dir = _resolve_option_path(config, "--torq-profile-dir")
    summary_path = _resolve_option_path(config, "--torq-summary-html")
    cleanup_messages = _clean_previous_artifacts(
        junit_path,
        profile_dir,
        summary_path,
    )

    junit_path.parent.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    state = TorqExamplesPluginState(junit_path, profile_dir, summary_path)
    setattr(config, _STATE_ATTRIBUTE, state)
    state.start_tee()

    print("Cleaning artifacts from the previous automation run...")
    for message in cleanup_messages:
        print(message)
    print(f"Main test log: {state.main_log_path}")
    print(f"JUnit report: {state.junit_path}")
    print(f"Combined HTML report: {state.summary_path}")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Generate the combined report after pytest's JUnit plugin closes XML."""
    state = getattr(session.config, _STATE_ATTRIBUTE, None)
    if state is None:
        return

    try:
        generate_summary(
            state.junit_path,
            state.profile_dir,
            state.summary_path,
        )
    except Exception as exc:
        print(
            f"Failed to generate combined HTML report: {exc}",
            file=sys.stderr,
        )
        if exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Always restore process streams and close the suite log."""
    state = getattr(config, _STATE_ATTRIBUTE, None)
    if state is None:
        return
    state.stop_tee()
    delattr(config, _STATE_ATTRIBUTE)
