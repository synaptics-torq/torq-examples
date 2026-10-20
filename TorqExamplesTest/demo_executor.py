"""Drive real TorqExamples demo validation from the host against an ADB DUT.

The module acquires a DUT through Torq board control, pushes the repository,
and runs setup, inference, and profiling commands on that DUT over ADB. Missing
Torq runtime support or failure to acquire a DUT is treated as a test failure;
there is no local-execution fallback.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from TorqExamplesTest.demo_registry import REPO_ROOT, demo_registry, repo_demo_names


PROFILE_METRIC_LABELS = (
    "Avg CPU usage",
    "Avg DRAM footprint",
    "Avg NPU usage",
    "Avg inference time",
    "Avg memory usage",
    "Peak DRAM footprint",
    "Peak memory usage",
)

SETUP_PATHS_BEGIN = "__TORQ_SETUP_PATHS_BEGIN__"
SETUP_PATHS_END = "__TORQ_SETUP_PATHS_END__"


def _image_decoder_prefix(vmfb_path: str) -> str:
    """Return the shared prefix from an A/B image-decoder VMFB path."""
    match = re.fullmatch(
        r"(?P<prefix>.+_)[AB](?:_[^/]*)?\.vmfb(?:\..*)?",
        vmfb_path,
    )
    if match is None:
        raise RuntimeError(
            "Could not derive image-decoder prefix from VMFB path: "
            f"{vmfb_path}"
        )
    return match.group("prefix")


# The host board utilities are shipped under TorqExamplesTest/torq as
# a top-level ``torq`` namespace package. Put its parent ahead of site-packages
# so an unrelated installed package named ``torq`` cannot shadow it.
_BUNDLED_TORQ_PARENT = Path(__file__).resolve().parent
_bundled_torq_parent_text = str(_BUNDLED_TORQ_PARENT)
if _bundled_torq_parent_text not in sys.path:
    sys.path.insert(0, _bundled_torq_parent_text)

try:  # pragma: no cover - exercised only in host-side board environments.
    from torq.lab.pipeline.remote import (
        RemoteCommandError,
        remote_command_runner_factory,
    )
    from torq.utils.boards import create_boards_control
except ImportError as exc:  # pragma: no cover - environment-specific.
    raise RuntimeError(
        "Torq host board utilities are unavailable. Expected bundled sources at "
        f"{_BUNDLED_TORQ_PARENT / 'torq'} or a compatible installed Torq package."
    ) from exc


@dataclass
class DemoExecutionResult:
    demo_name: str
    status: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    log_path: str | None = None
    model_dirs: dict[str, str] = field(default_factory=dict)
    model_versions: dict[str, dict[str, object]] = field(default_factory=dict)
    profile_records: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class AdbTarget:
    serial: str


@dataclass(frozen=True)
class DemoRunCommand:
    case_name: str
    command: str
    input_description: str | None = None


@dataclass(frozen=True)
class DemoProfileCommand:
    case_name: str
    model_path: str
    command: str


def parse_profile_metrics(output: str) -> dict[str, str]:
    """Extract the seven metrics emitted by torq-examples/profile.py."""
    metrics: dict[str, str] = {}
    for line in output.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        for label in PROFILE_METRIC_LABELS:
            match = re.search(re.escape(label), normalized, flags=re.IGNORECASE)
            if match is None:
                continue
            value = normalized[match.end():].strip()
            if value.startswith(":"):
                value = value[1:].strip()
            metrics[label] = value
            break
    return metrics


def parse_setup_model_paths(output: str) -> dict[str, dict[str, str]]:
    """Parse model paths captured from utils.model_setup.download_models()."""
    begin = output.rfind(SETUP_PATHS_BEGIN)
    end = output.find(SETUP_PATHS_END, begin + len(SETUP_PATHS_BEGIN))
    if begin < 0 or end < 0:
        raise RuntimeError("Model path discovery output is missing its JSON markers")
    payload = output[begin + len(SETUP_PATHS_BEGIN):end].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Model path discovery returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Model path discovery result must be a JSON object")

    result: dict[str, dict[str, str]] = {}
    for demo_name, model_paths in parsed.items():
        if not isinstance(demo_name, str) or not isinstance(model_paths, dict):
            raise RuntimeError("Model path discovery result has an invalid demo entry")
        normalized: dict[str, str] = {}
        for model_name, model_path in model_paths.items():
            if not isinstance(model_name, str) or not isinstance(model_path, str):
                raise RuntimeError(
                    f"Model path discovery result for {demo_name} must map strings to strings"
                )
            normalized[model_name] = model_path
        if not normalized:
            raise RuntimeError(f"No model paths were captured for demo '{demo_name}'")
        result[demo_name] = normalized
    return result


class CommandExecutionError(RuntimeError):
    """A failed host or DUT command with its complete process result."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        stdout: str = "",
        stderr: str = "",
        command: str | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.command = command


class HostDemoRunner:
    """Board-aware helper used by the host-side automation layer."""

    def __init__(
        self,
        repo_root: Path | None = None,
        remote_root: str = "/home/torq-examples",
    ):
        self.repo_root = repo_root or REPO_ROOT
        self.board_control_file = str(Path(__file__).resolve().parent / "board_control.json")
        self.remote_root = remote_root
        self._board_context: Any | None = None
        self._remote_runner: Any | None = None
        self._target: AdbTarget | None = None
        self._prepared_dut_environments: set[tuple[str, str]] = set()
        self._setup_attempted_demos: set[tuple[str, str, str]] = set()

    def _build_target(self) -> AdbTarget:
        """Acquire and retain one Torq-managed DUT until ``close()`` is called."""
        if self._target is not None:
            print(
                "[demo_executor] reusing locked DUT target: "
                f"serial={self._target.serial}"
            )
            return self._target

        print("[demo_executor] acquiring DUT board target via torq board control...")

        control = create_boards_control(self.board_control_file, backend="adb")
        board_context = control.board()
        board_entered = False
        try:
            board_user, serial, private_key = board_context.__enter__()
            board_entered = True
            remote_address = control.get_remote_address(board_user, serial)
            remote_runner = remote_command_runner_factory(
                remote_address,
                timeout=30 * 60,
                ssh_multiplex=True,
                ssh_port=22,
                ssh_private_key=private_key,
            )
        except Exception as exc:
            if board_entered:
                board_context.__exit__(type(exc), exc, exc.__traceback__)
            print("[demo_executor] failed to acquire DUT board target")
            raise RuntimeError("Failed to acquire a DUT board target for demo execution.") from exc

        self._board_context = board_context
        self._remote_runner = remote_runner
        self._target = AdbTarget(serial=serial)
        print(f"[demo_executor] acquired and locked DUT target: serial={serial}")
        return self._target

    def close(self) -> None:
        """Release the Torq board lock held by this runner."""
        board_context = self._board_context
        remote_runner = self._remote_runner
        target = self._target
        self._board_context = None
        self._remote_runner = None
        self._target = None
        try:
            if remote_runner is not None:
                remote_runner.close()
        finally:
            if board_context is not None:
                board_context.__exit__(None, None, None)
        if target is not None:
            print(f"[demo_executor] released DUT target: serial={target.serial}")

    def _resolve_demo_entry(self, demo_name: str) -> str:
        specs = demo_registry()
        spec = specs.get(demo_name)
        if spec is None:
            raise KeyError(f"Unknown demo: {demo_name}")
        if spec.inference_entry is None:
            raise ValueError(f"No inference entry configured for demo '{demo_name}'")
        entry = self.repo_root / spec.inference_entry
        if not entry.exists():
            raise FileNotFoundError(f"Demo inference entry not found: {entry}")
        return str(entry)

    def _resolve_profiling_entry(self, demo_name: str) -> str:
        spec = self._spec_for_demo(demo_name)
        if spec.profiling_entry is None:
            raise ValueError(f"No profiling entry configured for demo '{demo_name}'")
        entry = self.repo_root / spec.profiling_entry
        if not entry.exists():
            raise FileNotFoundError(f"Demo profiling entry not found: {entry}")
        return spec.profiling_entry.replace("\\", "/")

    def _write_log(self, demo_name: str, stdout: str, stderr: str, command: str | None = None) -> str:
        log_dir = self.repo_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{demo_name}.log"
        content = (stdout + "\n" + stderr).strip()
        if command:
            content = f"COMMAND: {command}\n\n{content}" if content else f"COMMAND: {command}"
        log_path.write_text(content + "\n", encoding="utf-8")
        return str(log_path)

    def _write_profile_results(
        self,
        artifact_name: str,
        records: list[dict[str, object]],
    ) -> str:
        log_dir = self.repo_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        result_path = log_dir / f"{artifact_name}_profile.json"
        result_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(result_path)

    def _artifact_name(self, demo_name: str, model_path: str | None = None) -> str:
        """Return a filesystem-safe artifact prefix for one pytest model case."""
        if model_path is None:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", demo_name).strip("_")
        repo_id = model_path.partition("/models/")[2] or PurePosixPath(model_path).name
        case_name = f"{demo_name}_{repo_id}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name).strip("_")

    def _setup_model_paths_file(self) -> Path:
        return self.repo_root / "logs" / "setup_model_paths.json"

    def _setup_model_versions_file(self) -> Path:
        return self.repo_root / "logs" / "setup_model_versions.json"

    def _write_setup_model_paths(
        self,
        model_paths: dict[str, dict[str, str]],
    ) -> str:
        result_path = self._setup_model_paths_file()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(model_paths, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(result_path)

    def _read_setup_model_paths(self) -> dict[str, dict[str, str]]:
        result_path = self._setup_model_paths_file()
        if not result_path.exists():
            return {}
        return parse_setup_model_paths(
            f"{SETUP_PATHS_BEGIN}\n{result_path.read_text(encoding='utf-8')}"
            f"{SETUP_PATHS_END}\n"
        )

    def _record_setup_model_versions(
        self,
        target: AdbTarget,
        demo_name: str,
        model_dirs: dict[str, str],
    ) -> dict[str, dict[str, object]]:
        """Read setup manifests and record the actual version used per model repo."""
        records: dict[str, dict[str, object]] = {}
        for model_name, model_dir in model_dirs.items():
            manifest_path = f"{model_dir.rstrip('/')}/.manifest.json"
            manifest_output = self._run_remote_command(
                target,
                f"if [ -f {shlex.quote(manifest_path)} ]; then "
                f"cat {shlex.quote(manifest_path)}; else printf '{{}}'; fi",
            ).strip()
            try:
                manifest = json.loads(manifest_output or "{}")
            except json.JSONDecodeError:
                manifest = {}
            if not isinstance(manifest, dict):
                manifest = {}

            inferred_repo_id = model_dir.partition("/models/")[2]
            repo_id = str(
                manifest.get("repo_id")
                or inferred_repo_id
                or PurePosixPath(model_dir).name
            )
            version = manifest.get("version")
            revision = manifest.get("revision")
            if not manifest:
                resolved_version = "UNKNOWN (manifest missing)"
            elif version is None:
                resolved_version = "HEAD (untracked)"
            else:
                resolved_version = str(version)

            record: dict[str, object] = {
                "repo_id": repo_id,
                "model_path": model_dir,
                "version": version,
                "revision": revision,
                "resolved_version": resolved_version,
                "manifest_path": manifest_path,
            }
            records[model_name] = record
            revision_text = str(revision) if revision else "not recorded"
            print(
                f"[demo_executor] setup model version: demo={demo_name} "
                f"model={model_name} repo={repo_id} version={resolved_version} "
                f"revision={revision_text}"
            )

        result_path = self._setup_model_versions_file()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        all_versions: dict[str, object] = {}
        if result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    all_versions.update(existing)
            except (OSError, json.JSONDecodeError):
                pass
        all_versions[demo_name] = records
        result_path.write_text(
            json.dumps(all_versions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[demo_executor] setup model versions saved to {result_path}")
        return records

    def _discover_setup_model_paths(
        self,
        demo_names: list[str],
        remote_repo: str,
    ) -> dict[str, dict[str, str]]:
        """Build download_models()-equivalent DUT paths without importing setup code."""
        specs = demo_registry()
        models_root = PurePosixPath(remote_repo) / "models"
        model_paths: dict[str, dict[str, str]] = {}
        missing_demos: list[str] = []
        for demo_name in demo_names:
            repositories = specs[demo_name].setup_model_repositories
            if not repositories:
                missing_demos.append(demo_name)
                continue
            model_paths[demo_name] = {
                model_name: str(models_root / repo_id)
                for model_name, repo_id in repositories
            }
        if missing_demos:
            raise RuntimeError(
                "No setup model repository metadata configured for demos: "
                + ", ".join(missing_demos)
            )
        self._write_setup_model_paths(model_paths)
        return model_paths

    def _spec_for_demo(self, demo_name: str):
        spec = demo_registry().get(demo_name)
        if spec is None:
            raise KeyError(f"Unknown demo: {demo_name}")
        return spec

    def _run_remote_command(self, target: AdbTarget, command: str, timeout: int = 300) -> str:
        """Run ``adb shell`` and keep stdout and stderr separate.

        This helper is intended for ordinary DUT shell operations such as file
        discovery, directory creation, dependency installation, and setup. It
        returns stdout on success; on failure, ``CommandExecutionError`` carries
        stdout and stderr separately.
        """
        full_cmd = ["adb", "-s", target.serial, "shell", command]
        completed = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise CommandExecutionError(
                f"Remote command failed on {target.serial}: {command}",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                command=command,
            )
        return completed.stdout

    def _run_remote_command_combined(
        self,
        target: AdbTarget,
        command: str,
        timeout: int = 300,
    ) -> str:
        """Run ``adb shell`` with stderr merged into stdout.

        This helper is intended for inference and profiling, where complete
        ordered output must be retained in one log stream. It also converts a
        command timeout into ``CommandExecutionError`` with return code 124 and
        preserves any partial output produced before the timeout.
        """
        full_cmd = ["adb", "-s", target.serial, "shell", command]
        try:
            completed = subprocess.run(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial_output = exc.stdout or ""
            if isinstance(partial_output, bytes):
                partial_output = partial_output.decode(errors="replace")
            raise CommandExecutionError(
                f"Remote command timed out after {timeout}s on {target.serial}: {command}",
                returncode=124,
                stdout=partial_output,
                command=command,
            ) from exc
        combined_output = completed.stdout or ""
        if completed.returncode != 0:
            raise CommandExecutionError(
                f"Remote command failed on {target.serial}: {command}",
                returncode=completed.returncode,
                stdout=combined_output,
                command=command,
            )
        return combined_output

    def _copy_files(
        self,
        target: AdbTarget,
        src: str,
        dst: str,
        *,
        recursive: bool = False,
        board_dst: bool,
    ) -> None:
        """Transfer files through Torq's board transport abstraction."""
        runner = self._remote_runner
        if runner is None or self._target != target:
            raise RuntimeError("A Torq-managed DUT must be acquired before file transfer")
        try:
            runner.copy_files(
                src,
                dst,
                recursive=recursive,
                board_dst=board_dst,
            )
        except RemoteCommandError as exc:
            direction = "to" if board_dst else "from"
            raise CommandExecutionError(
                f"File transfer {direction} DUT failed on {target.serial}: {src} -> {dst}",
                returncode=1,
                stderr=exc.output,
            ) from exc

    def _push_repo_to_dut(self, target: AdbTarget) -> None:
        local_repo = str(self.repo_root)
        remote_parent = str(PurePosixPath(self.remote_root).parent)
        print(f"[test_prepare] step 3/8: push repository: {local_repo} -> {self.remote_root}")
        self._run_remote_command(target, f"mkdir -p {shlex.quote(remote_parent)}")
        self._copy_files(
            target,
            local_repo,
            remote_parent,
            recursive=True,
            board_dst=True,
        )
        self._run_remote_command(
            target,
            f"rm -rf {shlex.quote(self.remote_root + '/logs')}",
        )

    def _demo_app_dir(self, demo_name: str, remote_repo: str) -> str:
        spec = self._spec_for_demo(demo_name)
        return f"{remote_repo.rstrip('/')}/{spec.app_dir}"

    def _models_root(self, remote_repo: str) -> str:
        return f"{remote_repo.rstrip('/')}/models/Synaptics"

    def _demo_model_dirs(
        self,
        target: AdbTarget,
        demo_name: str,
        remote_repo: str,
        model_path: str | None = None,
    ) -> list[str]:
        spec = self._spec_for_demo(demo_name)
        if not spec.asset_groups:
            raise RuntimeError(f"No asset discovery rules configured for demo '{demo_name}'")

        if model_path is not None:
            if not self._model_dir_has_expected_assets(target, model_path, spec.asset_groups):
                raise RuntimeError(
                    f"Expected model assets for demo '{demo_name}' are missing or incomplete: "
                    f"{model_path}"
                )
            return [model_path]

        planned_paths = self._read_setup_model_paths().get(demo_name, {})
        if planned_paths:
            planned_dirs = list(dict.fromkeys(planned_paths.values()))
            missing_or_invalid = [
                model_dir
                for model_dir in planned_dirs
                if not self._model_dir_has_expected_assets(target, model_dir, spec.asset_groups)
            ]
            if missing_or_invalid:
                raise RuntimeError(
                    f"Expected model assets for demo '{demo_name}' are missing or incomplete: "
                    f"{', '.join(missing_or_invalid)}"
                )
            print(
                f"[demo_executor] model directories from download_models() for {demo_name}: "
                f"{', '.join(planned_dirs)}"
            )
            return planned_dirs

        models_root = self._models_root(remote_repo)
        dirs_output = self._run_remote_command(
            target,
            f"find {shlex.quote(models_root)} -mindepth 1 -maxdepth 1 -type d",
        )
        candidate_dirs = sorted(line.strip() for line in dirs_output.splitlines() if line.strip())
        if spec.model_dir_patterns:
            candidate_dirs = [
                candidate_dir
                for candidate_dir in candidate_dirs
                if any(
                    fnmatch.fnmatchcase(PurePosixPath(candidate_dir).name, pattern)
                    for pattern in spec.model_dir_patterns
                )
            ]
        matched_dirs: list[str] = []
        for candidate_dir in candidate_dirs:
            if self._model_dir_has_expected_assets(target, candidate_dir, spec.asset_groups):
                matched_dirs.append(candidate_dir)

        if matched_dirs:
            print(
                f"[demo_executor] discovered model directories for {demo_name}: "
                f"{', '.join(matched_dirs)}"
            )
            return matched_dirs

        raise RuntimeError(
            f"Could not discover generated model assets for demo '{demo_name}' under {models_root}"
        )

    def _model_dir_has_expected_assets(
        self,
        target: AdbTarget,
        model_dir: str,
        asset_groups: tuple[tuple[str, ...], ...],
    ) -> bool:
        try:
            candidate_files = self._list_remote_files(target, model_dir)
        except CommandExecutionError:
            return False
        return all(
            any(
                fnmatch.fnmatchcase(PurePosixPath(remote_file).name, pattern)
                for remote_file in candidate_files
                for pattern in group
            )
            for group in asset_groups
        )

    def _demo_model_dir(self, target: AdbTarget, demo_name: str, remote_repo: str) -> str:
        return self._demo_model_dirs(target, demo_name, remote_repo)[0]

    def _list_remote_files(
        self,
        target: AdbTarget,
        search_root: str,
        *,
        maxdepth: int | None = None,
    ) -> list[str]:
        """List usable files without relying on GNU-specific find predicates."""
        depth_arg = f" -maxdepth {maxdepth}" if maxdepth is not None else ""
        stdout = self._run_remote_command(
            target,
            f"find {shlex.quote(search_root)}{depth_arg} -type f",
        )
        root_path = PurePosixPath(search_root)
        usable_files: list[str] = []
        for line in stdout.splitlines():
            remote_file = line.strip()
            if not remote_file:
                continue
            file_path = PurePosixPath(remote_file)
            try:
                relative_path = file_path.relative_to(root_path)
            except ValueError:
                continue
            if any(part.startswith(".") for part in relative_path.parts):
                continue
            if file_path.name.endswith((".metadata", ".lock", ".incomplete")):
                continue
            usable_files.append(remote_file)
        return sorted(usable_files)

    def _find_remote_first_match(
        self,
        target: AdbTarget,
        search_root: str,
        patterns: list[str],
        *,
        maxdepth: int = 4,
    ) -> str:
        matches = self._find_remote_matches(
            target,
            search_root,
            patterns,
            maxdepth=maxdepth,
        )
        if matches:
            return matches[0]
        raise RuntimeError(
            f"No remote match found under {search_root} for patterns: {', '.join(patterns)}"
        )

    def _find_remote_matches(
        self,
        target: AdbTarget,
        search_root: str,
        patterns: list[str],
        *,
        maxdepth: int = 4,
    ) -> list[str]:
        return [
            remote_file
            for remote_file in self._list_remote_files(
                target,
                search_root,
                maxdepth=maxdepth,
            )
            if any(
                fnmatch.fnmatchcase(PurePosixPath(remote_file).name, pattern)
                for pattern in patterns
            )
        ]

    def _ensure_remote_test_wav(self, target: AdbTarget, demo_name: str, remote_repo: str) -> str:
        remote_wav = f"{remote_repo.rstrip('/')}/refactor_inputs/{demo_name}.wav"
        command = (
            f"python3 -c \"import pathlib,wave; "
            f"p=pathlib.Path({remote_wav!r}); "
            f"p.parent.mkdir(parents=True, exist_ok=True); "
            f"w=wave.open(str(p),'wb'); "
            f"w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); "
            f"w.writeframes(b'\\x00\\x00'*16000); w.close()\""
        )
        self._run_remote_command(target, command)
        return remote_wav

    def _install_demo_requirements_on_dut(self, target: AdbTarget, demo_name: str, remote_repo: str) -> None:
        remote_app_dir = self._demo_app_dir(demo_name, remote_repo)
        print(f"[demo_executor] installing per-demo requirements for {demo_name} in {remote_app_dir}")
        self._run_remote_command(
            target,
            f"cd {shlex.quote(remote_app_dir)} && . {shlex.quote(remote_repo + '/.venv/bin/activate')} && "
            f"if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi",
            timeout=1200,
        )

    def _build_demo_run_commands(
        self,
        target: AdbTarget,
        demo_name: str,
        remote_repo: str,
        model_path: str | None = None,
    ) -> list[DemoRunCommand]:
        """Build explicitly defined inference commands for one discovered demo.

        Demo and model discovery is dynamic, but inference CLIs are demo-specific:
        they require different prompts, inputs, model parts, and flags. Unknown
        demos therefore fail fast instead of using a guessed command that could
        produce a false test result.
        """
        remote_app_dir = self._demo_app_dir(demo_name, remote_repo)
        remote_model_dirs = self._demo_model_dirs(
            target,
            demo_name,
            remote_repo,
            model_path=model_path,
        )
        activate = shlex.quote(remote_repo + "/.venv/bin/activate")

        if demo_name == "gemma3":
            commands: list[DemoRunCommand] = []
            for model_dir in remote_model_dirs:
                model_path = self._find_remote_first_match(
                    target,
                    model_dir,
                    ["transformer*.vmfb*", "model*.vmfb*"],
                )
                lm_head_arg = ""
                if "transformer" in Path(model_path).name.lower():
                    lm_head = self._find_remote_first_match(target, model_dir, ["lm_head*.vmfb*"])
                    lm_head_arg = f" --lm-head {shlex.quote(lm_head)}"
                prompt = "Hello, briefly introduce yourself."
                commands.append(DemoRunCommand(
                    case_name=f"gemma3:{Path(model_dir).name}",
                    input_description=f"conversation prompt: {prompt}",
                    command=(
                        f"cd {shlex.quote(remote_app_dir)} && . {activate} && "
                        f"printf '%s\\n%s\\n' {shlex.quote(prompt)} quit | "
                        f"python src/infer.py -m {shlex.quote(model_path)}{lm_head_arg} --instruct-model"
                    ),
                ))
            return commands

        if demo_name in {"LiquidAI-LFM2.5-230M", "LiquidAI-LFM2.5"}:
            commands = []
            for model_dir in remote_model_dirs:
                try:
                    model_path = self._find_remote_first_match(
                        target,
                        model_dir,
                        ["body*.vmfb*", "transformer*.vmfb*"],
                    )
                except RuntimeError:
                    model_path = self._find_remote_first_match(target, model_dir, ["model*.vmfb*"])
                lm_head_path = ""
                try:
                    lm_head = self._find_remote_first_match(target, model_dir, ["lm_head*.vmfb*"])
                    lm_head_path = f" --lm-head {shlex.quote(lm_head)}"
                except RuntimeError:
                    lm_head_path = ""
                prompt = "Hello, briefly introduce yourself."
                commands.append(DemoRunCommand(
                    case_name=f"{demo_name}:{Path(model_dir).name}",
                    input_description=f"conversation prompt: {prompt}",
                    command=(
                        f"cd {shlex.quote(remote_app_dir)} && . {activate} && "
                        f"printf '%s\\n%s\\n' {shlex.quote(prompt)} quit | "
                        f"python src/infer.py -m {shlex.quote(model_path)}"
                        f"{lm_head_path} --instruct-model"
                    ),
                ))
            return commands

        if demo_name == "moonshine":
            wav_path = self._ensure_remote_test_wav(target, demo_name, remote_repo)
            return [DemoRunCommand(
                case_name=f"moonshine:{Path(model_dir).name}",
                input_description=f"WAV input: {wav_path}",
                command=(
                    f"cd {shlex.quote(remote_app_dir)} && . {activate} && "
                    f"python src/infer.py -m {shlex.quote(model_dir)} {shlex.quote(wav_path)}"
                ),
            ) for model_dir in remote_model_dirs]

        if demo_name == "moonshine_streaming":
            wav_path = self._ensure_remote_test_wav(target, demo_name, remote_repo)
            return [DemoRunCommand(
                case_name=f"moonshine_streaming:{Path(model_dir).name}",
                input_description=f"streaming WAV input: {wav_path}",
                command=(
                    f"cd {shlex.quote(remote_app_dir)} && . {activate} && "
                    f"python src/infer.py -m {shlex.quote(model_dir)} --wav {shlex.quote(wav_path)}"
                ),
            ) for model_dir in remote_model_dirs]

        if demo_name == "LiquidAI-LFM2-VL-450M":
            remote_model_dir = remote_model_dirs[0]
            model_path = self._find_remote_first_match(target, remote_model_dir, ["decoder_nolm*.vmfb*", "model*.vmfb*"])
            lm_head = self._find_remote_first_match(target, remote_model_dir, ["lm_head*.vmfb*"])
            vision = self._find_remote_first_match(target, remote_model_dir, ["vision_encoder*.vmfb*"])
            image_decoder_a = self._find_remote_first_match(target, remote_model_dir, ["decoder_image*_A*.vmfb*"])
            image_decoder_b = self._find_remote_first_match(target, remote_model_dir, ["decoder_image*_B*.vmfb*"])
            image_decoder_prefix = _image_decoder_prefix(image_decoder_a)
            if _image_decoder_prefix(image_decoder_b) != image_decoder_prefix:
                raise RuntimeError(
                    "Image-decoder A/B VMFB files do not share a prefix: "
                    f"{image_decoder_a}, {image_decoder_b}"
                )
            image_path = self._find_remote_first_match(target, remote_model_dir, ["*.jpg", "*.jpeg", "*.png"])
            prompt = "What is in this image?"
            return [DemoRunCommand(
                case_name=f"LiquidAI-LFM2-VL-450M:{Path(remote_model_dir).name}",
                input_description=f"image: {image_path}; prompt: {prompt}",
                command=(
                    f"cd {shlex.quote(remote_app_dir)} && . {activate} && "
                    f"python src/infer.py --model {shlex.quote(model_path)} --lm-head {shlex.quote(lm_head)} "
                    f"--vision {shlex.quote(vision)} --image-decoder {shlex.quote(image_decoder_prefix)} "
                    f"--image {shlex.quote(image_path)} --prompt {shlex.quote(prompt)}"
                ),
            )]

        if demo_name == "object_detection":
            commands = []
            for model_dir in remote_model_dirs:
                image_path = self._find_remote_first_match(target, model_dir, ["*.jpg", "*.jpeg", "*.png"])
                labels_path = self._find_remote_first_match(target, model_dir, ["labels.json"])
                for model_path in self._find_remote_matches(target, model_dir, ["*.vmfb*"]):
                    model_name = Path(model_path).name.lower()
                    variant = "yolov8"
                    if "yolo26s" in model_name:
                        variant = "yolo26s"
                    elif "yolo26n" in model_name or "yolov26n" in model_dir.lower():
                        variant = "yolo26n"
                    commands.append(DemoRunCommand(
                        case_name=f"object_detection:{Path(model_dir).name}:{Path(model_path).name}",
                        input_description=f"image: {image_path}; variant: {variant}",
                        command=(
                            f"cd {shlex.quote(remote_app_dir)} && . {activate} && "
                            f"python src/infer.py --model {shlex.quote(model_path)} "
                            f"--image {shlex.quote(image_path)} --labels {shlex.quote(labels_path)} "
                            f"--variant {shlex.quote(variant)} --device torq --device-io"
                        ),
                    ))
            return commands

        if demo_name == "pose_estimation":
            commands = []
            for model_dir in remote_model_dirs:
                image_path = self._find_remote_first_match(target, model_dir, ["*.jpg", "*.jpeg", "*.png"])
                for model_path in self._find_remote_matches(target, model_dir, ["*.vmfb*"]):
                    commands.append(DemoRunCommand(
                        case_name=f"pose_estimation:{Path(model_dir).name}:{Path(model_path).name}",
                        input_description=f"image: {image_path}",
                        command=(
                            f"cd {shlex.quote(remote_app_dir)} && . {activate} && "
                            f"python src/infer.py --model {shlex.quote(model_path)} "
                            f"--image {shlex.quote(image_path)} --device torq --device-io"
                        ),
                    ))
            return commands

        if demo_name == "face_id":
            commands = []
            for model_dir in remote_model_dirs:
                model_path = self._find_remote_first_match(
                    target,
                    model_dir,
                    ["face_detection.vmfb*"],
                )
                image_path = self._find_remote_first_match(
                    target,
                    model_dir,
                    ["face.jpg", "*.jpg", "*.jpeg", "*.png"],
                )
                commands.append(DemoRunCommand(
                    case_name=f"face_id:{Path(model_dir).name}",
                    input_description=f"image: {image_path}",
                    command=(
                        f"cd {shlex.quote(remote_app_dir)} && . {activate} && "
                        f"python src/infer.py --model {shlex.quote(model_path)} "
                        f"--image {shlex.quote(image_path)} --device torq --device-io"
                    ),
                ))
            return commands

        raise ValueError(f"No runtime command builder configured for demo '{demo_name}'")

    def _profile_model_paths(
        self,
        target: AdbTarget,
        demo_name: str,
        remote_repo: str,
        model_path: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return one profile target per generated model or VMFB variant."""
        targets: list[tuple[str, str]] = []
        for model_dir in self._demo_model_dirs(
            target,
            demo_name,
            remote_repo,
            model_path=model_path,
        ):
            model_dir_name = PurePosixPath(model_dir).name

            if demo_name == "gemma3":
                model_path = ""
                for pattern in ("transformer*.vmfb*", "model.vmfb.trim", "model*.vmfb*"):
                    matches = self._find_remote_matches(target, model_dir, [pattern])
                    if matches:
                        model_path = matches[0]
                        break
                if not model_path:
                    raise RuntimeError(f"No profileable Gemma VMFB found under {model_dir}")
                targets.append((f"{demo_name}:{model_dir_name}", model_path))
                continue

            primary_patterns = {
                "LiquidAI-LFM2.5-230M": ["body*.vmfb*", "model*.vmfb*"],
                "LiquidAI-LFM2.5": ["body*.vmfb*", "transformer*.vmfb*", "model*.vmfb*"],
                "LiquidAI-LFM2-VL-450M": ["decoder_nolm*.vmfb*", "model*.vmfb*"],
                "moonshine": ["encoder.vmfb"],
                "moonshine_streaming": ["encoder.vmfb"],
            }
            if demo_name in primary_patterns:
                model_path = self._find_remote_first_match(
                    target,
                    model_dir,
                    primary_patterns[demo_name],
                )
                targets.append((f"{demo_name}:{model_dir_name}", model_path))
                continue

            if demo_name in {"object_detection", "pose_estimation"}:
                model_paths = self._find_remote_matches(target, model_dir, ["*.vmfb*"])
                if not model_paths:
                    raise RuntimeError(f"No profileable VMFB found under {model_dir}")
                for model_path in model_paths:
                    targets.append((
                        f"{demo_name}:{model_dir_name}:{PurePosixPath(model_path).name}",
                        model_path,
                    ))
                continue

            model_paths = self._find_remote_matches(target, model_dir, ["*.vmfb*"])
            if not model_paths:
                raise RuntimeError(f"No profileable VMFB found under {model_dir}")
            for discovered_model_path in model_paths:
                targets.append((
                    f"{demo_name}:{model_dir_name}:{PurePosixPath(discovered_model_path).name}",
                    discovered_model_path,
                ))

        return targets

    def _build_demo_profile_commands(
        self,
        target: AdbTarget,
        demo_name: str,
        remote_repo: str,
        model_path: str | None = None,
    ) -> list[DemoProfileCommand]:
        profiling_entry = self._resolve_profiling_entry(demo_name)
        activate = shlex.quote(remote_repo + "/.venv/bin/activate")
        return [
            DemoProfileCommand(
                case_name=case_name,
                model_path=model_path,
                command=(
                    f"cd {shlex.quote(remote_repo)} && . {activate} && "
                    f"python {shlex.quote(profiling_entry)} "
                    f"{shlex.quote(model_path)} --repeat 3"
                ),
            )
            for case_name, model_path in self._profile_model_paths(
                target,
                demo_name,
                remote_repo,
                model_path=model_path,
            )
        ]

    def _prepare_dut_python_env(self, target: AdbTarget, remote_repo: str) -> None:
        """Prepare the DUT environment for a single demo.

        This is intentionally fail-fast: if the DUT Python environment is invalid,
        the automation stops and asks for a manual DUT repair instead of trying to
        mutate or clean up the board environment automatically.
        """
        print(f"[test_prepare] preparing shared DUT Python environment at {remote_repo}")
        try:
            print("[test_prepare] step 4/8: check DUT python3 venv support")
            self._run_remote_command(target, f"cd {shlex.quote(remote_repo)} && python3 -c \"import venv\"")
        except RuntimeError as exc:
            print("[demo_executor] DUT python3 venv support check failed")
            raise RuntimeError(
                "DUT Python venv support is missing or broken. Fix the DUT environment manually before retrying demo execution."
            ) from exc

        try:
            print("[test_prepare] step 5/8: create DUT .venv if needed")
            self._run_remote_command(target, f"cd {shlex.quote(remote_repo)} && if [ ! -d .venv ]; then python3 -m venv --system-site-packages .venv; fi")
        except RuntimeError as exc:
            print("[demo_executor] DUT .venv creation failed")
            raise RuntimeError(
                "DUT Python virtual environment creation failed. Fix the DUT Python environment manually before retrying."
            ) from exc

        print("[test_prepare] step 6/8: install base Python dependencies into DUT .venv")
        self._run_remote_command(
            target,
            f"cd {shlex.quote(remote_repo)} && . .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel",
            timeout=1200,
        )
        self._run_remote_command(
            target,
            f"cd {shlex.quote(remote_repo)} && . .venv/bin/activate && python -m pip install torq-runtime",
            timeout=1200,
        )
        self._run_remote_command(
            target,
            f"cd {shlex.quote(remote_repo)} && . .venv/bin/activate && python -m pip install -r requirements.txt",
            timeout=1200,
        )
        pip_list = self._run_remote_command(
            target,
            f"cd {shlex.quote(remote_repo)} && . .venv/bin/activate && python -m pip list",
            timeout=1200,
        )
        print(f"[test_prepare] step 7/8: DUT pip list\n{pip_list}")
        self._write_log("dut_pip_list", pip_list, "")

    def _collect_prepare_metadata(
        self,
        target: AdbTarget,
        remote_repo: str,
    ) -> dict[str, str]:
        """Read the pushed repository commit and SDK revision from the DUT."""
        print("[test_prepare] step 8/8: collect Torq Examples commit and DUT SDK version")
        commit_output = self._run_remote_command(
            target,
            f"git -c safe.directory={shlex.quote(remote_repo)} "
            f"-C {shlex.quote(remote_repo)} rev-parse HEAD",
        )
        torq_examples_commit = commit_output.strip().splitlines()[-1]
        if re.fullmatch(r"[0-9a-fA-F]{40}", torq_examples_commit) is None:
            raise RuntimeError(
                "Invalid Torq Examples commit returned by DUT: "
                f"{torq_examples_commit!r}"
            )

        build_info = self._run_remote_command(target, "cat /etc/buildinfo")
        revision_match = re.search(
            r"^\s*SYNA_SDK_REVISION\s*=\s*(\S.*?)\s*$",
            build_info,
            flags=re.MULTILINE,
        )
        if revision_match is None:
            raise RuntimeError("SYNA_SDK_REVISION not found in DUT /etc/buildinfo")
        dut_sdk_version = revision_match.group(1).strip()
        full_version_match = re.fullmatch(r"[0-9]{12}_([0-9]{12})", dut_sdk_version)
        if full_version_match is not None:
            dut_sdk_version = full_version_match.group(1)

        metadata = {
            "torq_examples_commit": torq_examples_commit,
            "dut_sdk_version": dut_sdk_version,
        }
        metadata_path = self.repo_root / "logs" / "prepare_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[test_prepare] Torq Examples Commit: {torq_examples_commit}")
        print(f"[test_prepare] DUT SDK Version: {dut_sdk_version}")
        print(f"[test_prepare] metadata saved to {metadata_path}")
        return metadata

    def _run_dut_demo_setup(
        self,
        target: AdbTarget,
        demo_name: str,
        remote_repo: str,
    ) -> dict[str, str]:
        setup_cmd = (
            f"cd {shlex.quote(remote_repo)} && . .venv/bin/activate && "
            f"python setup_demos.py {shlex.quote(demo_name)}"
        )
        setup_stdout = self._run_remote_command(
            target,
            setup_cmd,
            timeout=3600,
        )
        self._write_log(f"{demo_name}_setup", setup_stdout, "", command=setup_cmd)
        model_dirs = self._read_setup_model_paths().get(demo_name, {})
        if not model_dirs:
            raise RuntimeError(
                f"No prepared model path mapping is available for demo '{demo_name}'"
            )
        return model_dirs

    def _prepare_shared_dut_environment(self, target: AdbTarget, remote_repo: str) -> None:
        preparation_key = (target.serial, remote_repo)
        if preparation_key in self._prepared_dut_environments:
            print(
                f"[demo_executor] shared DUT preparation already completed for "
                f"{target.serial}:{remote_repo}; skipping stages 1 and 2"
            )
            return

        print(f"[demo_executor] starting shared DUT preparation for {target.serial}:{remote_repo}")
        self._push_repo_to_dut(target)
        self._prepare_dut_python_env(target, remote_repo)
        self._prepared_dut_environments.add(preparation_key)
        print(f"[demo_executor] shared DUT preparation completed for {target.serial}:{remote_repo}")

    def _require_dut_prepared(self, target: AdbTarget, remote_repo: str) -> None:
        """Verify preparation without performing any Stage 1/2 operation."""
        version_path = f"{remote_repo.rstrip('/')}/VERSION"
        venv_python = f"{remote_repo.rstrip('/')}/.venv/bin/python"
        self._run_remote_command(
            target,
            f"if [ ! -f {shlex.quote(version_path)} ]; then "
            f"echo 'DUT preparation missing repository VERSION file: {version_path}' >&2; "
            f"exit 2; fi; "
            f"if [ ! -x {shlex.quote(venv_python)} ]; then "
            f"echo 'DUT preparation missing Python environment: {venv_python}' >&2; "
            f"exit 2; fi",
        )

    def _failed_demo_result(
        self,
        demo_name: str,
        phase: str,
        exc: Exception,
        *,
        artifact_name: str | None = None,
    ) -> DemoExecutionResult:
        failed_command: str | None = None
        if isinstance(exc, CommandExecutionError):
            returncode = exc.returncode
            failed_command = exc.command
            stdout = exc.stdout
            stderr = f"{phase} failed: {exc}"
            if exc.stderr:
                stderr += f"\n{exc.stderr.rstrip()}"
        else:
            returncode = 1
            stdout = ""
            stderr = f"{phase} failed: {type(exc).__name__}: {exc}"

        if demo_name == "test_prepare":
            case_name = "test_prepare"
        elif phase == "conversion":
            case_name = f"test_convert_demo_{demo_name}"
        elif phase == "profiling":
            case_name = f"test_profile_demo_{demo_name}"
        else:
            case_name = f"test_run_demo_{demo_name}"
        print(f"[{case_name}] FAIL: {phase} failed with rc={returncode}")
        if stdout:
            print(f"[demo_executor] failure stdout:\n{stdout.rstrip()}")
        if stderr:
            print(f"[demo_executor] failure stderr:\n{stderr.rstrip()}")
        log_path = self._write_log(
            f"{artifact_name or demo_name}_{phase}_failed",
            stdout,
            stderr,
            command=failed_command,
        )
        return DemoExecutionResult(
            demo_name=demo_name,
            status="fail",
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            log_path=log_path,
        )

    def prepare_dut(self) -> DemoExecutionResult:
        """Run the suite-level repository and Python environment preparation."""
        try:
            print("\n========== test_prepare ==========")
            print("[test_prepare] step 1/8: discover demos from setup_demos.py DEMOS")
            demo_names = repo_demo_names()
            if not demo_names:
                raise RuntimeError("No demos found in setup_demos.py DEMOS")
            remote_repo = self.remote_root
            print(
                "[test_prepare] deriving model paths from setup repository metadata "
                "without importing demo setup code or downloading models"
            )
            model_paths = self._discover_setup_model_paths(demo_names, remote_repo)
            print(f"[test_prepare] discovered {len(demo_names)} demo(s):")
            for index, demo_name in enumerate(demo_names, start=1):
                formatted_paths = ", ".join(
                    model_path
                    for model_path in model_paths[demo_name].values()
                )
                print(
                    f"[test_prepare]   {index}. {demo_name}   "
                    f"model path: {formatted_paths}"
                )
            print("[test_prepare] step 2/8: acquire DUT target")
            target = self._build_target()
            self._prepare_shared_dut_environment(target, remote_repo)
            metadata = self._collect_prepare_metadata(target, remote_repo)
            model_path_lines = [
                f"{demo_name}: "
                + ", ".join(
                    model_path
                    for model_path in model_paths[demo_name].values()
                )
                for demo_name in demo_names
            ]
            stdout = "".join([
                f"DUT preparation completed\n"
                f"demos={','.join(demo_names)}\n",
                "model_paths=\n",
                "\n".join(model_path_lines),
                "\n",
                f"target={target.serial}\n",
                f"remote_repo={remote_repo}\n",
                f"torq_examples_commit={metadata['torq_examples_commit']}\n",
                f"dut_sdk_version={metadata['dut_sdk_version']}\n",
                f"pip_list_log={self.repo_root / 'logs' / 'dut_pip_list.log'}",
            ])
            log_path = self._write_log("test_prepare", stdout, "")
            print("[test_prepare] PASS")
            return DemoExecutionResult(
                demo_name="test_prepare",
                status="pass",
                returncode=0,
                stdout=stdout,
                stderr="",
                log_path=log_path,
                model_dirs={
                    f"{demo_name}:{model_name}": model_path
                    for demo_name, paths in model_paths.items()
                    for model_name, model_path in paths.items()
                },
                metadata=metadata,
            )
        except Exception as exc:
            return self._failed_demo_result("test_prepare", "preparation", exc)

    def convert_demo_on_dut(
        self,
        demo_name: str,
        model_path: str | None = None,
    ) -> DemoExecutionResult:
        case_name = f"test_convert_demo_{demo_name}"
        artifact_name = self._artifact_name(demo_name, model_path)
        try:
            print(f"\n========== {case_name} ==========")
            print(f"[{case_name}] step 1/5: acquire DUT target")
            target = self._build_target()
            remote_repo = self.remote_root
            print(f"[{case_name}] step 2/5: verify shared DUT preparation")
            self._require_dut_prepared(target, remote_repo)
            print(f"[{case_name}] step 3/5: install per-demo requirements")
            self._install_demo_requirements_on_dut(target, demo_name, remote_repo)
            print(f"[{case_name}] step 4/5: download/convert model assets")
            conversion_key = (target.serial, remote_repo, demo_name)
            setup_error: CommandExecutionError | None = None
            model_dirs = self._read_setup_model_paths().get(demo_name, {})
            if conversion_key in self._setup_attempted_demos:
                print(f"[{case_name}] reuse setup attempt from another model case")
            else:
                self._setup_attempted_demos.add(conversion_key)
                try:
                    model_dirs = self._run_dut_demo_setup(target, demo_name, remote_repo)
                except CommandExecutionError as exc:
                    setup_error = exc
                    self._write_log(
                        f"{demo_name}_setup",
                        exc.stdout,
                        exc.stderr,
                        command=exc.command,
                    )
                    print(
                        f"[{case_name}] setup command returned rc={exc.returncode}; "
                        "validating the requested target model before deciding the result"
                    )
            print(f"[{case_name}] step 5/5: validate generated model assets")
            try:
                validated_dirs = self._demo_model_dirs(
                    target,
                    demo_name,
                    remote_repo,
                    model_path=model_path,
                )
            except Exception as validation_exc:
                if setup_error is not None:
                    raise CommandExecutionError(
                        f"Setup command failed and target model validation failed: "
                        f"{validation_exc}",
                        returncode=setup_error.returncode,
                        stdout=setup_error.stdout,
                        stderr=setup_error.stderr,
                        command=setup_error.command,
                    ) from validation_exc
                raise

            model_versions = self._record_setup_model_versions(
                target,
                demo_name,
                model_dirs,
            )

            result_details: dict[str, object] = {
                "model_dirs": model_dirs,
                "model_versions": model_versions,
                "validated_model_dirs": validated_dirs,
                "setup_returncode": setup_error.returncode if setup_error else 0,
            }
            if setup_error is not None:
                result_details["setup_warning"] = str(setup_error)
            stdout = json.dumps(result_details, indent=2, ensure_ascii=False)
            log_path = self._write_log(f"{artifact_name}_conversion", stdout, "")
            if setup_error is not None:
                print(
                    f"[{case_name}] target model is valid; ignoring setup rc="
                    f"{setup_error.returncode} for this model case"
                )
            print(f"[{case_name}] PASS")
            return DemoExecutionResult(
                demo_name=demo_name,
                status="pass",
                returncode=0,
                stdout=stdout,
                stderr="",
                log_path=log_path,
                model_dirs=model_dirs,
                model_versions=model_versions,
            )
        except Exception as exc:
            return self._failed_demo_result(
                demo_name,
                "conversion",
                exc,
                artifact_name=artifact_name,
            )

    def run_demo(
        self,
        demo_name: str,
        model_path: str | None = None,
    ) -> DemoExecutionResult:
        try:
            return self._run_demo_or_raise(demo_name, model_path=model_path)
        except Exception as exc:
            return self._failed_demo_result(
                demo_name,
                "execution",
                exc,
                artifact_name=self._artifact_name(demo_name, model_path),
            )

    def profile_demo(
        self,
        demo_name: str,
        model_path: str | None = None,
    ) -> DemoExecutionResult:
        try:
            return self._profile_demo_or_raise(demo_name, model_path=model_path)
        except Exception as exc:
            return self._failed_demo_result(
                demo_name,
                "profiling",
                exc,
                artifact_name=self._artifact_name(demo_name, model_path),
            )

    def _profile_demo_or_raise(
        self,
        demo_name: str,
        model_path: str | None = None,
    ) -> DemoExecutionResult:
        case_name = f"test_profile_demo_{demo_name}"
        artifact_name = self._artifact_name(demo_name, model_path)
        print(f"\n========== {case_name} ==========")
        print(f"[{case_name}] step 1/6: acquire DUT target")
        target = self._build_target()
        remote_repo = self.remote_root
        print(f"[{case_name}] step 2/6: verify shared DUT preparation")
        self._require_dut_prepared(target, remote_repo)
        print(f"[{case_name}] step 3/6: install per-demo requirements")
        self._install_demo_requirements_on_dut(target, demo_name, remote_repo)
        print(f"[{case_name}] step 4/6: verify converted model assets")
        self._demo_model_dirs(
            target,
            demo_name,
            remote_repo,
            model_path=model_path,
        )
        print(f"[{case_name}] converted model assets are ready")

        print(f"[{case_name}] step 5/6: build per-model profiling commands")
        profile_commands = self._build_demo_profile_commands(
            target,
            demo_name,
            remote_repo,
            model_path=model_path,
        )
        if not profile_commands:
            raise RuntimeError(f"No profileable models discovered for demo '{demo_name}'")

        print(f"[{case_name}] step 6/6: profile {len(profile_commands)} model(s) over ADB")
        outputs: list[str] = []
        records: list[dict[str, object]] = []
        for index, profile_command in enumerate(profile_commands, start=1):
            print(
                f"[{case_name}] profile case {index}/{len(profile_commands)}: "
                f"{profile_command.case_name}"
            )
            print(f"[{case_name}] model: {profile_command.model_path}")
            print(f"[{case_name}] DUT command: {profile_command.command}")
            log_lines = [
                f"===== {profile_command.case_name} =====",
                f"TARGET: {target.serial}",
                f"MODEL: {profile_command.model_path}",
                f"COMMAND: {profile_command.command}",
                "----- OUTPUT BEGIN -----",
            ]
            try:
                output = self._run_remote_command_combined(
                    target,
                    profile_command.command,
                    timeout=3600,
                )
            except CommandExecutionError as exc:
                log_lines.append(exc.stdout.rstrip())
                log_lines.extend([
                    "----- OUTPUT END -----",
                    f"RETURN CODE: {exc.returncode}",
                    "RESULT: FAIL",
                ])
                outputs.append("\n".join(log_lines))
                self._write_log(f"{artifact_name}_profile", "\n\n".join(outputs), "")
                self._write_profile_results(artifact_name, records)
                raise

            metrics = parse_profile_metrics(output)
            missing_metrics = [
                label for label in PROFILE_METRIC_LABELS if label not in metrics
            ]
            if missing_metrics:
                log_lines.append(output.rstrip())
                log_lines.extend([
                    "----- OUTPUT END -----",
                    "RETURN CODE: 0",
                    "RESULT: FAIL",
                    "MISSING METRICS: " + ", ".join(missing_metrics),
                ])
                outputs.append("\n".join(log_lines))
                self._write_log(
                    f"{artifact_name}_profile",
                    "\n\n".join(outputs),
                    "",
                )
                self._write_profile_results(artifact_name, records)
                raise RuntimeError(
                    f"Profiling output for {profile_command.case_name} is missing: "
                    f"{', '.join(missing_metrics)}"
                )

            record: dict[str, object] = {
                "demo_name": demo_name,
                "case_name": profile_command.case_name,
                "target": target.serial,
                "model_path": profile_command.model_path,
                "command": profile_command.command,
                "metrics": metrics,
            }
            records.append(record)
            log_lines.append(output.rstrip())
            log_lines.extend([
                "----- OUTPUT END -----",
                "RETURN CODE: 0",
                "RESULT: PASS",
                "PARSED METRICS: " + json.dumps(metrics, ensure_ascii=False),
            ])
            outputs.append("\n".join(log_lines))
            self._write_log(f"{artifact_name}_profile", "\n\n".join(outputs), "")
            self._write_profile_results(artifact_name, records)
            print(f"[{case_name}] parsed metrics for {profile_command.case_name}:")
            for label in PROFILE_METRIC_LABELS:
                print(f"[{case_name}]   {label}: {metrics[label]}")

        stdout = "\n\n".join(outputs)
        log_path = self._write_log(f"{artifact_name}_profile", stdout, "")
        profile_json_path = self._write_profile_results(artifact_name, records)
        print(f"[{case_name}] profile results: {profile_json_path}")
        print(f"[{case_name}] PASS")
        return DemoExecutionResult(
            demo_name=demo_name,
            status="pass",
            returncode=0,
            stdout=stdout,
            stderr="",
            log_path=log_path,
            profile_records=records,
        )

    def _run_demo_or_raise(
        self,
        demo_name: str,
        model_path: str | None = None,
    ) -> DemoExecutionResult:
        case_name = f"test_run_demo_{demo_name}"
        artifact_name = self._artifact_name(demo_name, model_path)
        print(f"\n========== {case_name} ==========")
        print(f"[{case_name}] step 1/6: acquire DUT target")
        target = self._build_target()
        self._resolve_demo_entry(demo_name)
        remote_repo = self.remote_root
        print(f"[{case_name}] step 2/6: verify shared DUT preparation")
        self._require_dut_prepared(target, remote_repo)
        print(f"[{case_name}] step 3/6: install per-demo requirements")
        self._install_demo_requirements_on_dut(target, demo_name, remote_repo)
        print(f"[{case_name}] step 4/6: verify converted model assets")
        self._demo_model_dirs(
            target,
            demo_name,
            remote_repo,
            model_path=model_path,
        )
        print(f"[{case_name}] converted model assets are ready")

        print(f"[{case_name}] step 5/6: build model-specific run commands")
        run_commands = self._build_demo_run_commands(
            target,
            demo_name,
            remote_repo,
            model_path=model_path,
        )
        if not run_commands:
            raise RuntimeError(f"No runnable model commands discovered for demo '{demo_name}'")

        print(f"[{case_name}] step 6/6: execute {len(run_commands)} model command(s)")
        outputs: list[str] = []
        for index, run_command in enumerate(run_commands, start=1):
            print(
                f"[{case_name}] model case {index}/{len(run_commands)}: "
                f"{run_command.case_name}"
            )
            if run_command.input_description:
                print(f"[{case_name}] test input: {run_command.input_description}")
            print(f"[{case_name}] DUT command: {run_command.command}")
            case_log_lines = [
                f"===== {run_command.case_name} =====",
                f"TARGET: {target.serial}",
            ]
            if run_command.input_description:
                case_log_lines.append(f"INPUT: {run_command.input_description}")
            case_log_lines.extend([
                f"COMMAND: {run_command.command}",
                "----- OUTPUT BEGIN -----",
            ])
            try:
                case_output = self._run_remote_command_combined(
                    target,
                    run_command.command,
                    timeout=3600,
                )
            except CommandExecutionError as exc:
                case_log_lines.append(exc.stdout.rstrip())
                case_log_lines.extend([
                    "----- OUTPUT END -----",
                    f"RETURN CODE: {exc.returncode}",
                    "RESULT: FAIL",
                ])
                outputs.append("\n".join(case_log_lines))
                self._write_log(f"{artifact_name}_run", "\n\n".join(outputs), "")
                raise

            case_log_lines.append(case_output.rstrip())
            case_log_lines.extend([
                "----- OUTPUT END -----",
                "RETURN CODE: 0",
                "RESULT: PASS",
            ])
            outputs.append("\n".join(case_log_lines))
            self._write_log(f"{artifact_name}_run", "\n\n".join(outputs), "")
            print(f"[{case_name}] model case passed: {run_command.case_name}")

        stdout = "\n".join(outputs)
        log_path = self._write_log(f"{artifact_name}_run", stdout, "")
        print(f"[{case_name}] PASS")
        return DemoExecutionResult(
            demo_name=demo_name,
            status="pass" if "ERROR" not in stdout.upper() else "fail",
            returncode=0 if "ERROR" not in stdout.upper() else 1,
            stdout=stdout,
            stderr="",
            log_path=log_path,
        )


class DemoExecutor:
    """Public execution entry point used by the host-side automation flow."""

    def __init__(self, repo_root: Path | None = None, log_dir: Path | None = None):
        self.repo_root = repo_root or REPO_ROOT
        self.log_dir = log_dir or (self.repo_root / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.runner = HostDemoRunner(repo_root=self.repo_root)

    def __enter__(self) -> DemoExecutor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release the suite DUT if it has been acquired."""
        self.runner.close()

    def run_demo(
        self,
        demo_name: str,
        model_path: str | None = None,
    ) -> DemoExecutionResult:
        return self.runner.run_demo(demo_name, model_path=model_path)

    def profile_demo(
        self,
        demo_name: str,
        model_path: str | None = None,
    ) -> DemoExecutionResult:
        return self.runner.profile_demo(demo_name, model_path=model_path)

    def prepare_dut(self) -> DemoExecutionResult:
        return self.runner.prepare_dut()

    def convert_demo_on_dut(
        self,
        demo_name: str,
        model_path: str | None = None,
    ) -> DemoExecutionResult:
        return self.runner.convert_demo_on_dut(demo_name, model_path=model_path)
