"""DUT deploy + inference runner (host-driven reconstruction, step 2 matrix).

Deploys the host-prepared stack (step 1, host_converter.py) onto the DUT
board and runs the per-demo inference matrix:

    deploy (once per run):
      1. connectivity       adb reachability + DUT arch check
      2. prepare workspace  wipe DUT root, push examples source, rebuild venv
      3. push models        all selected demos' model dirs, host -> DUT
      4. ko swap            rmmod syna_npu -> push .ko -> insmod -> lsmod verify
      5. wheel install      push aarch64 torq-runtime wheel -> pip --force-reinstall
    infer (per demo):
      6. demo deps + inputs + per-demo inference commands
    profile (per demo):
      7. repo-root profile.py --repeat 3 over the demo's VMFBs (7 metrics)

The DUT workspace is wiped and rebuilt on every run: stale files on the
board (old models, old venv state, old sources) would pollute the test
results. Debug escape hatch: TORQ_DUT_SKIP_PREPARE=1 keeps the existing
workspace (local fast iteration only).

Inference commands are ported 1:1 from the legacy DUT-driven builder
(demo_executor._build_demo_run_commands), with file discovery moved from
remote globs to host-side resolution against the pushed model dirs.

Board control:
    Boards are acquired through torq.utils.boards (shared control-file
    allocation with cross-process locking, same as the legacy executor) -
    never picked ad hoc. Control file defaults to
    <this file's dir>/board_control.json; override with TORQ_BOARDS_CONTROL.

Environment variables:
    TORQ_DUT_SERIAL          pin this specific board through the pool lock
                             (standalone fallback without torq.utils.boards)
    TORQ_DUT_REMOTE_ROOT     repo root on the DUT (default: /home/torq-examples)
    TORQ_DUT_RUNTIME_WHEEL   host path to the aarch64 torq-runtime wheel (required)
    TORQ_DUT_KO              host path to syna_npu.ko (required)
    TORQ_DUT_SKIP_PREPARE    "1" = keep existing DUT workspace (debug only)
    TORQ_DUT_SKIP_DEPLOY     "1" = skip all deploy phases (connectivity only),
                             run the inference matrix on the current DUT state
                             (debug iteration on infer commands)
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

# Board control: borrow torq.utils.boards (same as the legacy demo_executor)
# so boards are allocated/locked through the shared control file, not picked
# ad hoc. The bundled suite `torq/` package is importable when this file
# lives in the suite directory; otherwise fall back to an installed torq
# package. create_boards_control stays None when neither is available
# (standalone debug can then pin a serial via TORQ_DUT_SERIAL).
create_boards_control = None
try:
    from torq.utils.boards import create_boards_control
except ImportError:
    _bundled_parent = str(Path(__file__).resolve().parent)
    if _bundled_parent not in sys.path:
        sys.path.insert(0, _bundled_parent)
    try:
        from torq.utils.boards import create_boards_control
    except ImportError:
        create_boards_control = None

# ---------------------------------------------------------------------------
# Per-demo specification table.
#
# app_dir / model_dirs mirror torq-examples' setup_demos.py layout
# (model_dirs are relative to <checkout>/models). Keep in sync with it.
# builder=None means "known debt": the demo's infer.py CLI is not covered by
# a command builder yet (same gap as the legacy suite for these demos);
# models are still pushed and reported as skipped.
# ---------------------------------------------------------------------------

DEMO_SPECS: dict[str, dict] = {
    "gemma3": {
        "app_dir": "gemma3",
        "model_dirs": ["Synaptics/gemma-3-270-GPTQ-int4-it-torq"],
        "builder": "llm_chat",
        "expect": "Agent:",
    },
    "functiongemma": {
        "app_dir": "functiongemma",
        "model_dirs": ["Synaptics/functiongemma-270m-it-torq"],
        # same model structure as gemma3, but its infer.py is registered with
        # with_instruct=False: no --instruct-model flag; --no-refresh skips
        # the model freshness re-check on startup.
        "builder": "llm_chat_plain",
        "extra_flags": "--no-refresh",
        "expect": "Agent:",
    },
    "LiquidAI-LFM2.5": {
        "app_dir": "LiquidAI/LiquidAI-LFM2.5",
        "model_dirs": ["Synaptics/LiquidAI-LFM2.5-230M"],
        "builder": "llm_chat_body",
        "expect": None,
    },
    "moonshine": {
        "app_dir": "moonshine",
        "model_dirs": ["Synaptics/moonshine-tiny-bf16-torq"],
        "builder": "asr",
        "need_wav": True,
        "expect": None,
    },
    "moonshine_streaming": {
        "app_dir": "moonshine_streaming",
        "model_dirs": ["Synaptics/moonshine-streaming-tiny-torq"],
        "builder": "asr_streaming",
        "need_wav": True,
        "expect": None,
    },
    "LiquidAI-LFM2-VL-450M": {
        "app_dir": "LiquidAI/LiquidAI-LFM2-VL-450M",
        "model_dirs": ["Synaptics/LiquidAI-LFM2-VL-450M"],
        "builder": "vl",
        "expect": None,
    },
    "object_detection": {
        "app_dir": "object_detection",
        "model_dirs": [
            "Synaptics/yolov8-od-nano-320-int8-torq",
            "Synaptics/yolov26n_od",
        ],
        "builder": "od",
        "expect": None,
    },
    "pose_estimation": {
        "app_dir": "pose_estimation",
        "model_dirs": ["Synaptics/yolov8-pose-nano-320-int8-torq"],
        "builder": "pose",
        "expect": None,
    },
    "face_id": {
        "app_dir": "Face_ID",
        "model_dirs": ["Synaptics/face-id-torq"],
        "builder": "face",
        "expect": None,
    },
    "qwen2_5": {
        "app_dir": "qwen2_5",
        "model_dirs": ["Synaptics/Qwen2.5-0.5B-Instruct-GPTQ-int4-torq"],
        # instruct model; infer.py now reuses add_llm_inference_args (same CLI as gemma3)
        "builder": "llm_chat",
        "expect": "Agent:",
    },
    "qwen3": {
        "app_dir": "qwen3",
        "model_dirs": ["Synaptics/Qwen3-0.6B-GPTQ-int4-torq"],
        # base model (no -Instruct); same CLI as gemma3 minus --instruct-model.
        # --no-thinking: default thinking mode spends the whole context on
        # reasoning for a short prompt and emits "[No answer: context was full]"
        # instead of answering; --no-thinking makes it answer directly.
        "builder": "llm_chat_plain",
        "extra_flags": "--no-thinking",
        "expect": "Agent:",
    },
    "EmbeddingGemma-2": {
        "app_dir": "EmbeddingGemma-2",
        "model_dirs": ["Synaptics/Google-EmbeddingGemma-2"],
        # Known platform limitation on SL2619: the full text session keeps
        # text_body_s128 + text_body_s512 resident and text_body_s512 fails to
        # map (xram_write out of bounds) at warmup. All 7 vmfbs profile OK in
        # isolation, so the model runs; only the combined infer session is
        # blocked. The "embed" builder is retained -- flip builder back to
        # "embed" once the runner seq_lens/XRAM budget is fixed for this board.
        "builder": None,
        "debt": "full text session blocked on SL2619 (text_body_s512 XRAM OOB at warmup); all 7 vmfbs profile OK in isolation",
        "expect": None,
    },
}


class DutDeployError(RuntimeError):
    pass


@dataclass
class PhaseResult:
    phase: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    commands: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _file_size(path: str) -> str:
    try:
        n = float(Path(path).stat().st_size)
    except OSError:
        return "?"
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return "?"


def _walk_files(directory: Path, maxdepth: int = 4):
    for root, dirs, files in os.walk(directory):
        rel_parts = len(Path(root).relative_to(directory).parts)
        if rel_parts >= maxdepth:
            dirs.clear()
            continue
        for name in sorted(files):
            yield Path(root) / name


def _first_match(directory: Path, patterns: list[str], maxdepth: int = 4) -> Path | None:
    """Recursive first-match by file name (fnmatch); mirrors the legacy remote
    finder (_find_remote_first_match, maxdepth=4)."""
    for path in _walk_files(directory, maxdepth):
        if any(fnmatch.fnmatchcase(path.name, pat) for pat in patterns):
            return path
    return None


def _all_matches(directory: Path, pattern: str, maxdepth: int = 4) -> list[Path]:
    return [
        path for path in _walk_files(directory, maxdepth)
        if fnmatch.fnmatchcase(path.name, pattern)
    ]


def _image_decoder_prefix(vmfb_path: str) -> str:
    """Return the shared prefix from an A/B image-decoder VMFB path."""
    match = re.fullmatch(
        r"(?P<prefix>.+_)[AB](?:_[^/]*)?\.vmfb(?:\..*)?",
        vmfb_path,
    )
    if match is None:
        raise DutDeployError(
            "Could not derive image-decoder prefix from VMFB path: " + vmfb_path
        )
    return match.group("prefix")


PROFILE_METRIC_LABELS = (
    "Avg CPU usage",
    "Avg DRAM footprint",
    "Avg NPU usage",
    "Avg inference time",
    "Avg memory usage",
    "Peak DRAM footprint",
    "Peak memory usage",
)


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


def _profile_vmfb_files(directory: Path, maxdepth: int = 4) -> list[Path]:
    """Strict VMFB modules: names ending in .vmfb or .vmfb.trim only
    (excludes HF download sidecars .vmfb.lock / .vmfb.metadata)."""
    return sorted(
        path for path in _walk_files(directory, maxdepth)
        if path.name.endswith((".vmfb", ".vmfb.trim"))
    )


class DutDeployRunner:
    def __init__(self, examples_root: Path, demos: list[str] | None = None):
        self.examples_root = Path(examples_root).resolve()
        self.demos = demos or sorted(DEMO_SPECS)
        unknown = [d for d in self.demos if d not in DEMO_SPECS]
        if unknown:
            raise DutDeployError(f"unknown demo(s): {unknown}; known: {sorted(DEMO_SPECS)}")
        self.remote_root = os.getenv("TORQ_DUT_REMOTE_ROOT", "/home/torq-examples")
        self.wheel = os.getenv("TORQ_DUT_RUNTIME_WHEEL", "").strip()
        self.ko = os.getenv("TORQ_DUT_KO", "").strip()
        self.log_dir = self.examples_root / "logs"
        self.board_control_file = os.getenv(
            "TORQ_BOARDS_CONTROL",
            str(Path(__file__).resolve().parent / "board_control.json"),
        )
        self.serial = ""
        self._board_context = None
        self._wav_cache: dict[str, Path] = {}
        self._registry_cache: dict | None = None
        self._runtime_version_cache: str | None = None

    # ------------------------------------------------- spec resolution
    def _registry(self) -> dict:
        """The suite's demo_registry, derived from setup_demos.py's import
        structure - the authoritative source for app dirs and model repos.
        Empty dict in standalone mode (module not importable)."""
        if self._registry_cache is not None:
            return self._registry_cache
        try:
            from TorqExamplesTest import demo_registry
            self._registry_cache = demo_registry.demo_registry()
        except Exception:
            self._registry_cache = {}
        return self._registry_cache

    def _spec_for(self, demo: str) -> dict:
        """DEMO_SPECS entry, upgraded with registry-derived app_dir /
        model_dirs when available (table stays as standalone fallback)."""
        spec = dict(DEMO_SPECS[demo])
        reg = self._registry()
        if reg:
            rs = reg.get(demo)
            if rs is not None:
                if rs.app_dir:
                    spec["app_dir"] = rs.app_dir
                repos = [repo_id for _name, repo_id in rs.setup_model_repositories]
                if repos:
                    spec["model_dirs"] = repos
        return spec

    # ------------------------------------------------------------------ adb
    def _adb(self, *args: str, check: bool = True, timeout: int = 3600) -> subprocess.CompletedProcess:
        cmd = ["adb", "-s", self.serial, *[str(a) for a in args]]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise DutDeployError("adb CLI not found on host") from exc
        if check and proc.returncode != 0:
            raise DutDeployError(
                f"adb command failed (rc={proc.returncode}): {' '.join(map(shlex.quote, map(str, cmd)))}\n"
                f"stderr:\n{proc.stderr}"
            )
        return proc

    def _shell(self, script: str, check: bool = True, timeout: int = 3600) -> PhaseResult:
        proc = self._adb("shell", script, check=False, timeout=timeout)
        result = PhaseResult("shell", proc.returncode, proc.stdout, proc.stderr)
        # Record the exact host-side command so the phase log is reproducible
        # by someone who has never seen this repo.
        result.commands = [f"adb -s {self.serial} shell {script}"]
        return result

    def _runtime_version(self) -> str:
        """torq-runtime package version installed in the DUT venv (cached).

        profile.py runs after wheel-install, so torq-runtime is present by the
        time this is called. Queried once and reused across all profile records.
        """
        if self._runtime_version_cache is not None:
            return self._runtime_version_cache
        version = ""
        r = self._shell(
            f"cd {shlex.quote(self.remote_root)} && "
            f". {shlex.quote(self.remote_root + '/.venv/bin/activate')} && "
            f"pip show torq-runtime 2>/dev/null | grep -E '^Version:' || true",
            check=False,
        )
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "Version:":
                version = parts[1].strip()
                break
        self._runtime_version_cache = version
        return version

    def acquire_board(self) -> None:
        """Acquire and lock one DUT board via torq.utils.boards.

        Mirrors the legacy demo_executor._build_target: the board comes from
        the shared control-file allocation (cross-process locking), so this
        run cannot collide with another suite/CI using the same farm.
        TORQ_DUT_SERIAL pins a specific board (still through the lock);
        without it the first free board is taken.
        """
        if self.serial:
            return
        pin = os.getenv("TORQ_DUT_SERIAL", "").strip()
        if create_boards_control is None:
            if pin:
                print(
                    "[dut_deploy] torq.utils.boards unavailable - using direct "
                    f"serial {pin} (standalone debug only; no pool lock)"
                )
                self.serial = pin
                return
            raise DutDeployError(
                "torq.utils.boards is unavailable and TORQ_DUT_SERIAL is not set. "
                "Run from the suite layout (bundled torq package) or set "
                "TORQ_DUT_SERIAL for standalone debugging."
            )
        print(
            f"[dut_deploy] acquiring DUT board via torq.utils.boards "
            f"(control file: {self.board_control_file}, pin: {pin or 'any free board'})"
        )
        control = create_boards_control(self.board_control_file, backend="adb")
        ctx = control.board(address=pin) if pin else control.board()
        _board_user, serial, _private_key = ctx.__enter__()
        self._board_context = ctx
        self.serial = serial
        print(f"[dut_deploy] acquired and locked DUT board: serial={serial}")

    def release_board(self) -> None:
        """Release the board lock held by this runner."""
        if self._board_context is not None:
            try:
                self._board_context.__exit__(None, None, None)
                print(f"[dut_deploy] released DUT board: serial={self.serial}")
            except Exception as exc:  # never hide the real result behind release noise
                print(f"[dut_deploy] WARNING: board release failed: {exc}")
            finally:
                self._board_context = None

    # --------------------------------------------------------------- phases
    def phase_connectivity(self) -> PhaseResult:
        result = self._shell("echo ok && uname -m && whoami")
        result.stdout = (
            f"[dut_deploy] DUT check:\n"
            f"    serial: {self.serial}\n"
            f"    remote root: {self.remote_root}\n"
            f"    demos   : {', '.join(self.demos)}\n" + result.stdout
        )
        if result.passed and "aarch64" not in result.stdout:
            result.returncode = 1
            result.stderr += "\n[dut_deploy] DUT is not aarch64 - check TORQ_DUT_SERIAL"
        return result

    def _workspace_present(self) -> bool:
        probe = self._shell(
            f"test -f {shlex.quote(self.remote_root + '/setup_demos.py')} && "
            f"test -x {shlex.quote(self.remote_root + '/.venv/bin/python')} && "
            "echo WORKSPACE_PRESENT"
        )
        return probe.passed and "WORKSPACE_PRESENT" in probe.stdout

    def _push_source_and_bootstrap_venv(self, info: str) -> PhaseResult:
        tar_cmd = (
            f"tar -cf - -C {shlex.quote(str(self.examples_root))} "
            f"--exclude=models --exclude=.git --exclude=logs ."
        )
        tar = subprocess.run(
            [
                "tar", "-cf", "-", "-C", str(self.examples_root),
                "--exclude=models", "--exclude=.git", "--exclude=logs", ".",
            ],
            capture_output=True,
        )
        if tar.returncode != 0:
            result = PhaseResult(
                "prepare-workspace", tar.returncode, info, tar.stderr.decode(errors="replace")
            )
            result.commands = [tar_cmd]
            return result
        stream_cmd = f"adb -s {self.serial} shell tar -xf - -C {shlex.quote(self.remote_root)}"
        stream = subprocess.run(
            [
                "adb", "-s", self.serial, "shell",
                f"tar -xf - -C {shlex.quote(self.remote_root)}",
            ],
            input=tar.stdout,
            capture_output=True,
            timeout=3600,
        )
        if stream.returncode != 0:
            result = PhaseResult(
                "prepare-workspace", stream.returncode, info,
                stream.stderr.decode(errors="replace"),
            )
            result.commands = [tar_cmd, stream_cmd]
            return result
        script = (
            f"cd {shlex.quote(self.remote_root)} && "
            f"python3 -m venv --system-site-packages .venv && "
            f". .venv/bin/activate && "
            f"pip install --no-cache-dir -r requirements.txt"
        )
        result = self._shell(script, timeout=3600)
        result.commands = [tar_cmd, stream_cmd, *result.commands]
        result.stdout = info + result.stdout
        if result.passed:
            result.stdout += self._collect_metadata()
            result.stdout += self._collect_model_versions()
        return result

    def _collect_model_versions(self) -> str:
        """Read each model dir's .manifest.json (host side, written by
        setup_demos.py during host conversion) into
        logs/setup_model_versions.json for the report VERSION/revision
        columns. Demos without a manifest are reported as HEAD (untracked),
        matching the legacy convention."""
        lines = ["[dut_deploy] model version collection:"]
        data: dict[str, dict] = {}
        for demo in self.demos:
            spec = self._spec_for(demo)
            for model_rel in spec["model_dirs"]:
                model_dir = self.examples_root / "models" / model_rel
                record = {
                    "repo_id": model_rel,
                    "version": "HEAD (untracked)",
                    "revision": "",
                }
                manifest = model_dir / ".manifest.json"
                if manifest.exists():
                    try:
                        parsed = json.loads(manifest.read_text(encoding="utf-8"))
                        record["version"] = (
                            parsed.get("version") or parsed.get("resolved_version")
                            or record["version"]
                        )
                        record["revision"] = (
                            parsed.get("revision") or parsed.get("commit") or ""
                        )
                    except (OSError, json.JSONDecodeError):
                        lines.append(f"    {model_rel}: (manifest unreadable)")
                data.setdefault(demo, {})[Path(model_rel).name] = record
                lines.append(
                    f"    {model_rel}: {record['version']} {record['revision'][:12]}"
                )
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "setup_model_versions.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            lines.append("    -> logs/setup_model_versions.json")
        except OSError as exc:
            lines.append(f"    (write failed: {exc})")
        return "\n".join(lines) + "\n"

    def _collect_metadata(self) -> str:
        """Collect the report header anchors the legacy prepare stage used
        to provide (examples commit, DUT SDK revision, DUT pip list)."""
        notes = ["[dut_deploy] metadata collection:"]
        # examples commit: read on the HOST checkout (the DUT copy has no .git)
        commit = ""
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.examples_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                commit = proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            proc = None
        if commit:
            notes.append(f"    torq examples commit : {commit}")
        else:
            notes.append("    torq examples commit : (unavailable - not a git checkout or git missing)")
        # DUT SDK revision
        sdk = ""
        buildinfo = self._shell("cat /etc/buildinfo 2>/dev/null || true")
        for line in buildinfo.stdout.splitlines():
            m = re.match(r"^\s*SYNA_SDK_REVISION\s*=\s*(\S.*?)\s*$", line)
            if m:
                raw = m.group(1)
                full = re.fullmatch(r"[0-9]{12}_([0-9]{12})", raw)
                sdk = full.group(1) if full else raw
                break
        notes.append(f"    DUT SDK revision     : {sdk or '(not found in /etc/buildinfo)'}")
        # DUT pip list (runtime version anchor for the report header)
        pip_list = self._shell(
            f"cd {shlex.quote(self.remote_root)} && "
            f". .venv/bin/activate && pip list"
        )
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            pip_cmd = pip_list.commands[0] if pip_list.commands else "(pip list)"
            (self.log_dir / "dut_pip_list.log").write_text(
                f"returncode: {pip_list.returncode}\n"
                f"$ {pip_cmd}\n\n"
                f"{pip_list.stdout}", encoding="utf-8"
            )
            notes.append(f"    DUT pip list         : logs/dut_pip_list.log ({len(pip_list.stdout.splitlines())} lines)")
        except OSError:
            pass
        # prepare_metadata.json (consumed by the report header + JSON export)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "prepare_metadata.json").write_text(
                json.dumps(
                    {"torq_examples_commit": commit, "dut_sdk_version": sdk},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        return "\n" + "\n".join(notes) + "\n"

    def phase_prepare_workspace(self) -> PhaseResult:
        """Wipe and rebuild the DUT workspace (fresh state, no pollution)."""
        if os.getenv("TORQ_DUT_SKIP_PREPARE", "") == "1":
            if self._workspace_present():
                info = (
                    f"[dut_deploy] TORQ_DUT_SKIP_PREPARE=1 - keeping existing "
                    f"workspace at {self.remote_root} (debug mode)\n"
                )
                print(info)
                return PhaseResult("prepare-workspace", 0, info)
            raise DutDeployError(
                "TORQ_DUT_SKIP_PREPARE=1 but the DUT workspace is incomplete; "
                "rerun without it to rebuild"
            )

        info = (
            f"[dut_deploy] rebuild DUT workspace (stale files wiped):\n"
            f"    wipe   : rm -rf {self.remote_root}\n"
            f"    push   : {self.examples_root}  (excluding models/, .git/, logs/)\n"
            f"             -> {self.remote_root}\n"
            f"    venv   : python3 -m venv --system-site-packages .venv\n"
            f"    deps   : pip install -r requirements.txt (root)\n"
        )
        print(info)
        wipe = self._shell(
            f"rm -rf {shlex.quote(self.remote_root)} && "
            f"mkdir -p {shlex.quote(self.remote_root)}"
        )
        if not wipe.passed:
            result = PhaseResult("prepare-workspace", wipe.returncode, info, wipe.stderr)
            result.commands = wipe.commands
            return result
        return self._push_source_and_bootstrap_venv(info)

    def phase_push_models(self) -> PhaseResult:
        """Push every selected demo's model dirs from host to DUT."""
        model_dirs: list[str] = []
        for demo in self.demos:
            for md in self._spec_for(demo)["model_dirs"]:
                if md not in model_dirs:
                    model_dirs.append(md)
        info_lines = ["[dut_deploy] push models:"]
        missing = []
        for md in model_dirs:
            local = self.examples_root / "models" / md
            if not local.is_dir():
                missing.append(md)
                continue
            info_lines.append(f"    {md}  ({_file_size(str(local))})")
        info = "\n".join(info_lines) + "\n"
        info += f"    -> {self.remote_root}/models/<dir>\n"
        if missing:
            info += f"    MISSING on host: {missing}\n"
        print(info)

        if missing:
            return PhaseResult("push-models", 1, info,
                               "host-side models missing (run host conversion step 1 first)")
        proc_all = subprocess.run(["true"])
        result = PhaseResult("push-models", 0, info, "")
        for md in model_dirs:
            local = self.examples_root / "models" / md
            remote_parent = f"{self.remote_root}/models/{Path(md).parent}"
            mkdir = self._shell(f"mkdir -p {shlex.quote(remote_parent)}")
            result.commands.extend(mkdir.commands)
            push_cmd = f"adb -s {self.serial} push {local} {remote_parent}"
            proc = self._adb("push", str(local), remote_parent, check=False, timeout=10800)
            result.commands.append(push_cmd)
            if proc.returncode != 0:
                result.returncode = 1
                result.stderr += f"\n[push {md}] rc={proc.returncode}\n{proc.stderr}"
                break
            result.stdout += proc.stdout
        if result.passed:
            check = self._shell(
                " && ".join(
                    f"test -d {shlex.quote(self.remote_root + '/models/' + md)}"
                    for md in model_dirs
                )
            )
            result.commands.extend(check.commands)
            if not check.passed:
                result.returncode = 1
                result.stderr += "\n[dut_deploy] some model dirs missing on DUT after push"
        return result

    def phase_ko_swap(self) -> PhaseResult:
        remote_ko = f"{self.remote_root}/syna_npu.ko"
        info = (
            f"[dut_deploy] kernel module swap:\n"
            f"    local : {self.ko}  ({_file_size(self.ko)})\n"
            f"    remote: {remote_ko}\n"
            f"    steps : rmmod syna_npu -> adb push -> insmod {remote_ko} -> lsmod verify\n"
        )
        print(info)
        steps = []
        steps.append(self._shell("rmmod syna_npu 2>/dev/null || true"))
        push_cmd = f"adb -s {self.serial} push {self.ko} {remote_ko}"
        proc = self._adb("push", self.ko, remote_ko, check=False)
        push_ko = PhaseResult("push-ko", proc.returncode, info + proc.stdout, proc.stderr)
        push_ko.commands = [push_cmd]
        steps.append(push_ko)
        steps.append(self._shell(f"insmod {shlex.quote(remote_ko)}"))
        steps.append(self._shell("lsmod | grep syna_npu || echo 'syna_npu NOT LOADED'"))
        all_commands: list[str] = []
        for st in steps:
            all_commands.extend(st.commands)
        failed = [st for st in steps if not st.passed]
        if "syna_npu NOT LOADED" in steps[-1].stdout:
            failed.append(steps[-1])
        if failed:
            merged = failed[0]
            merged.stdout = "\n---\n".join(st.stdout for st in failed)
            merged.stderr = "\n---\n".join(st.stderr for st in failed)
            merged.commands = all_commands
            return merged
        steps[-1].commands = all_commands
        return steps[-1]

    def phase_wheel_install(self) -> PhaseResult:
        wheel_name = Path(self.wheel).name
        remote_wheel = f"/tmp/{wheel_name}"
        info = (
            f"[dut_deploy] push runtime wheel:\n"
            f"    local : {self.wheel}  ({_file_size(self.wheel)})\n"
            f"    remote: {remote_wheel}\n"
            f"    install into: {self.remote_root}/.venv  (pip --force-reinstall --no-deps)\n"
        )
        print(info)
        push_cmd = f"adb -s {self.serial} push {self.wheel} {remote_wheel}"
        proc = self._adb("push", self.wheel, remote_wheel, check=False)
        if proc.returncode != 0:
            result = PhaseResult("push-wheel", proc.returncode, info + proc.stdout, proc.stderr)
            result.commands = [push_cmd]
            return result
        script = (
            f"cd {shlex.quote(self.remote_root)} && . {shlex.quote(self.remote_root + '/.venv/bin/activate')} && "
            f"pip install --force-reinstall --no-deps {shlex.quote(remote_wheel)} && "
            f"pip show torq-runtime | grep -E '^(Name|Version)' && "
            f"python -c 'import torq.runtime; print(\"torq.runtime import OK\")'"
        )
        result = self._shell(script)
        result.commands = [push_cmd, *result.commands]
        result.stdout = info + result.stdout
        if result.passed:
            result.stdout += f"\n[dut_deploy] installed {wheel_name} on DUT\n"
        return result

    # ------------------------------------------------- per-demo inference
    def _generate_wav(self, demo: str) -> Path:
        """1s of 16 kHz mono silence, same as the legacy DUT test wav."""
        if demo in self._wav_cache:
            return self._wav_cache[demo]
        path = Path("/tmp") / f"torq_deploy_{demo}_test.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)
        self._wav_cache[demo] = path
        return path

    def _build_infer_commands(
        self, demo: str, model_dir: str | None = None,
    ) -> list[tuple[str, str, str | None, str, str]]:
        """Return (case_name, remote_command, expect_substring, repo_name, vmfb_name)
        tuples. vmfb_name is "" for whole-model-dir runs (ASR); then case_name stays
        demo:repo. Otherwise case_name is demo:repo:vmfb so the per-model-file record
        joins the profile records by the same key."""
        spec = self._spec_for(demo)
        builder = spec["builder"]
        app_dir = f"{self.remote_root}/{spec['app_dir']}"
        activate = f". {shlex.quote(self.remote_root + '/.venv/bin/activate')}"
        out: list[tuple[str, str, str | None, str, str]] = []
        expect = spec.get("expect")
        model_dirs = spec["model_dirs"]
        if model_dir is not None:
            if model_dir not in model_dirs:
                raise DutDeployError(
                    f"{demo}: model dir {model_dir} not in spec {model_dirs}"
                )
            model_dirs = [model_dir]

        def rel(model_rel: str, *parts: str) -> str:
            return f"{self.remote_root}/models/{model_rel}/{'/'.join(parts)}"

        for model_rel in model_dirs:
            local_model = self.examples_root / "models" / model_rel
            mdir = f"{self.remote_root}/models/{model_rel}"
            repo_name = Path(model_rel).name
            if builder in ("llm_chat", "llm_chat_plain"):
                tail = (
                    " --instruct-model"
                    if builder == "llm_chat"
                    else (" " + spec["extra_flags"] if spec.get("extra_flags") else "")
                )
                main = _first_match(local_model, ["transformer*.vmfb*", "model*.vmfb*"])
                if main is None:
                    raise DutDeployError(f"{demo}: no main vmfb in {model_rel}")
                main_r = rel(model_rel, main.name)
                lm_head_arg = ""
                if "transformer" in main.name.lower():
                    lm = _first_match(local_model, ["lm_head*.vmfb*"])
                    if lm is not None:
                        lm_head_arg = f" --lm-head {shlex.quote(rel(model_rel, lm.name))}"
                prompt = "Hello, briefly introduce yourself."
                out.append((
                    f"{demo}:{repo_name}:{main.name}",
                    f"cd {shlex.quote(app_dir)} && {activate} && "
                    f"printf '%s\\n%s\\n' {shlex.quote(prompt)} quit | "
                    f"python src/infer.py -m {shlex.quote(main_r)}{lm_head_arg}{tail}",
                    expect,
                    repo_name,
                    main.name,
                ))
            elif builder == "llm_chat_body":
                main = _first_match(local_model, ["body*.vmfb*", "transformer*.vmfb*"]) or \
                       _first_match(local_model, ["model*.vmfb*"])
                if main is None:
                    raise DutDeployError(f"{demo}: no main vmfb in {model_rel}")
                lm_head_arg = ""
                lm = _first_match(local_model, ["lm_head*.vmfb*"])
                if lm is not None:
                    lm_head_arg = f" --lm-head {shlex.quote(rel(model_rel, lm.name))}"
                prompt = "Hello, briefly introduce yourself."
                out.append((
                    f"{demo}:{repo_name}:{main.name}",
                    f"cd {shlex.quote(app_dir)} && {activate} && "
                    f"printf '%s\\n%s\\n' {shlex.quote(prompt)} quit | "
                    f"python src/infer.py -m {shlex.quote(rel(model_rel, main.name))}"
                    f"{lm_head_arg} --instruct-model",
                    expect,
                    repo_name,
                    main.name,
                ))
            elif builder in ("asr", "asr_streaming"):
                wav_local = self._generate_wav(demo)
                wav_remote = f"{self.remote_root}/refactor_inputs/{demo}.wav"
                out.append((
                    f"{demo}:{repo_name}",
                    f"cd {shlex.quote(app_dir)} && {activate} && "
                    f"python src/infer.py -m {shlex.quote(mdir)}"
                    + (f" {shlex.quote(wav_remote)}" if builder == "asr"
                       else f" --wav {shlex.quote(wav_remote)}"),
                    expect,
                    repo_name,
                    "",
                ))
            elif builder == "vl":
                main = _first_match(local_model, ["decoder_nolm*.vmfb*", "model*.vmfb*"])
                lm = _first_match(local_model, ["lm_head*.vmfb*"])
                vision = _first_match(local_model, ["vision_encoder*.vmfb*"])
                dec_a = _first_match(local_model, ["decoder_image*_A*.vmfb*"])
                dec_b = _first_match(local_model, ["decoder_image*_B*.vmfb*"])
                image = _first_match(local_model, ["*.jpg", "*.jpeg", "*.png"])
                for needed, label in ((main, "decoder_nolm"), (lm, "lm_head"),
                                      (vision, "vision_encoder"), (dec_a, "image_decoder_A"),
                                      (dec_b, "image_decoder_B"), (image, "image")):
                    if needed is None:
                        raise DutDeployError(f"{demo}: missing {label} file in {model_rel}")
                prefix = _image_decoder_prefix(dec_a.name)
                if _image_decoder_prefix(dec_b.name) != prefix:
                    raise DutDeployError(
                        f"{demo}: image-decoder A/B files do not share a prefix"
                    )
                prompt = "What is in this image?"
                out.append((
                    f"{demo}:{repo_name}:{main.name}",
                    f"cd {shlex.quote(app_dir)} && {activate} && "
                    f"python src/infer.py --model {shlex.quote(rel(model_rel, main.name))} "
                    f"--lm-head {shlex.quote(rel(model_rel, lm.name))} "
                    f"--vision {shlex.quote(mdir + '/' + str(vision.relative_to(local_model)))} "
                    f"--image-decoder {shlex.quote(mdir + '/' + prefix)} "
                    f"--image {shlex.quote(mdir + '/' + str(image.relative_to(local_model)))} "
                    f"--prompt {shlex.quote(prompt)}",
                    expect,
                    repo_name,
                    main.name,
                ))
            elif builder == "od":
                image = _first_match(local_model, ["*.jpg", "*.jpeg", "*.png"])
                labels = _first_match(local_model, ["labels.json"])
                if image is None or labels is None:
                    raise DutDeployError(f"{demo}: missing sample image or labels.json in {model_rel}")
                # exact .vmfb only: HF download sidecars (.vmfb.lock,
                # .vmfb.metadata, .cache/) must never be treated as modules
                for vmfb in _all_matches(local_model, "*.vmfb"):
                    name_l = vmfb.name.lower()
                    variant = "yolov8"
                    if "yolo26s" in name_l:
                        variant = "yolo26s"
                    elif "yolo26n" in name_l or "yolov26n" in Path(model_rel).name.lower():
                        variant = "yolo26n"
                    out.append((
                        f"{demo}:{Path(model_rel).name}:{vmfb.name}",
                        f"cd {shlex.quote(app_dir)} && {activate} && "
                        f"python src/infer.py --model {shlex.quote(mdir + '/' + str(vmfb.relative_to(local_model)))} "
                        f"--image {shlex.quote(mdir + '/' + str(image.relative_to(local_model)))} "
                        f"--labels {shlex.quote(mdir + '/' + str(labels.relative_to(local_model)))} "
                        f"--variant {shlex.quote(variant)} --device torq --device-io",
                        expect,
                        repo_name,
                        vmfb.name,
                    ))
            elif builder == "pose":
                image = _first_match(local_model, ["*.jpg", "*.jpeg", "*.png"])
                if image is None:
                    raise DutDeployError(f"{demo}: missing sample image in {model_rel}")
                # exact .vmfb only (see object_detection note above)
                for vmfb in _all_matches(local_model, "*.vmfb"):
                    out.append((
                        f"{demo}:{Path(model_rel).name}:{vmfb.name}",
                        f"cd {shlex.quote(app_dir)} && {activate} && "
                        f"python src/infer.py --model {shlex.quote(mdir + '/' + str(vmfb.relative_to(local_model)))} "
                        f"--image {shlex.quote(mdir + '/' + str(image.relative_to(local_model)))} --device torq --device-io",
                        expect,
                        repo_name,
                        vmfb.name,
                    ))
            elif builder == "face":
                main = _first_match(local_model, ["face_detection.vmfb*"])
                image = _first_match(local_model, ["face.jpg", "*.jpg", "*.jpeg", "*.png"])
                if main is None or image is None:
                    raise DutDeployError(f"{demo}: missing model or sample image in {model_rel}")
                out.append((
                    f"{demo}:{repo_name}:{main.name}",
                    f"cd {shlex.quote(app_dir)} && {activate} && "
                    f"python src/infer.py --model {shlex.quote(mdir + '/' + str(main.relative_to(local_model)))} "
                    f"--image {shlex.quote(mdir + '/' + str(image.relative_to(local_model)))} --device torq --device-io",
                    expect,
                    repo_name,
                    main.name,
                ))
            elif builder == "embed":
                # EmbeddingGemma-2: whole-model-dir run (like ASR), case is demo:repo.
                # Text-only query session: the model dir ships a 0-byte example image
                # and no example audio, and torq cannot load audio+image in one
                # session (NPU mapping budget). The single vmfb="" record maps every
                # profiled vmfb to this run's status (asr precedent).
                q1 = "a fluffy tabby and white cat with bright blue eyes"
                q2 = "a dog barking and growling"
                out.append((
                    f"{demo}:{repo_name}",
                    f"cd {shlex.quote(app_dir)} && {activate} && "
                    f"python src/infer.py -m {shlex.quote(mdir)} "
                    f"--queries {shlex.quote(q1)} {shlex.quote(q2)}",
                    expect,
                    repo_name,
                    "",
                ))
        return out

    def phase_infer(self, demo: str, model_dir: str | None = None) -> PhaseResult:
        """Demo deps + inputs + inference commands for one model dir.

        Runs every vmfb for the given model_dir (or all model_dirs when
        model_dir is None, standalone driver) without short-circuiting, and
        writes a per-demo merged {demo}_infer.json keyed by demo:repo:vmfb so
        the report can show each model file's inference status independently.
        """
        spec = self._spec_for(demo)
        repo = Path(model_dir).name if model_dir else ""
        base_phase = f"infer-{demo}__{repo}" if repo else f"infer-{demo}"

        if spec["builder"] is None:
            info = (
                f"[dut_deploy] infer {demo}: SKIP - known debt: {spec.get('debt', 'no builder')}\n"
                f"    (model dirs were pushed: {spec['model_dirs']})\n"
            )
            print(info)
            self._write_infer_json(demo, [{
                "demo_name": demo,
                "case_name": f"{demo}:{repo}" if repo else demo,
                "target": self.serial,
                "repo": repo,
                "vmfb": "",
                "command": "",
                "status": "skipped",
                "note": f"known debt: {spec.get('debt', 'no builder')}",
            }])
            return PhaseResult(base_phase, 0, info)

        info_lines = [f"[dut_deploy] infer {demo} (model {repo or 'ALL'}):"]
        app_reqs = f"{self.remote_root}/{spec['app_dir']}/requirements.txt"
        info_lines.append(f"    deps   : {app_reqs}  -> pip install into .venv")
        commands = self._build_infer_commands(demo, model_dir)
        for case_name, cmd, _expect, _r, _v in commands:
            info_lines.append(f"    case   : {case_name}")
        if spec.get("need_wav"):
            wav = self._generate_wav(demo)
            info_lines.append(f"    wav    : {wav}  -> {self.remote_root}/refactor_inputs/{demo}.wav")
        info = "\n".join(info_lines) + "\n"
        print(info)

        prep_script = (
            f"mkdir -p {shlex.quote(self.remote_root + '/refactor_inputs')} && "
            f"if [ -f {shlex.quote(app_reqs)} ]; then "
            f". {shlex.quote(self.remote_root + '/.venv/bin/activate')} && "
            f"pip install --no-cache-dir -r {shlex.quote(app_reqs)}; "
            f"else echo 'no per-demo requirements for {demo}'; fi"
        )
        prep = self._shell(prep_script)
        prep.stdout = info + prep.stdout
        if not prep.passed:
            prep.phase = base_phase
            return prep
        phase_commands: list[str] = list(prep.commands)
        if spec.get("need_wav"):
            wav = self._generate_wav(demo)
            wav_remote = f"{self.remote_root}/refactor_inputs/{demo}.wav"
            proc = self._adb("push", str(wav), wav_remote, check=False)
            phase_commands.append(f"adb -s {self.serial} push {wav} {wav_remote}")
            if proc.returncode != 0:
                prep.returncode = 1
                prep.commands = phase_commands
                prep.stderr += f"\n[wav push {demo}] rc={proc.returncode}\n{proc.stderr}"
                prep.phase = base_phase
                return prep

        records: list[dict] = []
        combined = PhaseResult(base_phase, 0, "", "")
        combined.commands = phase_commands
        for case_name, cmd, expect, rname, vmfb in commands:
            case = self._shell(cmd, timeout=1800)
            combined.commands.extend(case.commands)
            case.stdout = f"===== case {case_name} =====\n$ {cmd}\n{case.stdout}"
            status = "pass"
            if not case.passed:
                status = "fail"
            elif expect and expect not in case.stdout:
                status = "fail"
                case.returncode = 1
                case.stderr += f"\n[dut_deploy] expected {expect!r} in output, not found"
            combined.stdout += case.stdout + "\n"
            combined.stderr += case.stderr
            if status == "fail" and combined.returncode == 0:
                combined.returncode = case.returncode or 1
            records.append({
                "demo_name": demo,
                "case_name": case_name,
                "target": self.serial,
                "repo": rname,
                "vmfb": vmfb,
                "command": cmd,
                "status": status,
                "note": "",
            })
        self._write_infer_json(demo, records)
        failed = [r["case_name"] for r in records if r["status"] != "pass"]
        if failed:
            combined.stderr += (
                f"\n[dut_deploy] infer {demo} (model {repo or 'ALL'}): "
                f"{len(failed)}/{len(records)} model file(s) failed: {', '.join(failed)}"
            )
        return combined

    # ------------------------------------------------------------- profiling
    def _profile_targets(
        self, demo: str, model_dir: str | None = None,
    ) -> list[tuple[str, str]]:
        """(case_name, remote model path) pairs to profile for one demo.

        Mirrors the legacy _profile_model_paths semantics with host-side,
        strict file matching. When model_dir is given, only that model dir is
        targeted (per-model-dir test case); otherwise all of the demo's dirs.
        """
        spec = self._spec_for(demo)
        primary: dict[str, list[str]] = {
            "gemma3": ["transformer*.vmfb", "model.vmfb.trim", "model*.vmfb"],
            "LiquidAI-LFM2.5-230M": ["body*.vmfb", "model*.vmfb"],
            "LiquidAI-LFM2.5": ["body*.vmfb", "transformer*.vmfb", "model*.vmfb"],
            "LiquidAI-LFM2-VL-450M": ["decoder_nolm*.vmfb", "model*.vmfb"],
            "moonshine": ["encoder.vmfb"],
            "moonshine_streaming": ["encoder.vmfb"],
        }
        model_dirs = spec["model_dirs"]
        if model_dir is not None:
            if model_dir not in model_dirs:
                raise DutDeployError(f"{demo}: model dir {model_dir} not in spec {model_dirs}")
            model_dirs = [model_dir]
        targets: list[tuple[str, str]] = []
        for model_rel in model_dirs:
            local_model = self.examples_root / "models" / model_rel
            mdir = f"{self.remote_root}/models/{model_rel}"
            dir_name = Path(model_rel).name
            if demo in primary:
                found = _first_match(local_model, primary[demo])
                if found is None:
                    raise DutDeployError(
                        f"{demo}: no profileable VMFB matching {primary[demo]} in {model_rel}"
                    )
                targets.append(
                    (f"{demo}:{dir_name}:{found.name}", mdir + "/" + found.name)
                )
            else:
                files = _profile_vmfb_files(local_model)
                if not files:
                    raise DutDeployError(f"{demo}: no profileable VMFB in {model_rel}")
                for found in files:
                    targets.append((
                        f"{demo}:{dir_name}:{found.name}",
                        mdir + "/" + str(found.relative_to(local_model)),
                    ))
        return targets

    def _write_demo_json(self, demo: str, kind: str, records: list[dict]) -> None:
        """Merge `records` into the per-demo {demo}_{kind}.json, dedup by
        (demo_name, case_name, repo). Multi-repo demos are inferred/profiled in
        separate per-model-dir calls and must not clobber each other, so the
        file is read-back and unioned instead of overwritten."""
        path = self.log_dir / f"{demo}_{kind}.json"
        existing: list[dict] = []
        try:
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = []
        seen = {(r.get("demo_name"), r.get("case_name"), r.get("repo", ""))
                for r in existing}
        merged = list(existing)
        for r in records:
            key = (r.get("demo_name"), r.get("case_name"), r.get("repo", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _write_infer_json(self, demo: str, records: list[dict]) -> None:
        self._write_demo_json(demo, "infer", records)

    def _write_profile_json(self, demo: str, records: list[dict]) -> None:
        self._write_demo_json(demo, "profile", records)

    def phase_profile(self, demo: str, model_dir: str | None = None) -> PhaseResult:
        """Profile a demo's VMFBs via the repo-root profile.py.

        Runs every profile target for the given model_dir (or all model_dirs
        when model_dir is None, standalone driver) without short-circuiting,
        and writes a per-demo merged {demo}_profile.json keyed by
        demo:repo:vmfb where each record carries its own status
        (pass / fail / missing-metrics).
        """
        spec = self._spec_for(demo)
        repo = Path(model_dir).name if model_dir else ""
        base_phase = f"profile-{demo}__{repo}" if repo else f"profile-{demo}"
        try:
            targets = self._profile_targets(demo, model_dir)
        except DutDeployError as exc:
            return PhaseResult(base_phase, 1, "", str(exc))
        info = f"[dut_deploy] profile {demo} model {repo or 'ALL'} (profile.py --repeat 3):\n" + "".join(
            f"    case: {name}\n        model: {path}\n" for name, path in targets
        )
        print(info)
        # ensure per-demo requirements (legacy flow installed them before
        # profiling too; debt demos skip the infer phase and would miss it)
        app_reqs = f"{self.remote_root}/{spec['app_dir']}/requirements.txt"
        deps_script = (
            f"if [ -f {shlex.quote(app_reqs)} ]; then "
            f". {shlex.quote(self.remote_root + '/.venv/bin/activate')} && "
            f"pip install --no-cache-dir -r {shlex.quote(app_reqs)}; "
            f"else echo 'no per-demo requirements for {demo}'; fi"
        )
        deps = self._shell(deps_script)
        if not deps.passed:
            deps.phase = base_phase
            deps.stdout = info + deps.stdout
            deps.stderr += "\n[dut_deploy] per-demo dependency install failed before profiling"
            return deps
        check = self._shell(
            f"test -f {shlex.quote(self.remote_root + '/profile.py')} && echo PROFILE_ENTRY_OK"
        )
        if "PROFILE_ENTRY_OK" not in check.stdout:
            check.commands = [*deps.commands, *check.commands]
            check.phase = base_phase
            check.stdout = info + check.stdout
            check.stderr += "\n[dut_deploy] profile.py missing at DUT repo root"
            return check

        records: list[dict] = []
        combined = PhaseResult(base_phase, 0, info, "")
        combined.commands = [*deps.commands, *check.commands]
        runtime_version = self._runtime_version()
        for case_name, model_path in targets:
            command = (
                f"cd {shlex.quote(self.remote_root)} && "
                f". {shlex.quote(self.remote_root + '/.venv/bin/activate')} && "
                f"python profile.py {shlex.quote(model_path)} --repeat 3"
            )
            case = self._shell(command, timeout=3600)
            combined.commands.extend(case.commands)
            case.stdout = f"===== case {case_name} =====\n$ {command}\n{case.stdout}"
            combined.stdout += case.stdout + "\n"
            combined.stderr += case.stderr
            parts = case_name.split(":")
            rname = parts[1] if len(parts) >= 2 else ""
            vmfb = parts[2] if len(parts) >= 3 else ""
            status = "pass"
            note = ""
            metrics: dict = {}
            if not case.passed:
                status = "fail"
                note = f"profile command rc={case.returncode}"
            else:
                metrics = parse_profile_metrics(case.stdout)
                missing = [label for label in PROFILE_METRIC_LABELS if label not in metrics]
                if missing:
                    status = "missing-metrics"
                    note = "missing: " + ", ".join(missing)
            if status != "pass" and combined.returncode == 0:
                combined.returncode = 1
            records.append({
                "demo_name": demo,
                "case_name": case_name,
                "target": self.serial,
                "runtime_version": runtime_version,
                "model_path": model_path,
                "command": command,
                "repo": rname,
                "vmfb": vmfb,
                "status": status,
                "note": note,
                "metrics": metrics,
            })
        self._write_profile_json(demo, records)
        failed = [r["case_name"] for r in records if r["status"] != "pass"]
        combined.stdout += (
            f"[dut_deploy] profiled {len(records)} model(s) for {demo} "
            f"model {repo or 'ALL'}\n"
        )
        if failed:
            combined.stderr += (
                f"\n[dut_deploy] profile {demo} model {repo or 'ALL'}: "
                f"{len(failed)}/{len(records)} model file(s) not passing: "
                f"{', '.join(failed)}"
            )
        return combined

    # ----------------------------------------------------------------- logs
    def _write_log(self, result: PhaseResult) -> None:
        try:
            safe = result.phase.replace(":", "_")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            commands_block = ""
            if result.commands:
                commands_block = (
                    "----- commands (reproduce from host) -----\n"
                    + "".join(f"$ {c}\n" for c in result.commands)
                    + "\n"
                )
            (self.log_dir / f"dut_deploy_{safe}.log").write_text(
                f"returncode: {result.returncode}\n\n"
                f"{commands_block}"
                f"----- stdout -----\n{result.stdout}\n"
                f"----- stderr -----\n{result.stderr}\n",
                encoding="utf-8",
            )
        except OSError:
            pass


# --------------------------------------------------------------------------
# Standalone debug driver (no TorqExamplesTest / pytest dependency)
# --------------------------------------------------------------------------

def _print_result(result: PhaseResult) -> None:
    print(f"\n{'=' * 70}")
    print(f"== {result.phase}  (rc={result.returncode})")
    print("=" * 70)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("----- stderr -----")
        print(result.stderr)


def _standalone_main(argv: list[str]) -> int:
    """Usage:
        TORQ_DUT_RUNTIME_WHEEL=/path/torq_runtime-...-aarch64.whl \\
        TORQ_DUT_KO=/path/syna_npu.ko \\
        python3 dut_deploy.py <examples_root> [demo ...]

    No demo args = the full 12-demo matrix. Known-debt demos are skipped
    with an explicit note (models are still pushed).
    """
    if not argv:
        print(__doc__)
        return 2
    examples_root = Path(argv[0])
    demos = argv[1:] or None

    if not (examples_root / "setup_demos.py").exists():
        print(f"FATAL: {examples_root}/setup_demos.py not found - not a torq-examples checkout?")
        return 1
    if not os.getenv("TORQ_DUT_RUNTIME_WHEEL", "").strip():
        print("FATAL: set TORQ_DUT_RUNTIME_WHEEL (aarch64 torq-runtime wheel)")
        return 1
    if not os.getenv("TORQ_DUT_KO", "").strip():
        print("FATAL: set TORQ_DUT_KO (path to syna_npu.ko)")
        return 1

    runner = DutDeployRunner(examples_root, demos)
    print("== standalone DUT deploy + infer debug ==")
    print(f"   examples : {runner.examples_root}")
    print(f"   demos    : {', '.join(runner.demos)}")
    print(f"   wheel    : {runner.wheel}")
    print(f"   ko       : {runner.ko}")
    print(f"   remote   : {runner.remote_root}")

    try:
        runner.acquire_board()
    except DutDeployError as exc:
        print(f"FATAL: {exc}")
        return 1
    print(f"   serial   : {runner.serial}  (via torq.utils.boards)")

    failed = False
    deploy_phases = (
        ("connectivity", runner.phase_connectivity),
        ("prepare-workspace", runner.phase_prepare_workspace),
        ("push-models", runner.phase_push_models),
        ("ko-swap", runner.phase_ko_swap),
        ("wheel-install", runner.phase_wheel_install),
    )
    if os.getenv("TORQ_DUT_SKIP_DEPLOY", "") == "1":
        deploy_phases = deploy_phases[:1]
        print("   (TORQ_DUT_SKIP_DEPLOY=1: running infer on current DUT state)")
    try:
        for name, fn in deploy_phases:
            try:
                result = fn()
            except DutDeployError as exc:
                print(f"\nFATAL in phase {name}: {exc}")
                return 1
            result.phase = name
            _print_result(result)
            runner._write_log(result)
            if not result.passed:
                print(f"\n!! phase {name} failed - DUT left in current state.")
                print(f"   inspect: adb -s {runner.serial} shell")
                return 1

        for demo in runner.demos:
            try:
                result = runner.phase_infer(demo)
            except DutDeployError as exc:
                print(f"\nFATAL in infer {demo}: {exc}")
                failed = True
                break
            _print_result(result)
            runner._write_log(result)
            if not result.passed:
                failed = True
                print(f"\n!! infer {demo} failed - continuing with remaining demos")

        print("\n---------------- profiling ----------------")
        for demo in runner.demos:
            try:
                result = runner.phase_profile(demo)
            except DutDeployError as exc:
                print(f"\nFATAL in profile {demo}: {exc}")
                failed = True
                break
            _print_result(result)
            runner._write_log(result)
            if not result.passed:
                failed = True
                print(f"\n!! profile {demo} failed - continuing with remaining demos")
    finally:
        runner.release_board()

    if not failed:
        print("\n>> DUT DEPLOY + INFER + PROFILE MATRIX VERIFIED")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    sys.exit(_standalone_main(sys.argv[1:]))
