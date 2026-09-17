import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

#/home/TorqExamplesTest/log
SUITE_LOG_FOLDER = Path(__file__).resolve().parent
#/home/TorqExamplesTest/
SUITE_SCRIPT_DIR = Path(os.getenv("SUITE_SCRIPT_DIR"))


def resolve_dut_repo_path(env=None):
    env = env or os.environ
    if env.get("DUT_REPO_DIR"):
        return str(Path(env["DUT_REPO_DIR"]).expanduser().resolve())

    guessed = SUITE_LOG_FOLDER.parent.resolve()
    if (guessed / "setup_demos.py").exists() and (guessed / "requirements.txt").exists():
        return str(guessed)
    return "/home/torq-examples"


DEFAULT_SMOKE_DEMO = os.getenv("DEMO_NAME", "")
FULL_SUITE_ENABLED = os.getenv("RUN_FULL_DEMO_SUITE", "1").strip().lower() in {"1", "true", "yes", "on"}

DUT_REPO_PATH = resolve_dut_repo_path()
DUT_LOG_DIR = f"{DUT_REPO_PATH}/log"
SUITE_SCRIPT_DIR = Path(os.getenv("SUITE_SCRIPT_DIR"))
SUITE_LOG_FOLDER = Path(os.getenv("SUITE_LOG_FOLDER", str(Path(DUT_REPO_PATH) / "log")))
LOG_FMT = logging.Formatter('%(asctime)s %(levelname)-8s [%(funcName)s:%(lineno)d] %(message)s')
LOG_LEVEL = logging.INFO


def _run_local_shell(command, timeout=60):
    result = subprocess.run(["bash", "-lc", command], cwd=DUT_REPO_PATH, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if stderr:
            raise RuntimeError(stderr)
    return result.stdout.strip()


def _parse_demo_names(raw_value):
    if raw_value is None:
        return []
    cleaned = str(raw_value).strip()
    if not cleaned:
        return []
    cleaned = cleaned.strip("[]")
    cleaned = cleaned.replace("'", "").replace('"', "")
    return [name.strip() for name in cleaned.split(",") if name.strip()]


def _load_demo_names():
    local_setup = f"{DUT_REPO_PATH}/setup_demos.py"
    if not Path(local_setup).is_file():
        raise FileNotFoundError(f"torq-examples repo not present on DUT at {DUT_REPO_PATH}; ensure the repo is cloned and available")

    if os.getenv("DEMO_NAMES"):
        demo_names = _parse_demo_names(os.getenv("DEMO_NAMES"))
        if demo_names:
            return demo_names

    if DEFAULT_SMOKE_DEMO:
        return [DEFAULT_SMOKE_DEMO]

    output = _run_local_shell(f"python3 {DUT_REPO_PATH}/read_demo_names.py")

    try:
        parsed = _parse_demo_names(output)
        if parsed:
            return parsed
        raise RuntimeError(f"Empty demo list resolved from DUT: {output}")
    except Exception as exc:  # pragma: no cover - environment-dependent validation
        raise RuntimeError(f"Failed to parse DUT demo list: {output}") from exc


def _expected_demo_asset_paths(demo_name):
    expected = {
        "gemma3": [f"{DUT_REPO_PATH}/models/Synaptics/gemma-3-270m-it-torq"],
        "moonshine": [f"{DUT_REPO_PATH}/models/Synaptics/moonshine-tiny-bf16-torq"],
        "moonshine_streaming": [f"{DUT_REPO_PATH}/models/Synaptics/moonshine-streaming-tiny-torq"],
        "object_detection": [f"{DUT_REPO_PATH}/models/Synaptics/yolov8-od-nano-320-int8-torq"],
        "pose_estimation": [f"{DUT_REPO_PATH}/models/Synaptics/yolov8-pose-nano-320-int8-torq"],
        "LiquidAI-LFM2.5-230M": [f"{DUT_REPO_PATH}/models/Synaptics/LiquidAI-LFM2.5-230M"],
        "LiquidAI-LFM2-VL-450M": [f"{DUT_REPO_PATH}/models/Synaptics/LiquidAI-LFM2-VL-450M"],
    }
    return expected.get(demo_name, [])


class TestDemoBoard:
    _setup_failures = set()

    @classmethod
    def _record_setup_failure(cls, demo_name):
        cls._setup_failures.add(demo_name)

    def _skip_if_setup_failed(self, demo_name):
        if demo_name in self.__class__._setup_failures:
            pytest.skip(
                f"Skipped because setup previously failed for this demo: {demo_name}"
            )

    @staticmethod
    def _make_unique_dut_log_path(script_label, *extra_tokens):
        token_parts = [str(part) for part in [script_label, *extra_tokens] if str(part).strip()]
        normalized = []
        for part in token_parts:
            cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", part).strip("_")
            if cleaned:
                normalized.append(cleaned)
        safe_label = "_".join(normalized) if normalized else "run"
        safe_label = safe_label[:80]
        safe_label = re.sub(r"[-_.]+$", "", safe_label)
        return f"{DUT_LOG_DIR}/{safe_label}.log"

    @classmethod
    def setup_class(cls):
        if SUITE_LOG_FOLDER.exists():
            shutil.rmtree(str(SUITE_LOG_FOLDER))
        SUITE_LOG_FOLDER.mkdir(parents=True, exist_ok=True)
        logging.getLogger().setLevel(LOG_LEVEL)

        main_conhdl = logging.StreamHandler()
        main_loghdl = logging.FileHandler(SUITE_LOG_FOLDER / "main.log")
        main_loghdl.setFormatter(LOG_FMT)
        main_loghdl.setLevel(LOG_LEVEL)
        main_conhdl.setFormatter(LOG_FMT)
        cls.logger = logging.getLogger("main")
        cls.logger.addHandler(main_loghdl)
        cls.logger.addHandler(main_conhdl)

        case_loghdl = logging.FileHandler(SUITE_LOG_FOLDER / "suite.log")
        case_loghdl.setLevel(LOG_LEVEL)
        cls.suite_logger = logging.getLogger("suite")
        cls.suite_logger.addHandler(case_loghdl)

        execute_loghdl = logging.FileHandler(SUITE_LOG_FOLDER / "execute.log")
        execute_loghdl.setFormatter(LOG_FMT)
        execute_loghdl.setLevel(LOG_LEVEL)
        cls.execute_logger = logging.getLogger("execute")
        cls.execute_logger.addHandler(execute_loghdl)

        cls.logger.info("Local DUT execution mode detected")
        cls.logger.info("DUT_REPO_PATH: %s", DUT_REPO_PATH)
        cls.logger.info("DUT_LOG_DIR: %s", DUT_LOG_DIR)
        Path(DUT_LOG_DIR).mkdir(parents=True, exist_ok=True)
        cls.logger.info("Created DUT log directory: %s", DUT_LOG_DIR)

        cls._setup_failures.clear()
        cls.DEMOS = _load_demo_names()
        print(f"[SETUP] Demo queue: {cls.DEMOS}", flush=True)
        cls.logger.info("Demo queue: %s", cls.DEMOS)
        if cls.DEMOS:
            cls.logger.info("Selected demo setup flow: one setup + one test per demo, no unrelated requirements are installed")

    @classmethod
    def teardown_class(cls):
        cls.logger.info("Local DUT run: report artifacts remain under %s", SUITE_LOG_FOLDER)

    def _remote_path_exists(self, remote_path):
        return Path(str(remote_path)).exists()

    def _check_setup_artifacts(self, demo_name):
        targets = _expected_demo_asset_paths(demo_name)
        if not targets:
            self.logger.info("No setup artifact checkpoint defined for demo '%s'; skipping check", demo_name)
            return

        for target in targets:
            present = self._remote_path_exists(target)
            self.logger.info("Setup checkpoint for %s: %s -> %s", demo_name, target, "present" if present else "missing")
            if not present:
                self.logger.warning(
                    "Setup checkpoint failed for '%s': expected model/artifact directory is missing on DUT: %s",
                    demo_name,
                    target,
                )

    def _ensure_hello_world_wav_on_dut(self):
        host_wav = (SUITE_SCRIPT_DIR / "hello_world_16k.wav").resolve()
        if not host_wav.exists():
            raise FileNotFoundError(
                f"hello_world_16k.wav is required before running Moonshine tests: {host_wav}"
            )

        remote_wav = Path(DUT_REPO_PATH) / "hello_world_16k.wav"
        shutil.copy2(host_wav, remote_wav)
        self.logger.info("Placed hello_world_16k.wav in DUT repo: %s", remote_wav)
        return str(remote_wav)

    def _run_dut_script(self, script_name, *args, timeout=600):
        local_script = (SUITE_SCRIPT_DIR / script_name).resolve()
        if not local_script.exists():
            raise FileNotFoundError(f"DUT demo script not found: {local_script}")

        repo_root = Path(DUT_REPO_PATH)
        repo_root.mkdir(parents=True, exist_ok=True)
        log_dir = Path(DUT_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        if local_script.name in {"run_demo_moonshine.sh", "run_demo_moonshine_streaming.sh"}:
            self._ensure_hello_world_wav_on_dut()

        script_label = Path(script_name).stem
        log_file = Path(self._make_unique_dut_log_path(script_label, *args))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        cmd_text = " ".join(str(part) for part in [str(local_script), *args])
        shell_args = " ".join(shlex.quote(str(part)) for part in args)

        print(f"[RUN] {cmd_text}", flush=True)
        self.logger.info("CMD: bash %s %s", local_script, shell_args)
        self.logger.info("DUT script log file: %s", log_file)
        self.suite_logger.info("[STEP] Run cmd: bash %s %s", local_script, shell_args)
        self.execute_logger.info("Run cmd: bash %s %s", local_script, shell_args)
        with open(log_file, "w", encoding="utf-8") as handle:
            result = subprocess.run(["bash", str(local_script), *args], cwd=str(repo_root), stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        stdout = log_file.read_text(encoding="utf-8", errors="replace")
        stderr = ""
        self.execute_logger.info("DUT log file: %s", log_file)
        self.execute_logger.info("STDOUT:\n%s", stdout or "<empty>")
        if result.returncode != 0:
            failure_msg = (
                f"DUT script failed: {cmd_text}\n"
                f"STDOUT:\n{stdout}\n"
                f"EXPECTED LOG PATH: {log_file}\n"
                f"DUT LOG DIR: {DUT_LOG_DIR}"
            )
            self.logger.error(failure_msg)
            self.suite_logger.error(failure_msg)
            pytest.fail(failure_msg)
        return stdout, stderr

    @pytest.mark.parametrize("demo_name", _load_demo_names(), ids=lambda name: str(name))
    def test_demo_setup_on_dut(self, demo_name):
        """Run the real one-time setup entrypoint for every demo on the DUT."""
        self._skip_if_setup_failed(demo_name)

        targets = _expected_demo_asset_paths(demo_name)
        if targets:
            all_present = True
            for target in targets:
                if not self._remote_path_exists(target):
                    all_present = False
                    break
            if all_present:
                self.logger.info("Setup artifacts for %s already present on DUT; skipping setup execution", demo_name)
                pytest.skip(f"Setup checkpoints already present on DUT for {demo_name}")

        try:
            self._run_dut_script("run_demo_setup.sh", demo_name, timeout=1800)
            self._check_setup_artifacts(demo_name)
        except (Exception, pytest.fail.Exception):  # pragma: no cover - environment-dependent validation
            self.__class__._record_setup_failure(demo_name)
            raise

    def _discover_model_dirs(self):
        roots = [
            f"{DUT_REPO_PATH}/models",
            f"{DUT_REPO_PATH}/models/Synaptics",
        ]
        discovered = []
        for root in roots:
            if not self._remote_path_exists(root):
                continue
            find_cmd = (
                f"find {shlex.quote(root)} -maxdepth 3 -mindepth 1 -type d "
                f"! -path '*/.*' 2>/dev/null | sort"
            )
            completed = subprocess.run(["bash", "-lc", find_cmd], cwd=DUT_REPO_PATH, capture_output=True, text=True)
            stdout = completed.stdout
            stderr = completed.stderr
            if stderr and stderr.strip():
                self.logger.warning("Model directory discovery stderr for %s: %s", root, stderr.strip())
            for line in stdout.splitlines():
                path = line.strip()
                if path:
                    discovered.append(path)
        unique = []
        seen = set()
        for path in discovered:
            if path not in seen:
                seen.add(path)
                unique.append(path)
        return unique

    def _inference_spec_for_model_dir(self, model_dir):
        lower = model_dir.lower()
        if "liquidai-lfm2-vl-450m" in lower:
            return (
                "LiquidAI-LFM2-VL-450M",
                "run_demo_liquidai_lfm2_vl_450m.sh",
                [
                    "--model",
                    f"{model_dir}/decoder_nolm.vmfb",
                    "--lm-head",
                    f"{model_dir}/lm_head.vmfb",
                    "--vision",
                    f"{model_dir}/vision_encoder_256.vmfb",
                    "--image-decoder",
                    f"{model_dir}/decoder_image_2part_",
                    "--image",
                    f"{model_dir}/cats-and-dogs-256.jpg",
                    "--prompt",
                    "What is in this image?",
                ],
            )
        if "liquidai-lfm2.5-230m" in lower:
            return (
                "LiquidAI-LFM2.5-230M",
                "run_demo_liquidai_lfm2_5_230m.sh",
                [
                    "--model",
                    f"{model_dir}/body.vmfb",
                    "--lm-head",
                    f"{model_dir}/lm_head.vmfb",
                    "--instruct-model",
                    "--prompt",
                    "Hello there",
                ],
            )
        if "gemma" in lower:
            model_path = f"{model_dir}/transformer.vmfb"
            return (
                "gemma3",
                "run_demo_gemma3.sh",
                [
                    "--model",
                    model_path,
                    "--instruct-model",
                    "--prompt",
                    "Hello there",
                ],
            )
        if "moonshine" in lower and "streaming" in lower:
            return (
                "moonshine_streaming",
                "run_demo_moonshine_streaming.sh",
                [
                    "--model-dir",
                    model_dir,
                ],
            )
        if "moonshine" in lower:
            return (
                "moonshine",
                "run_demo_moonshine.sh",
                [
                    "--model-dir",
                    model_dir,
                ],
            )
        if "yolov8-od" in lower or "object" in lower:
            model_path = f"{model_dir}/yolo_8n_2.0.0_npu.vmfb"
            image_path = f"{model_dir}/samples/dog_bike_car.jpg"
            labels_path = f"{model_dir}/labels.json"
            return (
                "object_detection",
                "run_demo_object_detection.sh",
                [
                    "--model",
                    model_path,
                    "--image",
                    image_path,
                    "--labels",
                    labels_path,
                    "--device",
                    "torq",
                    "--device-io",
                ],
            )
        if "yolov8-pose" in lower or "pose" in lower:
            model_path = f"{model_dir}/yolo_pose.vmfb"
            image_path = f"{model_dir}/samples/pose.jpg"
            return (
                "pose_estimation",
                "run_demo_pose_estimation.sh",
                [
                    "--model",
                    model_path,
                    "--image",
                    image_path,
                    "--device",
                    "torq",
                    "--device-io",
                ],
            )
        return None

    def _find_runnable_spec_for_demo(self, demo_name):
        roots = [
            f"{DUT_REPO_PATH}/models",
            f"{DUT_REPO_PATH}/models/Synaptics",
        ]
        print(f"[MODEL] Scanning DUT model roots for demo={demo_name}:", flush=True)
        for root in roots:
            exists = self._remote_path_exists(root)
            print(f"[MODEL] root={root}: {'exists' if exists else 'missing'}", flush=True)
            self.logger.info("Model root check: %s -> %s", root, "exists" if exists else "missing")

        model_dirs = self._discover_model_dirs()
        print(f"[MODEL] Discovered model dirs: {model_dirs if model_dirs else '<none>'}", flush=True)
        self.logger.info("Discovered model dirs: %s", model_dirs)

        for model_dir in model_dirs:
            spec = self._inference_spec_for_model_dir(model_dir)
            if spec is not None and spec[0] == demo_name:
                print(f"[MODEL] {demo_name}: {model_dir}", flush=True)
                self.logger.info("Runnable model discovered: demo=%s model_dir=%s script=%s", demo_name, model_dir, spec[1])
                return (model_dir, *spec)

        expected_paths = _expected_demo_asset_paths(demo_name)
        for candidate in expected_paths:
            if not self._remote_path_exists(candidate):
                continue
            spec = self._inference_spec_for_model_dir(candidate)
            if spec is not None and spec[0] == demo_name:
                print(f"[MODEL] {demo_name}: {candidate} (fallback)", flush=True)
                self.logger.info(
                    "Runnable model discovered via fallback: demo=%s model_dir=%s script=%s",
                    demo_name,
                    candidate,
                    spec[1],
                )
                return (candidate, *spec)

        return None

    @pytest.mark.parametrize("demo_name", _load_demo_names(), ids=lambda name: str(name))
    def test_demo_inference_on_dut(self, demo_name):
        self._skip_if_setup_failed(demo_name)
        spec = self._find_runnable_spec_for_demo(demo_name)
        if spec is None:
            pytest.skip(f"No runnable inference spec found for selected demo '{demo_name}' on the DUT")

        model_dir, resolved_demo_name, script_name, args = spec
        self.logger.info("Running inferred demo '%s' from model dir %s", resolved_demo_name, model_dir)
        self._run_dut_script(script_name, *args)

    def _profiling_spec_for_model_dir(self, model_dir):
        lower = model_dir.lower()
        if "liquidai-lfm2-vl-450m" in lower:
            return (
                "LiquidAI-LFM2-VL-450M",
                f"{model_dir}/decoder_nolm.vmfb",
            )
        if "liquidai-lfm2.5-230m" in lower:
            return (
                "LiquidAI-LFM2.5-230M",
                f"{model_dir}/body.vmfb",
            )
        if "gemma" in lower:
            if self._remote_path_exists(f"{model_dir}/transformer.vmfb"):
                return (
                    "gemma3",
                    f"{model_dir}/transformer.vmfb",
                )
            elif self._remote_path_exists(f"{model_dir}/model.vmfb.trim"):
                return (
                    "gemma3",
                    f"{model_dir}/model.vmfb.trim",
                )
            return (
                "gemma3",
                f"{model_dir}/model.vmfb",
            )
        if "moonshine" in lower and "streaming" in lower:
            return (
                "moonshine_streaming",
                f"{model_dir}/encoder.vmfb",
            )
        if "moonshine" in lower:
            return (
                "moonshine",
                f"{model_dir}/encoder.vmfb",
            )
        if "yolov8-od" in lower or "object" in lower:
            return (
                "object_detection",
                f"{model_dir}/yolo_8n_2.0.0_npu.vmfb",
            )
        if "yolov8-pose" in lower or "pose" in lower:
            return (
                "pose_estimation",
                f"{model_dir}/yolo_pose.vmfb",
            )
        return None

    def _find_profiling_spec_for_demo(self, demo_name):
        model_dirs = self._discover_model_dirs()
        for model_dir in model_dirs:
            spec = self._profiling_spec_for_model_dir(model_dir)
            if spec is not None and spec[0] == demo_name:
                return spec

        expected_paths = _expected_demo_asset_paths(demo_name)
        for candidate in expected_paths:
            if not self._remote_path_exists(candidate):
                continue
            spec = self._profiling_spec_for_model_dir(candidate)
            if spec is not None and spec[0] == demo_name:
                return spec
        return None

    @pytest.mark.parametrize("demo_name", _load_demo_names(), ids=lambda name: str(name))
    def test_demo_profiling_on_dut(self, demo_name):
        """[Metrics Tracking Only] Run non-blocking profiling on successfully prepared VMFB models."""
        self._skip_if_setup_failed(demo_name)
        spec = self._find_profiling_spec_for_demo(demo_name)
        if spec is None:
            pytest.skip(f"No profiling model found for demo '{demo_name}' on the DUT")

        resolved_demo_name, vmfb_path = spec
        self.logger.info("Running NPU model profile for '%s' using model: %s", resolved_demo_name, vmfb_path)
        
        try:
            # Run profile.py via run_demo_profile.sh helper (repeat=3, non-warmup=False)
            stdout, stderr = self._run_dut_script("run_demo_profile.sh", vmfb_path, "--repeat", "3")
            
            # Print performance outputs clearly in metrics formats
            print(f"\n[METRICS] --- {resolved_demo_name} Profile Summary ---", flush=True)
            print(stdout, flush=True)
            print(f"[METRICS] ---------------------------------------------\n", flush=True)
            
            # Extract standard latency line from results summary if present
            for line in (stdout or "").splitlines():
                if "median" in line.lower() or "mean" in line.lower() or "latency" in line.lower() or "time" in line.lower():
                    self.logger.info("[METRICS] %s - %s", resolved_demo_name, line.strip())
        except Exception as exc:
            # Metrics tracking only: log warn but do NOT fail the test suite
            warn_msg = f"[WARN-METRICS] Failed to profile model for demo '{resolved_demo_name}': {exc}"
            self.logger.warning(warn_msg)
            print(warn_msg, flush=True)

