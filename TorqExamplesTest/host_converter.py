"""Host-side demo setup runner (container based).

Part of the host-driven TorqExamplesTest reconstruction: demo model
preparation (setup_demos.py: fetch + convert/validate) moves from the DUT
board to the CI host. The host (Debian 11, glibc 2.31) cannot run the
compiler wheel natively, so all preparation executes inside a container
with a compatible base image (ubuntu:24.04, glibc 2.39).

Configuration (environment variables):
    TORQ_HOST_IMAGE             container image tag
                                (default: torq-examples-host-setup:local)
    TORQ_HOST_COMPILER_WHEEL    host path to a torq-compiler wheel to install
                                (mounted read-only into the container)
    TORQ_HOST_COMPILER_VERSION  pip-install this version instead
                                (e.g. 2.3.0); at most one of WHEEL / VERSION
    TORQ_HOST_REBUILD           "1" to force-rebuild the image
    TORQ_MODEL_MODE             "release" (default): models are pre-compiled
                                and published on HF, no compiler required.
                                "dev": build from source with the PR compiler
                                - not implemented yet (fails explicitly).

The examples checkout is mounted at /examples (writable) so model
artifacts land in <checkout>/models/<org>/<repo-id>/, exactly mirroring
the legacy DUT layout that the push-to-board step expects.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class HostSetupError(RuntimeError):
    pass


# setup_demos.py reports missing per-demo packages with the exact
# requirements path to install (app dirs can be nested, e.g.
# LiquidAI/LiquidAI-LFM2.5), so we can self-heal instead of guessing.
_MISSING_PACKAGES_RE = re.compile(r"pip install -r (\S+\S*)")


@dataclass
class SetupResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def passed(self) -> bool:
        return self.returncode == 0


class HostSetupRunner:
    """Orchestrates one long-lived container for all host-side demo setup."""

    def __init__(self, examples_root: Path, log_dir: Path | None = None):
        self.examples_root = Path(examples_root).resolve()
        self.log_dir = Path(log_dir) if log_dir else self.examples_root / "logs"
        self.image = os.getenv("TORQ_HOST_IMAGE", "torq-examples-host-setup:local")
        self.model_mode = os.getenv("TORQ_MODEL_MODE", "release").strip().lower()
        self.wheel = os.getenv("TORQ_HOST_COMPILER_WHEEL", "").strip()
        self.wheel_version = os.getenv("TORQ_HOST_COMPILER_VERSION", "").strip()
        if self.model_mode == "dev":
            if self.wheel and self.wheel_version:
                print(
                    "[host_setup] WARNING: both TORQ_HOST_COMPILER_WHEEL and "
                    "TORQ_HOST_COMPILER_VERSION are set - using the WHEEL path "
                    f"({self.wheel}), ignoring version {self.wheel_version}"
                )
        elif self.wheel or self.wheel_version:
            print(
                "[host_setup] NOTE: TORQ_HOST_COMPILER_WHEEL/_VERSION are set but "
                "IGNORED in release mode - the compiler is never installed "
                "(models are pre-compiled at release time)"
            )
        self.container = f"torq-host-setup-{os.getpid()}-{int(time.time())}"
        self._started = False

    # ------------------------------------------------------------- docker
    def _docker(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = ["docker", *[str(a) for a in args]]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        except FileNotFoundError as exc:
            raise HostSetupError("docker CLI not found on host") from exc
        if check and proc.returncode != 0:
            raise HostSetupError(
                f"command failed: {' '.join(shlex.quote(str(a)) for a in cmd)}\n"
                f"stderr:\n{proc.stderr}"
            )
        return proc

    def docker_available(self) -> bool:
        return self._docker("version", check=False).returncode == 0

    def ensure_image(self) -> None:
        if (
            os.getenv("TORQ_HOST_REBUILD", "") != "1"
            and self._docker("image", "inspect", self.image, check=False).returncode == 0
        ):
            print(f"[host_setup] reusing existing image {self.image}")
            return
        context = Path(__file__).resolve().parent / "conversion"
        print(f"[host_setup] building image {self.image} from {context}")
        self._docker("build", "-f", str(context / "Dockerfile"), "-t", self.image, str(context))

    @staticmethod
    def _host_dns_servers() -> list[str]:
        """Nameservers from the host /etc/resolv.conf (max 3, docker limit)."""
        try:
            lines = Path("/etc/resolv.conf").read_text().splitlines()
        except OSError:
            return []
        servers = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                servers.append(parts[1])
            if len(servers) >= 3:
                break
        return servers

    def start(self) -> None:
        if self._started:
            return
        if os.getenv("TORQ_HOST_REUSE", "") == "1":
            # Debug convenience: reuse an idle container of the same image
            # bound to the SAME examples checkout (environment setup is
            # idempotent, so the venv converges to the requested state).
            for cid in self._docker(
                "ps", "-q", "--filter", f"ancestor={self.image}", check=False
            ).stdout.split():
                name = self._docker(
                    "inspect", "--format", "{{.Name}}", cid
                ).stdout.strip().lstrip("/")
                if not name.startswith("torq-host-setup-"):
                    continue
                mount_src = self._docker(
                    "inspect",
                    "--format",
                    "{{range .Mounts}}{{if eq .Destination \"/examples\">{{.Source}}{{end}}{{end}}",
                    cid,
                ).stdout.strip()
                if mount_src != str(self.examples_root):
                    continue
                self.container = name
                self._started = True
                print(f"[host_setup] reusing existing container {name}")
                return
        args = [
            "run", "-d", "--name", self.container, "--net=host",
            "-v", f"{self.examples_root}:/examples",
        ]
        # Mirror host DNS: the docker daemon may impose its own nameserver
        # that resolves egress hosts differently (or to unreachable
        # addresses) than the host resolver.
        for server in self._host_dns_servers():
            args += ["--dns", server]
        if self.wheel:
            wheel_dir = Path(self.wheel).resolve().parent
            args += ["-v", f"{wheel_dir}:/host_wheels:ro"]
        # Mount the host CA material: the lab egress performs TLS interception,
        # and the corporate root CA lives in the host trust store but not in
        # the container's (fresh ubuntu store / certifi).
        for host_path, container_path in (
            ("/etc/ssl/certs/ca-certificates.crt", "/host_ca/system.crt"),
            ("/usr/local/share/ca-certificates", "/host_ca/custom"),
        ):
            if os.path.exists(host_path):
                args += ["-v", f"{host_path}:{container_path}:ro"]
        # Forward proxy / HF settings: --net=host shares the network stack but
        # not environment variables, so the container would be "naked".
        for var in (
            "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
            "NO_PROXY", "no_proxy", "HF_ENDPOINT", "HF_TOKEN",
        ):
            value = os.getenv(var, "").strip()
            if value:
                args += ["-e", f"{var}={value}"]
        args += [self.image, "sleep", "infinity"]
        self._docker(*args)
        self._started = True
        print(f"[host_setup] container {self.container} started (image {self.image})")

    def stop(self) -> None:
        if self._started:
            self._docker("rm", "-f", self.container, check=False)
            self._started = False

    def _exec(self, script: str) -> SetupResult:
        proc = self._docker("exec", self.container, "bash", "-lc", script, check=False)
        return SetupResult(proc.returncode, proc.stdout, proc.stderr)

    # -------------------------------------------------------- environment
    def setup_environment(self) -> SetupResult:
        """Create the in-container venv and install what the model mode needs.

        Runs in the container (both modes).

        release (default): models are pre-compiled and published on HF; the
            compiler is NEVER installed (any TORQ_HOST_COMPILER_* input is
            ignored with a note) so nothing can pollute this run's report.
        dev: the PR compiler is force-installed from TORQ_HOST_COMPILER_*
            and its version verified; from-source model building (the
            per-demo part) is not implemented yet.
        """
        compiler_check = (
            '''python3.12 -c "import torq.compiler; print('torq.compiler import OK')"
torq-compile --version 2>&1 | head -2 || true
pip show torq-compiler 2>/dev/null | grep -E '^Version' || true'''
        )
        if self.model_mode == "dev":
            # dev mode: the PR compiler is required - force-install from the
            # env input whatever the container previously had; the installed
            # version is verified below.
            if not (self.wheel or self.wheel_version):
                raise HostSetupError(
                    "dev mode requires TORQ_HOST_COMPILER_WHEEL or "
                    "TORQ_HOST_COMPILER_VERSION (the compiler under test)"
                )
            if self.wheel:
                wheel_install = (
                    f"pip install --no-cache-dir --force-reinstall --no-deps "
                    f"/host_wheels/{Path(self.wheel).name}"
                )
            else:
                wheel_install = (
                    f"pip install --no-cache-dir --force-reinstall --no-deps "
                    f"torq-compiler=={self.wheel_version}"
                )
        else:
            # release mode: the compiler is NEVER installed - models are
            # pre-compiled at release time; anything previously present in
            # the container (other people/tests) must not affect this run.
            wheel_install = "echo 'release mode: compiler not installed (models are pre-compiled)'"
            compiler_check = "echo 'compiler: not installed (release mode)'"
            self.wheel = ""
            self.wheel_version = ""
        script = f"""
            set -euo pipefail
            if [ -f /host_ca/system.crt ]; then
                cat /host_ca/system.crt /host_ca/custom/* 2>/dev/null > /opt/cabundle.pem                     || cp /host_ca/system.crt /opt/cabundle.pem
                printf 'export REQUESTS_CA_BUNDLE=/opt/cabundle.pem\\nexport SSL_CERT_FILE=/opt/cabundle.pem\\nexport CURL_CA_BUNDLE=/opt/cabundle.pem\\n' > /opt/torq-ca.env
                echo "host CAs merged into /opt/cabundle.pem"
            fi
            python3.12 -m venv /opt/venv
            . /opt/venv/bin/activate
            [ -f /opt/torq-ca.env ] && . /opt/torq-ca.env || true
            {wheel_install}
            {compiler_check}
            cd /examples
            pip install --no-cache-dir -r requirements.txt
            pip install --no-cache-dir -q hf_xet || echo 'hf_xet unavailable, using regular HTTP downloads'
            if [ -f /opt/cabundle.pem ]; then
                python3.12 -c 'import ssl, urllib.request; print("HF TLS check:", urllib.request.urlopen("https://huggingface.co", timeout=20, context=ssl.create_default_context(cafile="/opt/cabundle.pem")).status)'                     || echo 'WARNING: HF TLS check failed with merged CAs'
            fi
            echo 'host environment ready'
        """
        result = self._exec(script)
        self._write_log("host_setup_environment", result)

        if not result.passed:
            raise HostSetupError(
                f"host environment setup failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
            )
        if self.model_mode == "dev":
            installed = re.search(r"^Version: (\S+)", result.stdout, re.MULTILINE)
            installed = installed.group(1) if installed else None
            expected = (
                Path(self.wheel).name.split("-")[1] if self.wheel else self.wheel_version
            )
            if installed != expected:
                raise HostSetupError(
                    f"dev mode compiler version mismatch: requested {expected!r}, "
                    f"installed {installed!r}"
                )
            # dev mode: the PR compiler builds the models under test - record
            # the verified version for the report header / JSON export.
            self._capture_compiler_version(result, role="build")
        return result

    # ------------------------------------------------------------ demos
    def setup_demo(
        self, demo: str, app_dir: str = "", expected_model_dir: str = ""
    ) -> SetupResult:
        """Run setup_demos.py for one demo inside the container.

        app_dir: demo app directory (relative to the examples root); its
            requirements.txt is installed first when present.
        expected_model_dir: when given, the model directory
            <root>/models/<expected_model_dir> must exist non-empty afterwards.
        """
        if self.model_mode == "dev":
            raise HostSetupError(
                f"dev mode: from-source model build for demo {demo!r} is not "
                "implemented yet (compiler install/verify is in place)"
            )

        demo_req = "echo 'no per-demo requirements'"
        if app_dir:
            app_req = self.examples_root / app_dir / "requirements.txt"
            if app_req.exists():
                demo_req = f"pip install --no-cache-dir -r {shlex.quote(app_dir + '/requirements.txt')}"
        script = f"""
            set -uo pipefail
            . /opt/venv/bin/activate
            [ -f /opt/torq-ca.env ] && . /opt/torq-ca.env
            cd /examples
            {demo_req}
            python setup_demos.py {shlex.quote(demo)}
        """
        result = self._exec(script)
        if not result.passed:
            req_path = (
                _MISSING_PACKAGES_RE.search(result.stdout + result.stderr) or None
            )
            if req_path:
                req = req_path.group(1)
                print(f"[host_setup] {demo}: missing packages reported; "
                      f"installing {req} and retrying")
                retry_script = f"""
                    set -uo pipefail
                    . /opt/venv/bin/activate
                    [ -f /opt/torq-ca.env ] && . /opt/torq-ca.env
                    pip install --no-cache-dir -r {shlex.quote(req)}
                    cd /examples
                    python setup_demos.py {shlex.quote(demo)}
                """
                result = self._exec(retry_script)
        self._write_log(f"{demo}_host_setup", result)
        if not result.passed:
            return result
        if expected_model_dir:
            model_dir = self.examples_root / "models" / expected_model_dir
            if not model_dir.is_dir() or not any(model_dir.rglob("*")):
                result.returncode = 1
                result.stderr += f"\n[model check] missing or empty model dir: {model_dir}"
        return result

    # --------------------------------------------------------------- logs
    def _capture_compiler_version(self, result: "SetupResult", role: str = "build") -> None:
        """Record the installed compiler version into prepare_metadata.json.
        Only called in dev mode, after version verification."""
        m = re.search(r"^Version: (\S+)", result.stdout, re.MULTILINE)
        if not m:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / "prepare_metadata.json"
            data = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        data = {}
                except (OSError, json.JSONDecodeError):
                    data = {}
            data["torq_compiler_version"] = m.group(1)
            data["torq_compiler_role"] = role
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"[host_setup] compiler version captured: {m.group(1)} (role: {role})")
        except OSError:
            pass

    def _write_log(self, stem: str, result: SetupResult) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"{stem}.log").write_text(
                f"returncode: {result.returncode}\n\n"
                f"----- stdout -----\n{result.stdout}\n"
                f"----- stderr -----\n{result.stderr}\n",
                encoding="utf-8",
            )
        except OSError:
            pass


# --------------------------------------------------------------------------
# Standalone debug driver (no TorqExamplesTest / pytest dependency)
# --------------------------------------------------------------------------

def _print_result(title: str, result: SetupResult) -> None:
    print(f"\n{'=' * 70}")
    print(f"== {title}  (rc={result.returncode})")
    print("=" * 70)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("----- stderr -----")
        print(result.stderr)


def _standalone_main(argv: list[str]) -> int:
    """Usage:
        TORQ_HOST_COMPILER_VERSION=2.3.0 python3 host_converter.py <examples_root> [demo ...]
        TORQ_HOST_COMPILER_WHEEL=/path/torq_compiler-*.whl python3 host_converter.py <examples_root> gemma3

    Runs the container build + environment setup + demo setup directly,
    printing everything. On failure (or with TORQ_HOST_KEEP=1) the container
    is kept alive for manual inspection, e.g. to debug torq-compile by hand:
        docker exec -it <container> bash
    """
    if not argv:
        print(__doc__)
        return 2
    examples_root = Path(argv[0]).resolve()
    demos = argv[1:] or ["gemma3"]
    if not (examples_root / "setup_demos.py").exists():
        print(f"FATAL: {examples_root}/setup_demos.py not found - not a torq-examples checkout?")
        return 1

    runner = HostSetupRunner(examples_root=examples_root)
    print("== standalone host-setup debug ==")
    print(f"   examples : {examples_root}")
    print(f"   image    : {runner.image}")
    print(f"   compiler : wheel={runner.wheel or '-'}  version={runner.wheel_version or '-'}")
    print(f"   demos    : {', '.join(demos)}")

    if not runner.docker_available():
        print("FATAL: docker not available on this host")
        return 1

    failed = False
    try:
        runner.ensure_image()
        runner.start()

        env = runner.setup_environment()
        _print_result("environment", env)
        if not env.passed:
            failed = True
        else:
            for demo in demos:
                app_dir = demo if (examples_root / demo / "requirements.txt").exists() else ""
                res = runner.setup_demo(demo, app_dir)  # no strict model check in debug mode
                _print_result(f"setup_demo {demo}", res)
                if not res.passed:
                    failed = True
    finally:
        if failed or os.getenv("TORQ_HOST_KEEP", "") == "1":
            print(f"\n!! container kept for debugging:")
            print(f"   docker exec -it {runner.container} bash")
            print(f"   (venv: /opt/venv   examples: /examples   wheel already installed)")
            print(f"   cleanup later: docker rm -f {runner.container}")
        else:
            runner.stop()
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    sys.exit(_standalone_main(sys.argv[1:]))
