"""Host-side demo setup runner (container based).

Part of the host-driven TorqExamplesTest reconstruction: demo model
preparation (setup_demos.py: fetch + convert/validate) moves from the DUT
board to the CI host. The host (Debian 11, glibc 2.31) cannot run the
compiler wheel natively, so all preparation executes inside a container
with a compatible base image (ubuntu:24.04, glibc 2.39).

Configuration (environment variables):
    TORQ_HOST_IMAGE      container image tag
                         (default: torq-examples-host-setup:local)
    TORQ_MODEL_MODE      "release" (default): download the configured release
                         models from HF.
                         "dev": download pre-compiled models from HF using
                         TORQ_MODEL_VERSION; no compiler is installed.
    TORQ_MODEL_VERSION   model release to use in dev mode (required).

The examples checkout is mounted at /examples (writable) so model
artifacts land in <checkout>/models/<org>/<repo-id>/, exactly mirroring
the legacy DUT layout that the push-to-board step expects. The container
runs with the invoking host user's UID/GID so checkout files remain writable
by the CI runner.
"""

from __future__ import annotations

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
        self.model_version = os.getenv("TORQ_MODEL_VERSION", "").strip()
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

    @staticmethod
    def _host_identity() -> tuple[int, int]:
        if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
            raise HostSetupError("Host model preparation requires a POSIX host UID/GID")
        return os.getuid(), os.getgid()

    def ensure_image(self) -> None:
        context = Path(__file__).resolve().parent / "conversion"
        uid, gid = self._host_identity()
        print(f"[host_setup] building image {self.image} from {context}")
        self._docker(
            "build", "-f", str(context / "Dockerfile"), "-t", self.image,
            "--build-arg", f"HOST_UID={uid}", "--build-arg", f"HOST_GID={gid}",
            str(context),
        )

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
        uid, gid = self._host_identity()
        args = [
            "run", "-d", "--name", self.container, "--net=host",
            "--user", f"{uid}:{gid}",
            "-v", f"{self.examples_root}:/examples",
        ]
        # Mirror host DNS: the docker daemon may impose its own nameserver
        # that resolves egress hosts differently (or to unreachable
        # addresses) than the host resolver.
        for server in self._host_dns_servers():
            args += ["--dns", server]
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
        # A login shell can replace HOME with a base-image user's home for this UID.
        proc = self._docker("exec", self.container, "bash", "-c", script, check=False)
        return SetupResult(proc.returncode, proc.stdout, proc.stderr)

    # -------------------------------------------------------- environment
    def setup_environment(self) -> SetupResult:
        """Create the in-container venv and install what the model mode needs.

        Runs in the container (both modes).

        Both modes download pre-compiled models and do not install a compiler.
        A pinned model version (TORQ_MODEL_VERSION) is applied by setup_demo()
        in both modes.
        """
        compiler_message = (
            "compiler: not installed (dev mode: HF model download)"
            if self.model_mode == "dev"
            else "compiler: not installed (release mode)"
        )
        script = f"""
            set -euo pipefail
            if [ -f /host_ca/system.crt ]; then
                cat /host_ca/system.crt /host_ca/custom/* 2>/dev/null > /work/cabundle.pem                     || cp /host_ca/system.crt /work/cabundle.pem
                printf 'export REQUESTS_CA_BUNDLE=/work/cabundle.pem\\nexport SSL_CERT_FILE=/work/cabundle.pem\\nexport CURL_CA_BUNDLE=/work/cabundle.pem\\n' > /work/torq-ca.env
                echo "host CAs merged into /work/cabundle.pem"
            fi
            python3.12 -m venv /work/venv
            . /work/venv/bin/activate
            [ -f /work/torq-ca.env ] && . /work/torq-ca.env || true
            echo '{compiler_message}'
            cd /examples
            pip install --no-cache-dir -r requirements.txt
            pip install --no-cache-dir -q hf_xet || echo 'hf_xet unavailable, using regular HTTP downloads'
            if [ -f /work/cabundle.pem ]; then
                python3.12 -c 'import ssl, urllib.request; print("HF TLS check:", urllib.request.urlopen("https://huggingface.co", timeout=20, context=ssl.create_default_context(cafile="/work/cabundle.pem")).status)'                     || echo 'WARNING: HF TLS check failed with merged CAs'
            fi
            echo 'host environment ready'
        """
        result = self._exec(script)
        self._write_log("host_setup_environment", result)

        if not result.passed:
            raise HostSetupError(
                f"host environment setup failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
            )
        return result

    # ------------------------------------------------------------ demos
    def setup_demo(
        self, demo: str, app_dir: str = "", expected_model_dir: str = "",
        *, model_name: str | None = None,
    ) -> SetupResult:
        """Run setup_demos.py for one demo inside the container.

        app_dir: demo app directory (relative to the examples root); its
            requirements.txt is installed first when present.
        expected_model_dir: when given, the model directory
            <root>/models/<expected_model_dir> must exist non-empty afterwards.
        model_name: select one model via the demo's setup_demo.py CLI instead
            of preparing all default models via setup_demos.py.
        """
        if self.model_mode == "dev" and not self.model_version:
            raise HostSetupError(
                "dev mode requires TORQ_MODEL_VERSION to pin the HF model release"
            )

        demo_req = "echo 'no per-demo requirements'"
        if app_dir:
            app_req = self.examples_root / app_dir / "requirements.txt"
            if app_req.exists():
                demo_req = f"pip install --no-cache-dir -r {shlex.quote(app_dir + '/requirements.txt')}"
        model_version_arg = (
            f" --model-version {shlex.quote(self.model_version)}"
            if self.model_version
            else ""
        )
        setup_command = f"python setup_demos.py {shlex.quote(demo)}{model_version_arg}"
        log_stem = f"{demo}_host_setup"
        if model_name is not None:
            if not app_dir:
                raise HostSetupError("per-model setup requires the demo app directory")
            setup_command = (
                "python -c 'from setup_demos import install; install()' && "
                f"python {shlex.quote(app_dir + '/setup_demo.py')} "
                f"{shlex.quote(model_name)}{model_version_arg}"
            )
            log_stem = f"{demo}_{Path(model_name).name}_host_setup"
        script = f"""
            set -uo pipefail
            . /work/venv/bin/activate
            [ -f /work/torq-ca.env ] && . /work/torq-ca.env
            cd /examples
            {demo_req}
            {setup_command}
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
                    . /work/venv/bin/activate
                    [ -f /work/torq-ca.env ] && . /work/torq-ca.env
                    pip install --no-cache-dir -r {shlex.quote(req)}
                    cd /examples
                    {setup_command}
                """
                result = self._exec(retry_script)
        self._write_log(log_stem, result)
        if not result.passed:
            return result
        if expected_model_dir:
            model_dir = self.examples_root / "models" / expected_model_dir
            if not model_dir.is_dir() or not any(model_dir.rglob("*")):
                result.returncode = 1
                result.stderr += f"\n[model check] missing or empty model dir: {model_dir}"
        return result

    # --------------------------------------------------------------- logs
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
        TORQ_MODEL_MODE=dev TORQ_MODEL_VERSION=v2.3.0 python3 host_converter.py <examples_root> [demo ...]

    Runs the container build + environment setup + demo setup directly,
    printing everything. On failure the container is kept alive for manual
    inspection:
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
    print(f"   mode     : {runner.model_mode}")
    print(f"   model version: {runner.model_version or '(examples default)'}")
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
        if failed:
            print(f"\n!! container kept for debugging:")
            print(f"   docker exec -it {runner.container} bash")
            print(f"   (venv: /opt/venv   examples: /examples)")
            print(f"   cleanup later: docker rm -f {runner.container}")
        else:
            runner.stop()
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    sys.exit(_standalone_main(sys.argv[1:]))
