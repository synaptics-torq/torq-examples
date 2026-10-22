"""Run the standalone Synaptics U-Boot eMMC flashing case.

Usage:
    python case_runner.py --config CONFIG [--section SECTION] [--junitxml REPORT] [--image-url URL]

- --config: path to an INI file whose sections each describe one board
  (see device_flash.resolve_values for required/optional keys, e.g. serial_port).
- --section: name of a single section to run; if omitted, every section in
  the config file is run in turn.
- --junitxml: path for the pytest JUnit XML report; defaults to
  TEST-SYNA_Uboot.<section>.xml in CASE_ROOT for each section.
- --image-url: URL of the eMMC image to flash; strongly recommended to set
  explicitly (or via the IMAGE_URL environment variable), since the fallback
  derived from BUILD_VERSION/PROFILE/IMAGE_BASE_URL requires all three to be
  set and errors out otherwise (see device_flash.prepare_tftp_image).

For each section, this spawns `pytest SYNA_UbootTest.py` with environment
variables resolved from the config/CLI, streams its output to
upgrade-syna.<section>.log, and emits periodic heartbeat messages while the
flash is in progress. On success for all sections, a platforminfo.json
summary is written to CASE_ROOT. Exit code is non-zero if any section fails.
"""
from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

CASE_ROOT = Path(__file__).resolve().parent
PYTHON_ROOT = CASE_ROOT.parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from torq.utils.burn.device_flash import (
    generate_platform_info,
    list_config_sections,
    prepare_tftp_image,
    resolve_values,
)


SECTION_TIMEOUT_SECONDS = 1200
HEARTBEAT_INTERVAL_SECONDS = 60


def _stream_output(process, log_handle, activity):
    """Mirror subprocess output to both stdout and the log file."""
    assert process.stdout is not None
    for line in process.stdout:
        activity['last_output_at'] = time.monotonic()
        print(line, end='')
        log_handle.write(line)
        log_handle.flush()


def _emit_heartbeat(message, log_handle, with_timestamp=False):
    if with_timestamp:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        message = '{} {}'.format(timestamp, message)
    print(message, flush=True)
    log_handle.write(message + '\n')
    log_handle.flush()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--section')
    parser.add_argument('--junitxml')
    parser.add_argument('--image-url', default=os.environ.get('IMAGE_URL'))
    return parser


def run_section(args, section, tftppath):
    section_args = copy(args)
    section_args.section = section
    required_keys = ('serial_port',)
    try:
        environment = resolve_values(
            section_args,
            required_keys=required_keys,
        )
    except ValueError as error:
        print('SYNA_Uboot [{}]: {}'.format(section, error), file=sys.stderr)
        return 2

    environment['TFTPPATH'] = tftppath
    environment['TFTPIP'] = tftppath.split(':', 1)[0]
    process_environment = os.environ.copy()
    process_environment.update(environment)
    process_environment['SYNA_LOG_FOLDER'] = str(CASE_ROOT / 'log' / section)
    report = args.junitxml or 'TEST-SYNA_Uboot.{}.xml'.format(section)
    log_path = CASE_ROOT / 'upgrade-syna.{}.log'.format(section)
    command = [
        sys.executable, '-m', 'pytest', '--capture=no', '--show-capture=no',
        '-o', 'addopts=',
        '--rootdir={}'.format(CASE_ROOT),
        '--junitxml={}'.format(report),
        str(CASE_ROOT / 'SYNA_UbootTest.py'),
    ]
    process = subprocess.Popen(
        command,
        cwd=CASE_ROOT,
        env=process_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    timer = threading.Timer(SECTION_TIMEOUT_SECONDS, process.kill)
    timer.start()
    with log_path.open('w', encoding='utf-8') as log_file:
        activity = {'last_output_at': time.monotonic()}
        stdout_thread = threading.Thread(
            target=_stream_output,
            args=(process, log_file, activity),
            daemon=True,
        )
        stdout_thread.start()
        last_heartbeat_at = activity['last_output_at']
        while process.poll() is None:
            now = time.monotonic()
            if (
                now - activity['last_output_at'] >= HEARTBEAT_INTERVAL_SECONDS
                and now - last_heartbeat_at >= HEARTBEAT_INTERVAL_SECONDS
            ):
                _emit_heartbeat(
                    '[{}] Command still running...'.format(section),
                    log_file,
                    with_timestamp=True,
                )
                last_heartbeat_at = now
            time.sleep(min(1, HEARTBEAT_INTERVAL_SECONDS))
        stdout_thread.join(timeout=5)
    timer.cancel()
    return process.wait()


def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        sections = [args.section] if args.section else list_config_sections(args.config)
        if not sections:
            raise ValueError('Configuration file does not contain any sections')
        tftppath = prepare_tftp_image(args.image_url)
    except ValueError as error:
        print('SYNA_Uboot: {}'.format(error), file=sys.stderr)
        return 2
    print('Configured sections: {}'.format(', '.join(sections)), flush=True)
    overall_status = 0
    for section in sections:
        print('Starting actual flash for {}'.format(section), flush=True)
        if run_section(args, section, tftppath) != 0:
            overall_status = 1
    if overall_status == 0:
        try:
            generate_platform_info(
                args,
                output_path=CASE_ROOT / 'platforminfo.json',
                sections=sections,
                tftppath=tftppath,
            )
        except ValueError as error:
            print('SYNA_Uboot: {}'.format(error), file=sys.stderr)
            return 2
    return overall_status


if __name__ == '__main__':
    raise SystemExit(main())
