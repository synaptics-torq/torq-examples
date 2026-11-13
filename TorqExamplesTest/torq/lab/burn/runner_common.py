"""Shared command-line and subprocess helpers for device runners."""
import argparse
import configparser
from datetime import datetime
from enum import IntEnum
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .runner_heartbeat import (
    Heartbeat,
    enable_heartbeat_control,
    parse_suppression_marker,
    make_heartbeat_callback,
    make_activity_condition,
)
from .device_flash import (
    generate_platform_info,
    list_config_sections,
    resolve_values,
    resolve_tftp_path,
)

class RunnerStatus(IntEnum):
    CONFIG_ERROR = 11
    NO_BOARD_RESPONSE = 12


def add_common_arguments(parser):
    """Add the command-line options shared by device runners."""
    parser.add_argument('--config', type=Path, required=True)
    board_group = parser.add_mutually_exclusive_group()
    board_group.add_argument(
        '--board',
        help="INI board section to use, or 'all'; defaults to the first active board",
    )
    board_group.add_argument('--section', help='Legacy explicit INI section selection')
    parser.add_argument('--junitxml')
    parser.add_argument('--image-url', default=os.environ.get('IMAGE_URL'))
    return parser


def section_junitxml(junitxml, section, has_multiple_sections):
    """Return a section-specific JUnit path when multiple sections are run."""
    junitxml_path = Path(junitxml)
    if not has_multiple_sections or section is None:
        return str(junitxml_path)
    return str(
        junitxml_path.with_name(
            '{}.{:s}{}'.format(
                junitxml_path.stem,
                section,
                junitxml_path.suffix,
            )
        )
    )


def resolve_sections(config_path, section=None, board=None):
    """Return requested sections or configured boards with enumerated serial ports.

    Raises:
      - ValueError: If the config file is missing or defines no sections.
    """
    config_path = Path(config_path).expanduser()
    if not config_path.is_file():
        raise ValueError(
            'Configuration file does not exist: {}'.format(config_path.resolve())
        )
    sections = list_config_sections(config_path)
    if not sections:
        raise ValueError('Configuration file does not contain any sections')
    if board and board.strip().lower() == 'all':
        return sections
    requested_board = board or section
    if requested_board:
        return [requested_board]

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)
    from serial.tools.list_ports import comports

    active_ports = {os.path.realpath(port.device) for port in comports()}
    active_sections = []
    for section_name in sections:
        serial_port = next(
            (
                value
                for key, value in config.items(section_name)
                if key.lower() == 'serial_port'
            ),
            None,
        )
        if serial_port and os.path.realpath(serial_port) in active_ports:
            active_sections.append(section_name)
    if active_sections:
        return active_sections
    raise ValueError('No active board found in configuration: {}'.format(config_path))


def resolve_section_values(args, section, case_name,
    required_keys=None, config_keys=None):
    """Resolve environment values for one section, reporting config errors."""
    section_args = argparse.Namespace(**vars(args))
    section_args.section = section
    try:
        return resolve_values(
            section_args,
            required_keys=required_keys,
            config_keys=config_keys,
        )
    except ValueError as error:
        print(
            '{} [{}]: {}'.format(case_name, section or '<direct>', error),
            file=sys.stderr,
        )
        return None


def build_process_environment(environment, extra_variables=None, pythonpath_entries=()):
    """Merge runner values and Python paths into a copy of the current environment."""
    process_environment = os.environ.copy()
    process_environment.update(environment)
    if extra_variables:
        process_environment.update(extra_variables)
    if pythonpath_entries:
        paths = [str(path) for path in pythonpath_entries]
        existing_pythonpath = process_environment.get('PYTHONPATH')
        if existing_pythonpath:
            paths.append(existing_pythonpath)
        process_environment['PYTHONPATH'] = os.pathsep.join(paths)
    return process_environment


def _stream_output(process, activity, log_handle):
    """Forward subprocess output to stdout and an optional log file."""
    assert process.stdout is not None
    for line in process.stdout:
        marker = line.rstrip('\r\n')
        is_heartbeat_marker, heartbeat_suppressed = parse_suppression_marker(marker)
        if is_heartbeat_marker:
            activity['heartbeat_suppressed'] = heartbeat_suppressed
            activity['last_output_at'] = time.monotonic()
            continue
        activity['last_output_at'] = time.monotonic()
        print(line, end='', flush=True)
        if log_handle is not None:
            log_handle.write(line)
            log_handle.flush()


def run_command(command, cwd, environment, section, log_handle=None, timeout=None):
    """Run a command, stream its output, and return its exit code.

    Args:
      - timeout: Optional process timeout in seconds; 
                 the process is killed when the limit is reached.
    """
    process_environment = environment.copy()
    enable_heartbeat_control(process_environment)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=process_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    timer = None
    if timeout is not None:
        timer = threading.Timer(timeout, process.kill)
        timer.start()

    activity = {
        'last_output_at': time.monotonic(),
        'suppressed': False,
    }

    heartbeat = Heartbeat(
        interval=60,
        on_heartbeat=make_heartbeat_callback(log_file=log_handle, label=section),
        should_emit=make_activity_condition(activity)
    )
    stdout_thread = threading.Thread(
        target=_stream_output,
        args=(process, activity, log_handle),
        daemon=True,
    )
    stdout_thread.start()
    heartbeat.start()
    try:
        process.wait()
        stdout_thread.join(timeout=5)
        return process.returncode
    finally:
        heartbeat.stop()
        if timer is not None:
            timer.cancel()


def run_pytest(test_file, junitxml, work_dir, environment, section, log_prefix, timeout=None):
    """Run pytest for a test file and log output under the working directory.

    The pytest root directory is the test file's parent.
    The log filename is ``{log_prefix}.{section}.log`` under the working directory.
    ``timeout`` is an optional limit in seconds.
    """
    test_file = Path(test_file).resolve()
    command = [
        sys.executable,
        '-m',
        'pytest',
        '--capture=no',
        '--show-capture=no',
        '-o',
        'addopts=',
        '--rootdir={}'.format(test_file.parent),
        '--junitxml={}'.format(junitxml),
        str(test_file),
    ]
    log_path = Path(work_dir) / '{}.{}.log'.format(log_prefix, section)
    with log_path.open('w', encoding='utf-8') as log_handle:
        return run_command(
            command,
            cwd=work_dir,
            environment=environment,
            section=section,
            log_handle=log_handle,
            timeout=timeout,
        )


def run_case(args, case_name, output_path, section_runner, continue_on_failure=True):
    """Run configured sections and generate platform info if all pass.

    ``section_runner`` returns ``(status, board_version)``; ``board_version`` is the
    build version read from the board, used instead of BUILD_VERSION when set.

    Returns 0 on success.
    Returns 1 when a section fails.
    Returns RunnerStatus.CONFIG_ERROR for configuration or platform-info errors.
    """
    try:
        sections = resolve_sections(args.config, args.section, args.board)
        tftppath = resolve_tftp_path(args.image_url)
    except ValueError as error:
        print('{}: {}'.format(case_name, error), file=sys.stderr)
        return RunnerStatus.CONFIG_ERROR

    print('Configured sections: {}'.format(', '.join(sections)), flush=True)
    automatic_board_selection = not args.board and not args.section
    platform_sections = sections
    overall_status = 0
    board_versions = {}
    for section in sections:
        print('Starting actual flash for {}'.format(section), flush=True)
        status, board_version = section_runner(args, section, tftppath, len(sections) > 1)
        if status == RunnerStatus.NO_BOARD_RESPONSE:
            if automatic_board_selection:
                overall_status = 1
                print(
                    'No serial response from {}; trying the next board'.format(section),
                    flush=True,
                )
                continue
            status = 1
        if board_version:
            board_versions[section] = board_version
        if status != 0:
            overall_status = 1
            if automatic_board_selection:
                return status
            if not continue_on_failure:
                return status
        elif automatic_board_selection:
            overall_status = 0
            platform_sections = [section]
            break

    if overall_status == 0:
        try:
            generate_platform_info(
                args,
                output_path=output_path,
                sections=platform_sections,
                tftppath=tftppath,
                board_versions=board_versions,
            )
        except ValueError as error:
            print('{}: {}'.format(case_name, error), file=sys.stderr)
            return RunnerStatus.CONFIG_ERROR
    return overall_status
