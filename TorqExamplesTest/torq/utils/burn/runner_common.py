"""Shared command-line and subprocess helpers for device runners."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from torq.utils.burn.device_flash import (
    generate_platform_info,
    list_config_sections,
    prepare_tftp_image,
    resolve_values,
)


HEARTBEAT_INTERVAL_SECONDS = 60


def add_common_arguments(parser):
    """Add the command-line options shared by device runners."""
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--section')
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


def resolve_sections(config_path, section):
    """Return the requested section or all sections defined in a config file.

    Raises:
      - ValueError: If the config file is missing or defines no sections.
    """
    config_path = Path(config_path).expanduser()
    if not config_path.is_file():
        raise ValueError(
            'Configuration file does not exist: {}'.format(config_path.resolve())
        )
    if section is not None:
        return [section]
    sections = list_config_sections(config_path)
    if not sections:
        raise ValueError('Configuration file does not contain any sections')
    return sections


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
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    timer = None
    if timeout is not None:
        timer = threading.Timer(timeout, process.kill)
        timer.start()

    activity = {'last_output_at': time.monotonic()}
    stdout_thread = threading.Thread(
        target=_stream_output,
        args=(process, activity, log_handle),
        daemon=True,
    )
    stdout_thread.start()
    last_heartbeat_at = activity['last_output_at']
    try:
        while process.poll() is None:
            now = time.monotonic()
            if (
                now - activity['last_output_at'] >= HEARTBEAT_INTERVAL_SECONDS
                and now - last_heartbeat_at >= HEARTBEAT_INTERVAL_SECONDS
            ):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                message = '{} [{}] Command still running...'.format(timestamp, section)
                print(message, flush=True)
                if log_handle is not None:
                    log_handle.write(message + '\n')
                    log_handle.flush()
                last_heartbeat_at = now
            time.sleep(1)
        stdout_thread.join(timeout=5)
        return process.wait()
    finally:
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

    Returns 0 on success.
    Returns 1 when a section fails.
    Returns 2 for configuration or platform-info errors.
    """
    try:
        sections = resolve_sections(args.config, args.section)
        tftppath = prepare_tftp_image(args.image_url)
    except ValueError as error:
        print('{}: {}'.format(case_name, error), file=sys.stderr)
        return 2

    print('Configured sections: {}'.format(', '.join(sections)), flush=True)
    overall_status = 0
    for section in sections:
        print('Starting actual flash for {}'.format(section), flush=True)
        status = section_runner(args, section, tftppath, len(sections) > 1)
        if status != 0:
            overall_status = 1
            if not continue_on_failure:
                return status

    if overall_status == 0:
        try:
            generate_platform_info(
                args,
                output_path=output_path,
                sections=sections,
                tftppath=tftppath,
            )
        except ValueError as error:
            print('{}: {}'.format(case_name, error), file=sys.stderr)
            return 2
    return overall_status
