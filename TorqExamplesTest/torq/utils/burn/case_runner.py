"""Run the standalone Synaptics U-Boot eMMC flashing case."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parent
PYTHON_ROOT = CASE_ROOT.parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from torq.utils.burn.device_flash import BOARD_BUILD_VERSION_FILENAME
from torq.utils.burn.runner_common import (
    add_common_arguments,
    build_process_environment,
    resolve_section_values,
    run_case,
    run_pytest,
    section_junitxml,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        '--use-pdu-ac',
        action='store_true',
        help='Use the PDU for AC power cycling',
    )
    parser.add_argument(
        '--force-flash',
        action='store_true',
        help='Flash even if the running U-Boot version is not older than BUILD_VERSION',
    )
    return parser


def _run_section(args, section, tftppath, has_multiple_sections):
    required_keys = ('serial_port', 'netbooter_ip', 'outlet')
    environment = resolve_section_values(
        args,
        section,
        case_name='SYNA_Uboot',
        required_keys=required_keys,
    )
    if environment is None:
        return 2, None

    log_folder = CASE_ROOT / 'log' / section
    process_environment = build_process_environment(
        environment,
        extra_variables={
            'TFTPPATH': tftppath,
            'TFTPIP': tftppath.split(':', 1)[0],
            'USE_PDU_AC': '1' if args.use_pdu_ac else '0',
            'FORCE_FLASH': '1' if args.force_flash else '0',
            'SYNA_LOG_FOLDER': str(log_folder),
        },
        pythonpath_entries=(PYTHON_ROOT,),
    )
    report = section_junitxml(
        args.junitxml or 'TEST-SYNA_Uboot.{}.xml'.format(section),
        section,
        bool(args.junitxml and has_multiple_sections),
    )
    status = run_pytest(
        test_file=CASE_ROOT / 'SYNA_UbootTest.py',
        junitxml=report,
        work_dir=CASE_ROOT,
        environment=process_environment,
        section=section,
        log_prefix='upgrade-syna',
    )
    board_version_file = log_folder / BOARD_BUILD_VERSION_FILENAME
    board_version = None
    if status == 0 and board_version_file.is_file():
        board_version = board_version_file.read_text(encoding='utf-8').strip()
        board_version_file.unlink()
    return status, board_version

def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return run_case(
        args,
        case_name='SYNA_Uboot',
        output_path=CASE_ROOT / 'platforminfo.json',
        section_runner=_run_section,
    )


if __name__ == '__main__':
    raise SystemExit(main())
