"""Run the standalone Synaptics U-Boot eMMC flashing case."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CASE_ROOT.parent
PYTHON_ROOT = CASE_ROOT.parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

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
        return 2

    process_environment = build_process_environment(
        environment,
        extra_variables={
            'TFTPPATH': tftppath,
            'TFTPIP': tftppath.split(':', 1)[0],
            'USE_PDU_AC': '1' if args.use_pdu_ac else '0',
            'SYNA_LOG_FOLDER': str(CASE_ROOT / 'log' / section),
        },
        pythonpath_entries=(PYTHON_ROOT,),
    )
    report = section_junitxml(
        args.junitxml or 'TEST-SYNA_Uboot.{}.xml'.format(section),
        section,
        bool(args.junitxml and has_multiple_sections),
    )
    return run_pytest(
        test_file=CASE_ROOT / 'SYNA_UbootTest.py',
        junitxml=report,
        work_dir=REPOSITORY_ROOT,
        environment=process_environment,
        section=section,
        log_prefix='upgrade-syna',
    )

def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return run_case(
        args,
        case_name='SYNA_Uboot',
        output_path=REPOSITORY_ROOT / 'platforminfo.json',
        section_runner=_run_section,
    )


if __name__ == '__main__':
    raise SystemExit(main())
