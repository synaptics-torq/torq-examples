"""Programmatic API for flashing configured SYNA boards."""
import argparse
import os
import subprocess
import time
from pathlib import Path
import sys

from .device_flash import (
    BOARD_BUILD_VERSION_FILENAME,
    _parse_image_url,
    generate_platform_info,
    prepare_tftp_image,
    resolve_tftp_path,
    resolve_values,
)
from .runner_common import (
    RunnerStatus,
    resolve_section_values,
    resolve_sections,
)


def netbooter_ac_cycle(netbooter_ip, outlet, delay=0):
    """Power cycle a device via NetBooter PDU.

    Args:
        netbooter_ip: NetBooter IP address
        outlet: NetBooter outlet number
        delay: Additional delay in seconds after power-on (default: 0)
    """
    if not netbooter_ip or not outlet:
        raise ValueError('netbooter_ip and outlet are required')
    command = ['netbooter', '-s', str(netbooter_ip), '-o', str(outlet), '-c']
    subprocess.run(command + ['OFF'], check=True)
    time.sleep(20)
    subprocess.run(command + ['ON'], check=True)
    time.sleep(20)
    if delay > 0:
        time.sleep(delay)


def _environment_flag(name, default=False):
    """Parse environment variable as boolean flag.
    Supports: 'true', 'True', 'TRUE', '1' as True
    Otherwise returns the default value.
    """
    value = os.environ.get(name, '').strip().lower()
    if value in {'1', 'true'}:
        return True
    elif value in {'0', 'false'}:
        return False
    else:
        return default


class BurnTool:
    """Flash one or more boards using the SYNA U-Boot TFTP flow."""

    def __init__(
        self,
        config=None,
        build_version=None,
        profile=None,
        image_url=None,
        board=None,
        section=None,
        force_flash=None,
        use_pdu_ac=None,
        image_base_url=None,
        output_dir=None,
        tftp_root=None,
    ):
        self.config = Path(
            config or os.environ.get('CONFIG_PATH') or Path.home() / 'cases' / 'env_config.ini'
        )
        self.build_version = build_version or os.environ.get('BUILD_VERSION')
        self.profile = (
            profile
            or os.environ.get('PROFILE')
            or 'sl2619_oobe_scarthgap'
        )
        self.image_url = image_url or os.environ.get('IMAGE_URL')
        self.board = board if board is not None else os.environ.get('BOARD') or None
        self.section = section if section is not None else os.environ.get('SECTION') or None
        self.force_flash = _environment_flag('FORCE_FLASH') or force_flash
        self.use_pdu_ac = _environment_flag('USE_PDU_AC', default=True) if use_pdu_ac is None else use_pdu_ac
        self.image_base_url = image_base_url or os.environ.get('IMAGE_BASE_URL')
        self.output_dir = Path(
            output_dir or os.environ.get('OUTPUT_DIR') or Path.cwd()
        ).expanduser().resolve()
        self.tftp_root = tftp_root or os.environ.get('TFTP_ROOT')

    def _has_no_serial_output(self, output_log):
        if not output_log.is_file():
            return False
        output = output_log.read_text(encoding='utf-8', errors='replace')
        return (
            "No recognized boot prompt; serial output: ''" in output
            and 'Serial has no output' in output
        )

    def run(self) -> int:
        """Run the flash flow and return its process exit code."""
        try:
            from .device_session import BurnSession

            requested_board = None if self.section else self.board
            sections = resolve_sections(self.config, self.section, requested_board)
            if not self.build_version:
                version_file = (
                    Path(__file__).resolve().parents[4]
                    / 'scripts'
                    / 'astra_build_version.txt'
                )
                if version_file.is_file():
                    self.build_version = version_file.read_text(
                        encoding='utf-8'
                    ).strip()
            if not self.build_version or not self.profile:
                config_args = argparse.Namespace(
                    config=self.config,
                    section=sections[0],
                    build_version=self.build_version,
                    profile=self.profile,
                )
                values = resolve_values(
                    config_args,
                    config_keys=('build_version', 'profile'),
                )
                self.build_version = values.get('BUILD_VERSION')
                self.profile = values.get('PROFILE')

            image_build_version, image_profile = _parse_image_url(self.image_url)
            build_version = image_build_version or self.build_version
            profile = image_profile or self.profile
            tftppath = resolve_tftp_path(
                self.image_url,
                build_version=build_version,
                profile=profile,
                update_environment=False,
            )
        except ImportError as error:
            print(
                "Burning requires pyserial. Install torq-compiler with the 'burn' extra.",
                file=sys.stderr,
            )
            print(str(error), file=sys.stderr)
            return int(RunnerStatus.CONFIG_ERROR)
        except ValueError as error:
            print(f'SYNA_Uboot: {error}', file=sys.stderr)
            return int(RunnerStatus.CONFIG_ERROR)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_folder = self.output_dir / 'log'
        platform_info_json = self.output_dir / 'platforminfo.json'
        automatic_board_selection = not self.board and not self.section
        platform_sections = sections
        overall_status = 0
        board_versions = {}
        run_args = argparse.Namespace(
            config=self.config,
            section=self.section,
            board=requested_board,
            image_url=self.image_url,
        )

        for section in sections:
            print(f'Starting actual flash for {section}', flush=True)
            environment = resolve_section_values(
                run_args,
                section,
                case_name='SYNA_Uboot',
                required_keys=('serial_port', 'netbooter_ip', 'outlet'),
            )
            if environment is None:
                return int(RunnerStatus.CONFIG_ERROR)

            section_log_folder = log_folder / section
            output_log_path = self.output_dir / f'upgrade-syna.{section}.log'
            session = None
            status = 0
            try:
                session = BurnSession(
                    serial_port=environment['SERIAL_PORT'],
                    netbooter_ip=environment['NETBOOTER_IP'],
                    outlet=environment['OUTLET'],
                    build_version=build_version,
                    force_flash=self.force_flash,
                    use_pdu_ac=self.use_pdu_ac,
                    log_folder=section_log_folder,
                    section=section,
                    output_log=output_log_path,
                )
                session.run(
                    tftppath,
                    image_preparer=lambda: prepare_tftp_image(
                        self.image_url,
                        build_version=build_version,
                        profile=profile,
                        image_base_url=self.image_base_url,
                        tftp_root=self.tftp_root,
                        update_environment=False,
                    ),
                )
            except Exception as error:
                print(f'SYNA_Uboot [{section}]: {error}', file=sys.stderr)
                status = (
                    RunnerStatus.NO_BOARD_RESPONSE
                    if self._has_no_serial_output(output_log_path)
                    else 1
                )
            finally:
                if session is not None:
                    session.close()

            if status == RunnerStatus.NO_BOARD_RESPONSE:
                if automatic_board_selection:
                    overall_status = 1
                    print(
                        f'No serial response from {section}; trying the next board',
                        flush=True,
                    )
                    continue
                status = 1

            if status == 0:
                board_version_file = section_log_folder / BOARD_BUILD_VERSION_FILENAME
                if board_version_file.is_file():
                    board_versions[section] = board_version_file.read_text(
                        encoding='utf-8'
                    ).strip()
                    board_version_file.unlink()
            else:
                overall_status = 1
                if automatic_board_selection:
                    return int(status)
                continue

            if automatic_board_selection:
                overall_status = 0
                platform_sections = [section]
                break

        if overall_status == 0:
            try:
                generate_platform_info(
                    run_args,
                    output_path=platform_info_json,
                    sections=platform_sections,
                    tftppath=tftppath,
                    board_versions=board_versions,
                    build_version=build_version,
                    profile=profile,
                )
            except ValueError as error:
                print(f'SYNA_Uboot: {error}', file=sys.stderr)
                return int(RunnerStatus.CONFIG_ERROR)
        return overall_status
