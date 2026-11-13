#!/usr/bin/env python3
"""Flash an eMMC image using the Synaptics U-Boot tftp2emmc flow."""
import logging
import os
import re
import shutil
import time
from pathlib import Path

from .device_flash import (
    BOARD_BUILD_VERSION_FILENAME,
    UBOOT_VERSION_PATTERN,
    parse_uboot_build_version,
    image_needs_upgrade,
)
from .tool import netbooter_ac_cycle
from .serial_wrapper import SerialThread, SerialTimeoutException
from .runner_heartbeat import Heartbeat, make_heartbeat_callback

UBOOT_PREFIX = '=>'
ANSI_ESCAPE_PATTERN = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|.)')
LINUX_SHELL_PROMPT = re.compile(
    r'(?m)^(?:[\w.-]+@[\w.-]+:[^\r\n]*[#$]|[\w.-]+:[^\r\n]*[#$])\s*$'
)
LINUX_LOGIN_PROMPT = re.compile(r'(?m)^[^\r\n]*login:\s*$')
LINUX_PASSWORD_PROMPT = re.compile(r'(?mi)^[^\r\n]*password:\s*$')
TFTP_SUCCESS = ['tftp2emmc success', 'Burn images OK', 'list2emmc success']
TFTP_FAILURE = [
    'tftp2emmc fail',
    'list2emmc fail',
    'flash write failure',
    'gpt_restore failed',
    'TFTP error:',
]
EMMC_TIMEOUT = 3600
SPI_MODE_ENTER_TIMEOUT = 10
LINUX_LOGIN_TIMEOUT = 30
LINUX_REBOOT_TO_UBOOT_TIMEOUT = 15
LOG_FORMAT = '[%(levelname)s] %(message)s'


class BurnSession:
    """Flash one configured board; no post-flash tests are included."""

    def __init__(
        self,
        serial_port,
        netbooter_ip,
        outlet,
        build_version,
        force_flash,
        use_pdu_ac,
        log_folder,
        section,
        output_log,
    ):
        self.serial_port = serial_port
        self.netbooter_ip = netbooter_ip
        self.outlet = outlet
        self.build_version = build_version
        self.force_flash = force_flash
        self.use_pdu_ac = use_pdu_ac
        self.log_folder = Path(log_folder)
        self.section = section
        self.output_log = Path(output_log)
        if self.log_folder.is_dir():
            shutil.rmtree(self.log_folder)
        self.log_folder.mkdir(parents=True)
        self.output_log.parent.mkdir(parents=True, exist_ok=True)

        # Support LOG_LEVEL environment variable (default: DEBUG)
        # In workflow, this is set to INFO to suppress debug logs like "Clean data and data_d"
        log_level_str = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
        log_level = getattr(logging, log_level_str, logging.DEBUG)

        logging.getLogger().setLevel(log_level)
        formatter = logging.Formatter(LOG_FORMAT)

        serial_logger = logging.getLogger(f'serial.{section}')
        serial_logger.setLevel(log_level)
        serial_logger.propagate = False  # Don't propagate to root logger (prevent stdout output)
        serial_handler = logging.FileHandler(self.log_folder / 'serial_opt.log')
        serial_handler.setFormatter(formatter)
        serial_handler.setLevel(logging.INFO)
        serial_logger.addHandler(serial_handler)

        self.logger = logging.getLogger(f'main.{section}')
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # Don't propagate to root logger (prevent duplicate output)
        self.main_handlers = []
        for path in (self.log_folder / 'main.log', self.output_log):
            main_handler = logging.FileHandler(path)
            main_handler.setFormatter(formatter)
            self.logger.addHandler(main_handler)
            self.main_handlers.append(main_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.main_handlers.append(stream_handler)

        self.suite_logger = logging.getLogger(f'suite.{section}')
        self.suite_logger.setLevel(log_level)
        self.suite_logger.propagate = False  # Don't propagate to root logger (prevent duplicate output)
        suite_handler = logging.FileHandler(self.log_folder / 'suite.log')
        suite_handler.setFormatter(formatter)
        self.suite_logger.addHandler(suite_handler)
        self.suite_handler = suite_handler

        self.serial = SerialThread(self.serial_port, logger=serial_logger)
        self.serial_logger = serial_logger
        self.serial_handler = serial_handler
        self.serial_log = self.serial.create_logger(str(self.log_folder / 'serial.log'))

    def close(self):
        self.serial.close_logger(self.serial_log)
        self.serial.close()
        for logger, handlers in (
            (self.logger, self.main_handlers),
            (self.suite_logger, [self.suite_handler]),
            (self.serial_logger, [self.serial_handler]),
        ):
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()

    def wait_for_serial_output(self, patterns, timeout, input_data=None, repeat_gap=None):
        output_log = self.serial.create_logger()
        deadline = time.monotonic() + timeout
        last_write = None
        try:
            while True:
                raw_data = output_log.readall()
                clean_data = ANSI_ESCAPE_PATTERN.sub('', raw_data)
                if any(
                    pattern.search(clean_data)
                    if hasattr(pattern, 'search')
                    else pattern in clean_data
                    for pattern in patterns
                ):
                    return True, raw_data, clean_data

                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    return False, raw_data, clean_data
                if input_data is not None and (
                    last_write is None
                    or (repeat_gap is not None and now - last_write >= repeat_gap)
                ):
                    self.serial.write(input_data)
                    last_write = now
                time.sleep(min(0.05, remaining))
        finally:
            self.serial.close_logger(output_log)

    def login_linux_root(self):
        self.logger.info('Linux login prompt detected. Trying to login...')
        self.suite_logger.info('[STEP] Login to Linux...')
        matched, raw_data, clean_data = self.wait_for_serial_output(
            [LINUX_SHELL_PROMPT, LINUX_PASSWORD_PROMPT],
            timeout=LINUX_LOGIN_TIMEOUT,
            input_data='root\n',
        )
        if not matched:
            if LINUX_LOGIN_PROMPT.search(clean_data):
                self.logger.error(
                    'Returned to login prompt; serial output: %r',
                    raw_data[-500:],
                )
                raise RuntimeError(
                    'Login failed; Linux login prompt is still present.'
                )
            self.logger.error(
                'Login timed out; serial output: %r', raw_data[-500:]
            )
            raise RuntimeError('Timed out trying to log in.')

        if LINUX_SHELL_PROMPT.search(clean_data):
            self.logger.info('Logged in to Linux shell')
            return
        if LINUX_PASSWORD_PROMPT.search(clean_data):
            self.logger.error(
                'Login requires a password; serial output: %r',
                raw_data[-500:],
            )
            raise RuntimeError('Login requires a password; refusing to continue.')
        self.logger.error(
            'Login was rejected; serial output: %r', raw_data[-500:]
        )
        raise RuntimeError('Login failed; Linux login prompt is still present.')

    def enter_spi_mode(self):
        self.logger.info('Trying to enter SPI mode...')
        matched, raw_data, clean_data = self.wait_for_serial_output(
            [UBOOT_PREFIX, LINUX_SHELL_PROMPT, LINUX_LOGIN_PROMPT],
            timeout=SPI_MODE_ENTER_TIMEOUT,
            input_data='\n',
            repeat_gap=0.1,
        )
        if not matched:
            self.logger.error(
                'No recognized boot prompt; serial output: %r', raw_data[-500:]
            )
            raise RuntimeError('Failed to identify the device prompt.')

        if UBOOT_PREFIX in clean_data:
            self.logger.info('Entered SPI mode')
            return True

        if LINUX_LOGIN_PROMPT.search(clean_data):
            self.login_linux_root()
        elif not LINUX_SHELL_PROMPT.search(clean_data):
            self.logger.error('Unrecognized prompt; serial output: %r', raw_data[-500:])
            raise RuntimeError('Failed to identify the device prompt.')

        self.logger.info('Linux shell detected; rebooting to enter SPI mode')
        self.suite_logger.info('[STEP] Reboot Linux to enter SPI mode')
        self.serial.write('reboot\n')
        try:
            self.serial.expect_for_write(
                '\n',
                UBOOT_PREFIX,
                True,
                repeat_gap=0.1,
                timeout=LINUX_REBOOT_TO_UBOOT_TIMEOUT,
            )
            self.logger.info('Entered SPI mode after reboot')
        except Exception as error:
            self.logger.error(f'Failed to enter spi mode: {error}')
            raise RuntimeError('Failed to enter spi mode.') from error

    def serial_exec_shell(
        self, command, expected, repeat=False, repeat_gap=1, all_output=True,
        timeout=60, is_enter=True,
    ):
        self.logger.info('CMD: %s', command if command else '<ENTER>')
        command = command + '\n' if is_enter else command

        heartbeat = None
        try:
            # For long operations, emit heartbeat logs every 60 seconds to prevent GitHub Actions timeout
            if timeout > 60:
                heartbeat = Heartbeat(
                    interval=60,
                    on_heartbeat=make_heartbeat_callback(logger=self.logger)
                )
                heartbeat.start()

            return self.serial.expects_for_write(
                command, expected, repeat, repeat_gap,
                all_=all_output, timeout=timeout,
            )
        except SerialTimeoutException as error:
            raise RuntimeError(
                f'Timeout, no specified keyword found: {expected}'
            ) from error
        finally:
            if heartbeat is not None:
                heartbeat.stop()

    def wait_spi_dhcp(self):
        console_data = self.serial.create_logger()
        try:
            for _ in range(3):
                try:
                    self.serial_exec_shell('net_init;dhcp', ['bound to address'], timeout=60)
                    return
                except RuntimeError:
                    if re.search(r'ipaddr=\d+\.\d+\.\d+\.\d+', console_data.readall()):
                        return
                    self.logger.warning('Get IP timeout, Ctrl+C')
                    self.serial.write(chr(3))
            raise RuntimeError('Fail to get IP in SPI')
        finally:
            self.serial.close_logger(console_data)

    def read_uboot_build_version(self):
        _, _, version_output = self.serial_exec_shell(
            'version', [UBOOT_VERSION_PATTERN],
            all_output=False, timeout=SPI_MODE_ENTER_TIMEOUT,
        )
        return parse_uboot_build_version(version_output)

    def reset_device(self, use_pdu=False):
        if use_pdu:
            self.suite_logger.info('[STEP] Power cycle via NetBooter')
            self.logger.info('Power cycling...')
            netbooter_ac_cycle(self.netbooter_ip, self.outlet, delay=5)
        else:
            self.logger.info('Resetting...')
            self.serial.write('reset\n')
            # Let the old prompt echo drain so enter_spi_mode only matches the new U-Boot prompt.
            time.sleep(1)

    def verify_flashed_version(self, use_pdu=False):
        self.logger.info('Restarting to verify flashed U-Boot version')
        self.suite_logger.info('[STEP] Verify U-Boot version after flash')
        self.reset_device(use_pdu)
        try:
            self.enter_spi_mode()
        except RuntimeError as error:
            self.logger.warning('Skipping version verification: %s', error)
            return
        try:
            current_version = self.read_uboot_build_version()
        except RuntimeError:
            self.logger.warning('No U-Boot version line found after flash')
            return
        target_version = self.build_version
        # Running U-Boot may not come from eMMC (e.g. SPI boot), so a mismatch is not fatal.
        if current_version != target_version:
            message = f'U-Boot build {current_version} does not match flashed build {target_version}'
            self.logger.warning(message)
            if os.getenv('GITHUB_ACTIONS') == 'true':
                print(f'::warning title=U-Boot version mismatch::{message}', flush=True)
            return
        self.logger.info('U-Boot build %s matches flashed build', current_version)

    def check_image_upgrade(self):
        if self.force_flash:
            self.logger.info('FORCE_FLASH set; skipping U-Boot version check')
            self.reset_device(use_pdu=self.use_pdu_ac)
            return True

        try:
            self.enter_spi_mode()
        except RuntimeError:
            if not self.use_pdu_ac:
                # Without a PDU power cycle, retrying would not recover the device.
                raise
            self.logger.warning('Failed to enter SPI mode; power cycling and retrying')
            self.reset_device(use_pdu=True)
            self.enter_spi_mode()
        try:
            current_version = self.read_uboot_build_version()
        except RuntimeError:
            self.logger.warning('No U-Boot version line found; need flashing')
            return True
        target_version = self.build_version
        if current_version is None:
            self.logger.warning('Unable to parse the current U-Boot version; need flashing')
            return True
        if image_needs_upgrade(current_version, target_version):
            self.logger.info(
                'Current build %s is older than %s; need flashing',
                current_version, target_version,
            )
            return True
        self.logger.info(
            'Current build %s is not older than %s; skipping flash',
            current_version, target_version,
        )
        (self.log_folder / BOARD_BUILD_VERSION_FILENAME).write_text(
            current_version, encoding='utf-8',
        )
        # No upgrade needed; reset device to boot the existing eMMC image.
        self.reset_device(use_pdu=self.use_pdu_ac)
        return False

    def run(self, tftppath=None, image_preparer=None):
        print(flush=True)
        console_data = self.serial.create_logger()
        try:
            if self.use_pdu_ac and shutil.which('netbooter') is None:
                failure_msg = "Required command 'netbooter' was not found in PATH"
                self.logger.error(failure_msg)
                raise RuntimeError(failure_msg)

            if not self.check_image_upgrade():
                return False
            if image_preparer is not None:
                tftppath = image_preparer()
            if not tftppath:
                raise RuntimeError('TFTPPATH is required')
            tftp_ip = tftppath.split(':', 1)[0]
            if not tftppath or not tftp_ip:
                raise RuntimeError('TFTPPATH and TFTPIP are required')
            if not all(ord(char) < 128 for char in tftppath):
                raise RuntimeError('TFTPPATH must contain only ASCII characters')
            error_reason = None
            for _ in range(3):
                self.enter_spi_mode()
                self.wait_spi_dhcp()
                try:
                    self.serial_exec_shell(
                        f'setenv serverip {tftp_ip}',
                        [UBOOT_PREFIX], timeout=2,
                    )
                    time.sleep(2)
                    self.serial.write(chr(3))
                    _, keywords, raw_data = self.serial_exec_shell(
                        f'tftp2emmc {tftppath}',
                        TFTP_SUCCESS + TFTP_FAILURE,
                        all_output=False,
                        timeout=EMMC_TIMEOUT,
                    )
                except RuntimeError as error:
                    self.logger.warning(
                        'TFTP sequence failed: %s; serial output tail=%r',
                        error,
                        console_data.readall()[-1000:],
                    )
                    continue
                if set(TFTP_SUCCESS) & set(keywords):
                    self.logger.info('Burn eMMC image success')
                    self.serial.write(chr(3))
                    self.serial_exec_shell(
                        '', [UBOOT_PREFIX], timeout=SPI_MODE_ENTER_TIMEOUT,
                    )
                    self.verify_flashed_version(use_pdu=self.use_pdu_ac)
                    self.reset_device(use_pdu=self.use_pdu_ac)
                    return True
                self.logger.warning(
                    'TFTP command returned without success; matched keywords=%r; '
                    'serial output tail=%r',
                    keywords,
                    raw_data[-1000:],
                )
                if 'TFTP server died' in raw_data:
                    error_reason = 'TFTP Server Died'
                    continue
                if 'T T T T T T T T' in raw_data:
                    error_reason = 'TFTP Burn eMMC Timeout'
                    time.sleep(60)
                    continue
                self.serial.write(chr(3))
            raise RuntimeError(error_reason or 'Burn eMMC image failed')
        finally:
            if not console_data.readall().strip():
                self.logger.error('Serial has no output')
            self.serial.close_logger(console_data)
