#!/usr/bin/env python3
"""Flash an eMMC image using the Synaptics U-Boot tftp2emmc flow."""
from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
import unittest
from pathlib import Path

if __package__ in (None, ''):
    python_root = Path(__file__).resolve().parents[3]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

from torq.utils.burn.serial_wrapper import SerialThread, SerialTimeoutException


SERIAL_PORT = os.getenv('SERIAL_PORT')
EMMC_TFTPPATH = os.getenv('TFTPPATH')
EMMC_TFTPIP = os.getenv('TFTPIP')

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
LOG_FOLDER = Path(
    os.getenv('SYNA_LOG_FOLDER', str(Path(__file__).with_name('log')))
)
LOG_FORMAT = '%(asctime)s %(levelname)-8s %(message)s'

class SynaUbootTest(unittest.TestCase):
    """Run the image flashing test only; no post-flash tests are included."""

    @classmethod
    def setUpClass(cls):
        if LOG_FOLDER.is_dir():
            shutil.rmtree(LOG_FOLDER)
        LOG_FOLDER.mkdir(parents=True)
        logging.getLogger().setLevel(logging.DEBUG)
        formatter = logging.Formatter(LOG_FORMAT)

        serial_logger = logging.getLogger('serial')
        serial_handler = logging.FileHandler(LOG_FOLDER / 'serial_opt.log')
        serial_handler.setFormatter(formatter)
        serial_handler.setLevel(logging.INFO)
        serial_logger.addHandler(serial_handler)

        cls.logger = logging.getLogger('main')
        main_handler = logging.FileHandler(LOG_FOLDER / 'main.log')
        main_handler.setFormatter(formatter)
        cls.logger.addHandler(main_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        cls.logger.addHandler(stream_handler)

        cls.suite_logger = logging.getLogger('suite')
        suite_handler = logging.FileHandler(LOG_FOLDER / 'suite.log')
        suite_handler.setFormatter(formatter)
        cls.suite_logger.addHandler(suite_handler)

        cls.serial = SerialThread(SERIAL_PORT, logger=serial_logger)
        cls.serial_log = cls.serial.create_logger(str(LOG_FOLDER / 'serial.log'))

    @classmethod
    def tearDownClass(cls):
        cls.serial.close_logger(cls.serial_log)
        del cls.serial

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
            self.logger.error('Failed to enter spi mode:{}'.format(str(error)))
            raise RuntimeError('Failed to enter spi mode.') from error

    def serial_exec_shell(
        self, command, expected, repeat=False, repeat_gap=1, all_output=True,
        timeout=60, is_enter=True,
    ):
        self.logger.info('CMD: %s', command if command else '<ENTER>')
        command = command + '\n' if is_enter else command
        try:
            return self.serial.expects_for_write(
                command, expected, repeat, repeat_gap,
                all_=all_output, timeout=timeout,
            )
        except SerialTimeoutException as error:
            raise RuntimeError(
                'Timeout, no specified keyword found: {}'.format(expected)
            ) from error

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
            self.fail('Fail to get IP in SPI')
        finally:
            self.serial.close_logger(console_data)

    def test_burn_emmc(self):
        self.assertTrue(EMMC_TFTPPATH, 'TFTPPATH is required')
        self.assertTrue(EMMC_TFTPIP, 'TFTPIP is required')
        self.assertTrue(all(ord(char) < 128 for char in EMMC_TFTPPATH))
        print(flush=True)
        console_data = self.serial.create_logger()
        try:
            error_reason = None
            for _ in range(3):
                self.enter_spi_mode()
                self.wait_spi_dhcp()
                try:
                    self.serial_exec_shell(
                        'setenv serverip {}'.format(EMMC_TFTPIP),
                        [UBOOT_PREFIX], timeout=2,
                    )
                    time.sleep(2)
                    self.serial.write(chr(3))
                    _, keywords, raw_data = self.serial_exec_shell(
                        'tftp2emmc {}'.format(EMMC_TFTPPATH),
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
                    self.logger.info('Booting from eMMC')
                    self.suite_logger.info('[STEP] Boot from eMMC')
                    self.serial.write('reset\n')
                    time.sleep(5)
                    return
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
            self.fail(error_reason or 'Burn eMMC image failed')
        finally:
            if not console_data.readall().strip():
                self.logger.error('Serial has no output')
            self.serial.close_logger(console_data)


if __name__ == '__main__':
    unittest.main()
