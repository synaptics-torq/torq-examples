# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
import select

try:
    import termios
except ImportError:  # no POSIX terminal control: Ctrl+C/D watching is off
    termios = None


def _set_cbreak(fd: int) -> None:
    """Put the terminal in cbreak mode: keys arrive one by one, unechoed.

    Same as ``tty.setcbreak``, done with termios directly: some minimal board
    images ship a Python without the ``tty`` module.
    """
    mode = termios.tcgetattr(fd)
    mode[3] &= ~(termios.ECHO | termios.ICANON)  # lflag
    mode[6][termios.VMIN] = 1
    mode[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSAFLUSH, mode)


class InferenceStopInput:
    """Watch for Ctrl+C/D while inference is running."""

    _CTRL_C = b"\x03"
    _CTRL_D = b"\x04"

    def __init__(self, stream):
        self._stream = stream
        self._fd: int | None = None
        self._old_attrs = None
        self._stopped = False

    def __enter__(self):
        try:
            fd = self._stream.fileno()
        except (AttributeError, OSError, ValueError):
            return self
        if termios is None or not os.isatty(fd):
            return self
        try:
            self._old_attrs = termios.tcgetattr(fd)
            _set_cbreak(fd)
        except termios.error:
            self._old_attrs = None
            return self
        self._fd = fd
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._old_attrs is not None and self._fd is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            except termios.error:
                pass
        return False

    def __call__(self) -> bool:
        if self._stopped:
            return True
        if self._fd is None:
            return False
        while True:
            try:
                readable, _, _ = select.select([self._fd], [], [], 0)
            except (OSError, ValueError):
                return False
            if not readable:
                return self._stopped
            try:
                data = os.read(self._fd, 1024)
            except BlockingIOError:
                return self._stopped
            except OSError:
                return False
            if not data or self._CTRL_C in data or self._CTRL_D in data:
                self._stopped = True
                return True
