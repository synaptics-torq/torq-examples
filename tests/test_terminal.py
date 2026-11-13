# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""utils.terminal must not need the ``tty`` module (absent on some board images)."""

import builtins
import importlib
import os
import sys

import pytest

termios = pytest.importorskip("termios")


def _import_terminal_without(monkeypatch, *names):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in names:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "utils.terminal", raising=False)
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("utils.terminal")


def test_imports_without_tty(monkeypatch):
    terminal = _import_terminal_without(monkeypatch, "tty")
    assert terminal.termios is not None


def test_without_termios_stop_watching_is_off(monkeypatch):
    terminal = _import_terminal_without(monkeypatch, "termios", "tty")
    r, w = os.pipe()
    try:
        with open(r, "rb", buffering=0) as stream:
            with terminal.InferenceStopInput(stream) as stop:
                assert stop() is False
    finally:
        os.close(w)


def test_cbreak_matches_tty_setcbreak():
    import pty
    import tty

    master, slave = pty.openpty()
    try:
        original = termios.tcgetattr(slave)
        tty.setcbreak(slave)
        expected = termios.tcgetattr(slave)
        termios.tcsetattr(slave, termios.TCSANOW, original)

        from utils.terminal import _set_cbreak

        _set_cbreak(slave)
        assert termios.tcgetattr(slave) == expected
    finally:
        os.close(master)
        os.close(slave)
