"""Single-command launcher for the Qt face enrollment application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Face_ID.src.infer import main


if __name__ == "__main__":
    # Ensure Qt starts with UTF-8 locale and expected board Wayland session.
    os.environ["LC_ALL"] = "en_US.utf8"
    os.environ["LANG"] = "en_US.utf8"
    os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
    os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"
    os.environ["WAYLAND_DISPLAY"] = "wayland-1"
    os.environ["QT_QPA_PLATFORM"] = "wayland"

    main()
