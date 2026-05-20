#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def patch_click_winconsole() -> None:
    """绕过当前 Windows 环境中 click._winconsole 导入 _ctypes 的权限问题。"""
    if "click._winconsole" in sys.modules:
        return
    module = types.ModuleType("click._winconsole")

    def _get_windows_console_stream(*args, **kwargs):
        return None

    module._get_windows_console_stream = _get_windows_console_stream
    sys.modules["click._winconsole"] = module


def main() -> int:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    patch_click_winconsole()
    from streamlit.web.cli import main as streamlit_main

    sys.argv = [
        "streamlit",
        "run",
        str(REPO_ROOT / "agent_system" / "app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        env.get("AGENT_APP_PORT", "8501"),
    ]
    return streamlit_main()


if __name__ == "__main__":
    raise SystemExit(main())
