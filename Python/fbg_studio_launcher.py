"""Console-free launcher support for FBG OptiSense Studio."""

from __future__ import annotations

import datetime as _datetime
import os
from pathlib import Path
import sys
import traceback

from runtime_paths import ensure_client_data, output_path, resource_root

APP_DIR = resource_root()
LOG_PATH = output_path("logs", "fbg_studio_startup.log")
_LOG_STREAM = None


def _load_user_environment() -> None:
    """Refresh credentials that Explorer may not yet have inherited."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in ("FBG_MQTT_USERNAME", "FBG_MQTT_PASSWORD"):
                if os.environ.get(name):
                    continue
                try:
                    value, _value_type = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                if value:
                    os.environ[name] = str(value)
    except (ImportError, OSError):
        # The app still starts without MQTT credentials and reports connection
        # status in its own UI.
        pass


def _configure_logging() -> Path:
    global _LOG_STREAM
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG_STREAM = LOG_PATH.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _LOG_STREAM
    sys.stderr = _LOG_STREAM
    return LOG_PATH


def _show_startup_error(log_path: Path) -> None:
    message = (
        "FBG OptiSense Studio 启动失败。\n\n"
        f"详细错误已写入：\n{log_path}"
    )
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            None, message, "FBG OptiSense Studio", 0x10
        )
    except Exception:
        pass


def main(argv=None) -> int:
    os.chdir(APP_DIR)
    ensure_client_data()
    _load_user_environment()
    log_path = _configure_logging()
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    timestamp = _datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"\n[{timestamp}] FBG OptiSense Studio starting; argv={effective_argv!r}")
    try:
        from fbg_unified_app import main as app_main

        return int(app_main(effective_argv) or 0)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        _show_startup_error(log_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
