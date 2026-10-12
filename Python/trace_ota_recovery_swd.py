"""Trace one normal OTA recovery through read-only SWD scalar samples.

The utility never writes target memory and never logs the private Wi-Fi/OTA
password.  It closes optical output before and after the run and issues no CNC
command.  Only named state-machine scalars from the exact current map are read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

from fbg_ota_protocol import OtaClient, OtaProgress, load_package
from validate_ota_hardware_stability import (
    ensure_shutter,
    load_private_wifi_password,
)


FIELDS = {
    "wifiState": (1, False),
    "wifiConnected": (1, False),
    "mqttConnected": (1, False),
    "wifiLastError": (2, False),
    "uartTxBusy": (1, False),
    "uartTxCompleteCount": (2, False),
    "uartRxByteCount": (2, False),
    "lanSocketId": (1, True),
    "lanCleanupSocketId": (1, True),
    "lanDeleteTargetId": (1, True),
    "lanClientActive": (1, False),
    "lanRawMode": (1, False),
    "lanSocketResetPending": (1, False),
    "lanCleanupCanPromote": (1, False),
    "lanCleanupOverflow": (1, False),
    "lanReadPending": (1, False),
    "lanDeleteKind": (1, False),
    "lanStartupStaleChildId": (1, True),
    "lanStartupServerStatus": (1, True),
    "lanStartupCleanupAttempts": (1, False),
    "lanServerDeleteVerifyAttempts": (1, False),
    "lanServerDeleteVerifyStartup": (1, False),
    "otaResetAfterLanDelete": (1, False),
    "otaResponseReady": (1, False),
    "otaRebootAfterResponse": (1, False),
    "publishKind": (1, False),
    "publishOverLan": (1, False),
    "bootId": (4, False),
}


def ensure_shutter_any(port: str, host: str) -> str:
    """Require an exact dark-state ACK, preferring USB before OTA."""

    return ensure_shutter(port, host=host)


def load_symbols(path: Path) -> dict[str, tuple[int, int, bool]]:
    result: dict[str, tuple[int, int, bool]] = {}
    pattern = re.compile(r"^\s*(\w+)\s+(0x[0-9a-fA-F]+)\s+Data\s+")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match and match.group(1) in FIELDS:
            name = match.group(1)
            width, signed = FIELDS[name]
            result[name] = (int(match.group(2), 16), width, signed)
    missing = sorted(set(FIELDS) - set(result))
    if missing:
        raise RuntimeError(f"Current map is missing trace symbols: {missing}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--private-config-backup", type=Path, required=True)
    parser.add_argument("--map", dest="map_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output report already exists")

    package = load_package(args.package)
    password, _expected_config = load_private_wifi_password(
        args.private_config_backup
    )
    symbols = load_symbols(args.map_file)
    report: dict[str, object] = {
        "schema": "ota_recovery_readonly_swd_trace_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": package.version_text,
        "password_logged_or_saved": False,
        "no_cnc_commands": True,
        "target_memory_writes": 0,
        "events": [],
        "progress": [],
        "complete": False,
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    report["preflight_shutter_path"] = ensure_shutter_any(args.port, args.host)
    report["preflight_shutter_verified"] = True
    save()

    client = OtaClient(args.host, password, response_timeout_s=30.0)
    upload_error: list[BaseException] = []
    upload_done = threading.Event()
    started = time.monotonic()

    def progress(item: OtaProgress) -> None:
        event = {
            "elapsed_s": round(time.monotonic() - started, 3),
            "sent": item.sent,
            "total": item.total,
            "message": item.message,
        }
        if not report["progress"] or report["progress"][-1]["message"] != item.message:
            report["progress"].append(event)
            print(json.dumps({"progress": event}, ensure_ascii=False), flush=True)

    def upload() -> None:
        try:
            client.upload(package, progress=progress)
        except BaseException as exc:
            upload_error.append(exc)
        finally:
            upload_done.set()

    session = ConnectHelper.session_with_chosen_probe(
        blocking=False,
        unique_id=args.probe,
        options={
            "target_override": "cortex_m",
            "connect_mode": "attach",
            "auto_unlock": False,
            "resume_on_disconnect": True,
            "no_config": True,
            "frequency": 1_000_000,
            "cache.enable_memory": False,
            "cache.enable_register": False,
        },
    )
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError("Exact debug probe is unavailable")

    last_state: dict[str, object] | None = None
    worker = threading.Thread(target=upload, name="normal-ota-upload", daemon=True)
    try:
        with session:
            target = session.target
            if target.get_state() != Target.State.RUNNING:
                raise RuntimeError("CPU must be running before the read-only trace")
            worker.start()
            while not upload_done.is_set():
                try:
                    state: dict[str, object] = {"cpu": str(target.get_state())}
                    for name, (address, width, signed) in symbols.items():
                        raw = bytes(target.read_memory_block8(address, width))
                        state[name] = int.from_bytes(raw, "little", signed=signed)
                    if state != last_state:
                        event = {
                            "elapsed_s": round(time.monotonic() - started, 3),
                            "state": state,
                        }
                        report["events"].append(event)
                        print(json.dumps(event), flush=True)
                        last_state = state
                except Exception as exc:
                    error_state = {"read_error": f"{type(exc).__name__}: {exc}"}
                    if error_state != last_state:
                        report["events"].append(
                            {
                                "elapsed_s": round(time.monotonic() - started, 3),
                                "state": error_state,
                            }
                        )
                        last_state = error_state
                time.sleep(0.1)
            worker.join(timeout=5.0)
            report["cpu_state_after"] = str(target.get_state())
        if upload_error:
            raise upload_error[0]
        boot_ids = [
            event["state"]["bootId"]
            for event in report["events"]
            if isinstance(event.get("state"), dict)
            and "bootId" in event["state"]
        ]
        report["source_boot_id"] = boot_ids[0] if boot_ids else None
        report["final_boot_id"] = boot_ids[-1] if boot_ids else None
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            report["final_shutter_path"] = ensure_shutter_any(
                args.port, args.host
            )
            report["final_shutter_verified"] = True
        except Exception as exc:
            report["final_shutter_verified"] = False
            report["final_shutter_error"] = f"{type(exc).__name__}: {exc}"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(
            json.dumps(
                {
                    "complete": report["complete"],
                    "source_boot_id": report.get("source_boot_id"),
                    "final_boot_id": report.get("final_boot_id"),
                    "final_shutter_verified": report.get(
                        "final_shutter_verified", False
                    ),
                    "report": str(args.output),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
